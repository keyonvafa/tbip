"""Convert Senate tweets from 114th Congress to bag of words format.

The data is provided by Voxgov [1]. We make sure Senate speeches have been 
processed before running this script, because we use the author list from 
Senate speeches to align the two corpora.

We also save collapsed counts so it is in the format used by Wordfish. That is,
we sum over each tweet for a given Senator so the count matrix has shape
[num_authors, num_words]. We save this in a file called `wordfish_counts.npz`.

#### References
[1]: VoxGovFEDERAL, U.S. Senators tweets from the 114th Congress. 2020. 
     www.voxgov.com.  
"""

import os

import numpy as np
import pandas as pd
from scipy import sparse
import setup_utils as utils
from sklearn.feature_extraction.text import CountVectorizer

project_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)) 
data_dir = os.path.join(project_dir, "data/senate-tweets-114/raw")
save_dir = os.path.join(project_dir, "data/senate-tweets-114/clean")

tweets = pd.read_csv(os.path.join(data_dir, "voxgov_tweets.csv"))

# Only use tweets in 114th Senate.
tweets = tweets[tweets.date >= "2015-01-03"]
tweets = tweets[tweets.date < "2017-01-03"]

# Remove tweets that don't from a Senator account.
twitter_authors = tweets['source'].unique()
twitter_authors = twitter_authors[twitter_authors != "U.S. Senate"]
twitter_authors = twitter_authors[twitter_authors != 
                                  "Senate Republican Conference"]
twitter_authors = twitter_authors[twitter_authors != 
                                  "Senate Majority Leader - Mitch McConnell"]

# make lower case.
twitter_lower = np.array([author.lower() for author in twitter_authors])
twitter_last_names = np.array([name.split('sen.')[1].split(",")[0][1:] 
                               for name in twitter_lower])
# We assume we already have senate _speeches_ preprocessed. We use the 
# corresponding author map to find the relevant tweeters.
senate_speech_data_dir = os.path.join(project_dir, 
                                      "data/senate-speeches-114/clean")
speech_author_map = np.loadtxt(
    os.path.join(senate_speech_data_dir, "author_map.txt"),
    dtype=str,
    delimiter="\n")

twitter_name_to_speech_author = {}
for speech_author in speech_author_map:
  last_name = speech_author.split(' ')[-2].lower()
  matches = np.where(twitter_last_names == last_name)[0]
  if len(matches) == 1:
    twitter_name_to_speech_author[
        twitter_authors[matches[0]]] = speech_author
  else:
    print("Found {} matches for {}.".format(len(matches), speech_author))

matched_tweet_authors = twitter_authors[
    [name in list(twitter_name_to_speech_author.keys()) 
     for name in twitter_authors]]
assert(len(matched_tweet_authors) == 
       len(list(twitter_name_to_speech_author.keys())))
print("Matched {:.1f}% of speakers to tweeters".format(
    100. * len(matched_tweet_authors) / len(speech_author_map)))

matched_df = tweets[tweets['source'].isin(matched_tweet_authors)]
matched_tweet_authors = np.array([
    twitter_name_to_speech_author[twitter_name]
    for twitter_name in np.array(matched_df['source'])])
tweets = np.array(matched_df['title'])

senator_to_senator_id = dict(
    [(y.title(), x) for x, y in 
     enumerate(sorted(set(matched_tweet_authors)))])
author_indices = np.array([senator_to_senator_id[s.title()] 
                           for s in matched_tweet_authors])
author_map = np.array(list(senator_to_senator_id.keys()))

stopwords = set(np.loadtxt(os.path.join(project_dir, 
                                        "setup/stopwords/senate_tweets.txt"),
                dtype=str,
                delimiter="\n"))
count_vectorizer = CountVectorizer(min_df=0.0005,
                                   max_df=0.3,
                                   ngram_range=(1, 3),
                                   stop_words=stopwords,
                                   token_pattern="[a-zA-Z#]+")
counts = count_vectorizer.fit_transform(tweets)
vocabulary = np.array(
    [k for (k, v) in sorted(count_vectorizer.vocabulary_.items(),
                            key=lambda kv: kv[1])])

# Remove phrases used by less than 10 Senators.
counts_per_author = utils.bincount_2d(author_indices, counts.toarray())
min_authors_per_word = 10
author_counts_per_word = np.sum(counts_per_author > 0, axis=0)
acceptable_words = np.where(
    author_counts_per_word >= min_authors_per_word)[0]

# Fit final document-term matrix with new vocabulary.
count_vectorizer = CountVectorizer(ngram_range=(1, 3),
                                   vocabulary=vocabulary[acceptable_words],
                                   token_pattern="[a-zA-Z#]+")
counts = count_vectorizer.fit_transform(tweets)
vocabulary = np.array(
    [k for (k, v) in sorted(count_vectorizer.vocabulary_.items(), 
                            key=lambda kv: kv[1])])

# Adjust counts by removing unigram/n-gram pairs which co-occur.
counts_dense = utils.remove_cooccurring_ngrams(counts, vocabulary)

# Remove tweets with no words.
existing_tweets = np.where(np.sum(counts_dense, axis=1) > 0)[0]
counts_dense = counts_dense[existing_tweets]
author_indices = author_indices[existing_tweets]
counts = sparse.csr_matrix(counts_dense).astype(np.float32)

# Save data.
if not os.path.exists(save_dir):
  os.makedirs(save_dir)

# `counts.npz` is a [num_documents, num_words] sparse matrix containing the
# word counts for each document.
sparse.save_npz(os.path.join(save_dir, "counts.npz"), counts)
# `author_indices.npy` is a [num_documents] vector where each entry is an
# integer indicating the author of the corresponding document.
np.save(os.path.join(save_dir, "author_indices.npy"), author_indices)
# `vocabulary.txt` is a [num_words] vector where each entry is a string
# denoting the corresponding word in the vocabulary.
np.savetxt(os.path.join(save_dir, "vocabulary.txt"), vocabulary, fmt="%s")
# `author_map.txt` is a [num_authors] vector of strings providing the name of
# each author in the corpus.
np.savetxt(os.path.join(save_dir, "author_map.txt"), author_map, fmt="%s")
# `raw_documents.txt` contains all the documents we ended up using.
stripped_tweets = [tweet.replace("\n", ' ').replace("\r", ' ') 
                   for tweet in tweets[existing_tweets]]
np.savetxt(os.path.join(save_dir, "raw_documents.txt"), 
           stripped_tweets, 
           fmt="%s")

# Collapse by authors to get in Wordfish format. 
per_author_counts = utils.bincount_2d(author_indices, counts.toarray())
sparse.save_npz(os.path.join(save_dir, "wordfish_counts.npz"),
                sparse.csr_matrix(per_author_counts).astype(np.float32))
