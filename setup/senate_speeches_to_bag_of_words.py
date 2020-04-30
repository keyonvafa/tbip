"""Convert Senate speech data from 114th Congress to bag of words format.

The data is provided by [1]. Specifically, we use the `hein-daily` data. To 
run this script, make sure the relevant files are in 
`text-based-ideal-points/data/senate-speeches-114/raw/`. The files needed for
this script are `speeches_114.txt`, `descr_114.txt`, and `114_SpeakerMap.txt`.

#### References
[1]: Gentzkow, Matthew, Jesse M. Shapiro, and Matt Taddy. Congressional Record 
     for the 43rd-114th Congresses: Parsed Speeches and Phrase Counts. Palo 
     Alto, CA: Stanford Libraries [distributor], 2018-01-16. 
     https://data.stanford.edu/congress_text
"""

import os
import setup_utils as utils

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

project_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)) 
data_dir = os.path.join(project_dir, "data/senate-speeches-114/raw")
save_dir = os.path.join(project_dir, "data/senate-speeches-114/clean")

speeches = pd.read_csv(os.path.join(data_dir, 'speeches_114.txt'), 
                       encoding="ISO-8859-1", 
                       sep="|",
                       error_bad_lines=False)
description = pd.read_csv(os.path.join(data_dir, 'descr_114.txt'), 
                          encoding="ISO-8859-1", 
                          sep="|")
speaker_map = pd.read_csv(os.path.join(data_dir, '114_SpeakerMap.txt'), 
                          encoding="ISO-8859-1", 
                          sep="|") 

# Merge all data into a single dataframe.
merged_df = speeches.merge(description, 
                           left_on='speech_id', 
                           right_on='speech_id')
df = merged_df.merge(speaker_map, left_on='speech_id', right_on='speech_id')

# Only look at senate speeches.
senate_df = df[df['chamber_x'] == 'S']
speaker = np.array(
    [' '.join([first, last]) for first, last in 
     list(zip(np.array(senate_df['firstname']), 
              np.array(senate_df['lastname'])))])
speeches = np.array(senate_df['speech'])
party = np.array(senate_df['party'])

# Remove senators who make less than 24 speeches.
min_speeches = 24
unique_speaker, speaker_counts = np.unique(speaker, return_counts=True)
absent_speakers = unique_speaker[np.where(speaker_counts < min_speeches)]
absent_speaker_inds = [ind for ind, x in enumerate(speaker) 
                       if x in absent_speakers]
speaker = np.delete(speaker, absent_speaker_inds)
speeches = np.delete(speeches, absent_speaker_inds)
party = np.delete(party, absent_speaker_inds)
speaker_party = np.array(
    [speaker[i] + " (" + party[i] + ")" for i in range(len(speaker))])

# Create mapping between names and IDs.
speaker_to_speaker_id = dict(
    [(y.title(), x) for x, y in enumerate(sorted(set(speaker_party)))])
author_indices = np.array(
    [speaker_to_speaker_id[s.title()] for s in speaker_party])
author_map = np.array(list(speaker_to_speaker_id.keys()))

stopwords = set(
    np.loadtxt(os.path.join(project_dir, 
                            "setup/stopwords/senate_speeches.txt"),
               dtype=str,
               delimiter="\n"))
count_vectorizer = CountVectorizer(min_df=0.001,
                                   max_df=0.3, 
                                   stop_words=stopwords, 
                                   ngram_range=(1, 3),
                                   token_pattern="[a-zA-Z]+")
# Learn initial document term matrix. This is only initial because we use it to
# identify words to exclude based on author counts.
counts = count_vectorizer.fit_transform(speeches)
vocabulary = np.array(
    [k for (k, v) in sorted(count_vectorizer.vocabulary_.items(), 
                            key=lambda kv: kv[1])])

# Remove phrases spoken by less than 10 Senators.
counts_per_author = utils.bincount_2d(author_indices, counts.toarray())
min_authors_per_word = 10
author_counts_per_word = np.sum(counts_per_author > 0, axis=0)
acceptable_words = np.where(
    author_counts_per_word >= min_authors_per_word)[0]

# Fit final document-term matrix with modified vocabulary.
count_vectorizer = CountVectorizer(ngram_range=(1, 3),
                                   vocabulary=vocabulary[acceptable_words])
counts = count_vectorizer.fit_transform(speeches)
vocabulary = np.array(
    [k for (k, v) in sorted(count_vectorizer.vocabulary_.items(), 
                            key=lambda kv: kv[1])])

# Adjust counts by removing unigram/n-gram pairs which co-occur.
counts_dense = utils.remove_cooccurring_ngrams(counts, vocabulary)

# Remove speeches with no words.
existing_speeches = np.where(np.sum(counts_dense, axis=1) > 0)[0]
counts_dense = counts_dense[existing_speeches]
author_indices = author_indices[existing_speeches]

# Save data.
if not os.path.exists(save_dir):
  os.makedirs(save_dir)

# `counts.npz` is a [num_documents, num_words] sparse matrix containing the
# word counts for each document.
sparse.save_npz(os.path.join(save_dir, "counts.npz"),
                sparse.csr_matrix(counts_dense).astype(np.float32))
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
raw_documents = [document.replace("\n", ' ').replace("\r", ' ') 
                 for document in speeches[existing_speeches]]
np.savetxt(os.path.join(save_dir, "raw_documents.txt"), 
           raw_documents, 
           fmt="%s")
