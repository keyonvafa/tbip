"""Preprocess Senate speeches using method described in Wordshoal paper.

This script is a Python version of the R script by Benjamin E. Lauderdale and 
Alexander Herzog used for Wordshoal [1]. Specifically, the data and script are 
located at [2].

The specific files needed for this script are: `speaker_senator_link_file.csv`
and `speeches_Senate_{senate_session}.tab`, where {senate_session} is the 
Senate session number (e.g. 113). These files should be in 
`senate-speech-comparisons/raw/`.

### References
[1] Benjamin E. Lauderdale and Alexander Herzog. Measuring Political Positions
    from Legislative Speech. In _Political Analysis_, 2016.
    
[2] Benjamin E. Lauderdale and Alexander Herzog. Replication Data for: 
    Measuring Political Positions from Legislative Speech. In _Harvard 
    Dataverse_, 2016. https://doi.org/10.7910/DVN/RQMIV3.
"""

import os

from absl import app
from absl import flags
import numpy as np
import pandas as pd
from scipy import sparse
import setup_utils as utils
from sklearn.feature_extraction.text import CountVectorizer


flags.DEFINE_integer("senate_session",
                     default=113,
                     help="Senate session (used only for senate speeches).")
FLAGS = flags.FLAGS


def main(argv):
  del argv
  senate_session = FLAGS.senate_session
  print("Preprocessing session {}...".format(senate_session))
  project_dir = os.path.abspath(
      os.path.join(os.path.dirname(__file__), os.pardir)) 
  data_dir = os.path.join(project_dir, "data/senate-speech-comparisons/raw")
  tbip_save_dir = os.path.join(
      project_dir, 
      "data/senate-speech-comparisons/tbip/{}/clean".format(senate_session))
  wordfish_save_dir = os.path.join(
      project_dir, 
      "data/senate-speech-comparisons/wordfish/{}/clean".format(
          senate_session))
  wordshoal_save_dir = os.path.join(
      project_dir, 
      "data/senate-speech-comparisons/wordshoal/{}/clean".format(
          senate_session))

  senators = pd.read_csv(os.path.join(data_dir, 
                                      'speaker_senator_link_file.csv'))
  senators['speaker_name'] = (
      senators['first.name'] + " " + 
      senators['last.name'] + 
      " (" + senators['party'] + ")")

  speeches = pd.read_csv(
      os.path.join(data_dir, 'speeches_Senate_{}.tab'.format(senate_session)),
      encoding="UTF-8",
      sep="\t")
  speeches = speeches[speeches['chamber'] == 'Senate']
  speeches = speeches[speeches['congress'] == senate_session]

  speeches['month_number'] = np.nan
  speeches.loc[speeches['month'] == 'January', 'month_number'] = '01'
  speeches.loc[speeches['month'] == "January", 'month_number'] = "01"
  speeches.loc[speeches['month'] == "February", 'month_number'] = "02"
  speeches.loc[speeches['month'] == "March", 'month_number'] = "03"
  speeches.loc[speeches['month'] == "April", 'month_number'] = "04"
  speeches.loc[speeches['month'] == "May", 'month_number'] = "05"
  speeches.loc[speeches['month'] == "June", 'month_number'] = "06"
  speeches.loc[speeches['month'] == "July", 'month_number'] = "07"
  speeches.loc[speeches['month'] == "August", 'month_number'] = "08"
  speeches.loc[speeches['month'] == "September", 'month_number'] = "09"
  speeches.loc[speeches['month'] == "October", 'month_number'] = "10"
  speeches.loc[speeches['month'] == "November", 'month_number'] = "11"
  speeches.loc[speeches['month'] == "December", 'month_number'] = "12"

  speeches['day_number'] = speeches['day'].apply(lambda x: str(x).zfill(2))
  speeches['date'] = (
      speeches['year'].astype(str) + "-" + 
      speeches['month_number'] + "-" + speeches['day_number'])

  # Remove speeches with less than 50 words.
  speeches['word_count'] = speeches['speech'].str.split().apply(len)
  speeches = speeches[speeches['word_count'] > 50]

  speeches['speaker'] = speeches['speaker'].str.lower()
  # Remove punctuation.
  speeches['speaker'] = speeches['speaker'].replace(r"[^\w\s]", "", regex=True)
  # Remove leading and trailing spaces
  speeches['speaker'] = speeches['speaker'].replace("^\\s+", "", regex=True)
  speeches['speaker'] = speeches['speaker'].replace("^\\s+$", "", regex=True)

  # Remove speakers whose name contains "presiding officer".
  speeches = speeches[~speeches['speaker'].str.contains('presiding officer')]
  # Remove the acting president pro tempore because we don't know who this is.
  speeches = speeches[
      ~speeches['speaker'].str.contains('acting president pro tempore')]

  # Code correct for president pro tempore.
  if senate_session in [104, 105, 106]:
    speeches.loc[speeches['speaker'].str.contains(
        'president pro tempore'), 'speaker'] = 'mr thurmond'
  elif senate_session == 107:
    speeches.loc[
        ((speeches['date'] <= "2001-01-20") | 
         (speeches['date'] >= "2001-06-07")) &
        (speeches['speaker'].str.contains('president pro tempore')), 
        'speaker'] = 'mr byrd'
    speeches.loc[
        (speeches['date'] < "2001-06-07") & 
        (speeches['date'] < "2001-06-07") &
        (speeches['speaker'].str.contains('president pro tempore')), 
        'speaker'] = 'mr thurmond'
  elif senate_session in [108, 109]:
    speeches.loc[speeches['speaker'].str.contains(
        'president pro tempore'), 'speaker'] = 'mr stevens'
  elif senate_session == 110:
    speeches.loc[speeches['speaker'].str.contains(
        'president pro tempore'), 'speaker'] = 'mr byrd'
  elif senate_session == 111:
    speeches.loc[
        (speeches['date'] <= "2010-06-28") & 
        (speeches['speaker'].str.contains('president pro tempore')), 
        'speaker'] = 'mr byrd'
    speeches.loc[
        (speeches['date'] > "2010-06-28") & 
        (speeches['speaker'].str.contains('president pro tempore')), 
        'speaker'] = 'mr inouye'
  elif senate_session == 112:
    speeches.loc[
        (speeches['date'] <= "2012-12-17") & 
        (speeches['speaker'].str.contains('president pro tempore')), 
        'speaker'] = 'mr inouye'
    speeches.loc[
        (speeches['date'] > "2012-12-18") & 
        (speeches['speaker'].str.contains('president pro tempore')), 
        'speaker'] = 'mr leahy'
  elif senate_session == 113:
    speeches.loc[speeches['speaker'].str.contains(
        'president pro tempore'), 'speaker'] = 'mr leahy'
  else:
    raise ValueError("Haven't implemented president matching for this session")

  # Remove speakers whose name contains "speaker pro tempore".
  speeches = speeches[~speeches['speaker'].str.contains('speaker pro tempore')]

  # Remove speakers who address the chair.
  speeches = speeches[~speeches['speaker'].str.contains('addressed the chair')]
  speeches = speeches[~speeches['speaker'].str.contains('addressed to chair')]
   
  # Remove speeches that have two speakers.
  speeches = speeches[~speeches['speaker'].str.contains(' and ')]

  # Remove speakers that are not correctly parsed by removing names with
  # more than 5 words.
  speeches['speaker_word_count'] = speeches['speaker'].str.split().apply(len)
  speeches = speeches[speeches['speaker_word_count'] <= 5]

  # Remove president, vice president, and chief justice.
  invalid_speakers = ["mr president", "mr vice president", 
                      "mr chief justice", "the president", 
                      "the vice president", "the chief justice"]
  speeches = speeches[~speeches['speaker'].isin(invalid_speakers)]

  # Remove some other incorrect speaker names.
  speeches = speeches[~speeches['speaker'].str.contains(' received ')]
  speeches = speeches[~speeches['speaker'].str.contains(" gave ")]
  speeches = speeches[~speeches['speaker'].str.contains(" for ")]
  speeches = speeches[~speeches['speaker'].str.contains(" is not ")]
  speeches = speeches[~speeches['speaker'].str.contains(" exactly ")]
  speeches = speeches[~speeches['speaker'].str.contains(" earned ")]
  speeches = speeches[~speeches['speaker'].str.contains(" said")]
  speeches = speeches[~speeches['speaker'].str.contains("deserves better")]
  speeches = speeches[~speeches['speaker'].str.contains("is currently")]
  speeches = speeches[~speeches['speaker'].str.contains("notes that")]
  speeches = speeches[~speeches['speaker'].str.contains("summoned mr")]
  speeches = speeches[~speeches['speaker'].str.contains("is correct")]
  speeches = speeches[~speeches['speaker'].str.contains("as will i")]
  speeches = speeches[~speeches['speaker'].str.contains("has a job")]
  speeches = speeches[~speeches['speaker'].str.contains(
      "overcame tremendous odds")]
  speeches = speeches[~speeches['speaker'].str.contains("made that point")]
  speeches = speeches[~speeches['speaker'].str.contains("was incredulous")]
  speeches = speeches[~speeches['speaker'].str.contains("is right")]
  speeches = speeches[~speeches['speaker'].str.contains("attended the u")]
  speeches = speeches[~speeches['speaker'].str.contains("absolutely not")]
  speeches = speeches[~speeches['speaker'].str.contains("had it correct")]
  speeches = speeches[~speeches['speaker'].str.contains("amendment no")]
  speeches = speeches[~speeches['speaker'].str.contains("is a veteran")]
  speeches = speeches[~speeches['speaker'].str.contains(
      "demolished that argument")]
  speeches = speeches[~speeches['speaker'].str.contains("begged dr")]
  speeches = speeches[~speeches['speaker'].str.contains("is retired")]
  speeches = speeches[~speeches['speaker'].str.contains("died yesterday")]
  speeches = speeches[~speeches['speaker'].str.contains("explains that")]
  speeches = speeches[~speeches['speaker'].str.contains("quotes")]
  speeches = speeches[~speeches['speaker'].str.contains("visits the")]
  speeches = speeches[~speeches['speaker'].str.contains(
      "dismissed his argument")]
  speeches = speeches[~speeches['speaker'].str.contains("has neither")]
  speeches = speeches[~speeches['speaker'].str.contains("has two amendments")]
  speeches = speeches[~speeches['speaker'].str.contains("is tough")]
  speeches = speeches[~speeches['speaker'].str.contains("quit")]
  speeches = speeches[~speeches['speaker'].str.contains("did encourage")]
  speeches = speeches[~speeches['speaker'].str.contains("should take note")]
  speeches = speeches[~speeches['speaker'].str.contains("met with")]
  speeches = speeches[~speeches['speaker'].str.contains("has two children")]
  speeches = speeches[~speeches['speaker'].str.contains("shared that")]
  speeches = speeches[~speeches['speaker'].str.contains("denies that")]
  speeches = speeches[~speeches['speaker'].str.contains(
      "is eminently qualified")]
  speeches = speeches[~speeches['speaker'].str.contains("again called")]
  speeches = speeches[~speeches['speaker'].str.contains("went by")]
  speeches = speeches[~speeches['speaker'].str.contains("has a")]
  speeches = speeches[~speeches['speaker'].str.contains("hit over")]
  speeches = speeches[~speeches['speaker'].str.contains("passed away")]
  speeches = speeches[~speeches['speaker'].str.contains(
      "remains the exception")]
  speeches = speeches[~speeches['speaker'].str.contains("is a nurse")]
  speeches = speeches[~speeches['speaker'].str.contains("apparently told")]
  speeches = speeches[~speeches['speaker'].str.contains("der followed")]
  speeches = speeches[~speeches['speaker'].str.contains("is a democrat")]
  speeches = speeches[~speeches['speaker'].str.contains("comes in")]
  speeches = speeches[~speeches['speaker'].str.contains(
      "epitomized these qualities")]
  speeches = speeches[~speeches['speaker'].str.contains("replied that")]
  speeches = speeches[~speeches['speaker'].str.contains("criticizing")]
  speeches = speeches[~speeches['speaker'].str.contains("is a politician")]
  speeches = speeches[~speeches['speaker'].str.contains("states that mr")]
  speeches = speeches[~speeches['speaker'].str.contains("has recused himself")]
  speeches = speeches[~speeches['speaker'].str.contains("found numerous")]
  speeches = speeches[~speeches['speaker'].str.contains("is a laywer")]
  speeches = speeches[~speeches['speaker'].str.contains(
      "was recess-appointed")]
  speeches = speeches[~speeches['speaker'].str.contains("acknowledges it")]
  speeches = speeches[~speeches['speaker'].str.contains("understands this")]
  speeches = speeches[~speeches['speaker'].str.contains("knows the risks")]
  speeches = speeches[~speeches['speaker'].str.contains("also testified mr")]
  speeches = speeches[~speeches['speaker'].str.contains("joined the u")]

  # Prepare to merge senator data to speeches.
  session_senators = senators[senators['congress'] == senate_session]
  session_senators = session_senators.drop(columns=['congress'])

  # Merge.
  speeches = speeches.merge(session_senators, on="speaker")

  # Remove speakers with null last names.
  speeches = speeches[~pd.isnull(speeches["last.name"])]

  # Make titles lowercase
  speeches['title'] = speeches['title'].str.lower()

  # Remove procedural speeches.
  speeches = speeches[~speeches['title'].str.contains("prayer")]
  speeches = speeches[~speeches['title'].str.contains("order of")]
  speeches = speeches[~speeches['title'].str.contains("orders of")]
  speeches = speeches[~speeches['title'].str.contains("order for")]
  speeches = speeches[~speeches['title'].str.contains("orders for")]
  speeches = speeches[~speeches['title'].str.contains("schedule")]
  speeches = speeches[~speeches['title'].str.contains("tribute")]
  speeches = speeches[~speeches['title'].str.contains("remembering")]
  speeches = speeches[~speeches['title'].str.contains("recess")]
  speeches = speeches[~speeches['title'].str.contains("the calendar")]
  speeches = speeches[~speeches['title'].str.contains("vote schedule")]
  speeches = speeches[~speeches['title'].str.contains("notice of hearing")]
  speeches = speeches[~speeches['title'].str.contains("change of vote")]
  speeches = speeches[~speeches['title'].str.contains("----")]
  speeches = speeches[speeches['title'] != "0"]
  speeches = speeches[speeches['title'] != "* * * * *"]

  # Create count matrix where we don't collapse by debates. We will be using
  # this data for the TBIP.
  speeches['speaker_name'] = speeches['speaker_name'].astype('category')
  speeches['speaker_id'] = speeches['speaker_name'].cat.codes
  stopwords = set(np.loadtxt(
      os.path.join(project_dir, 
                   "setup/stopwords/senate_speech_comparisons.txt"),
      dtype=str,
      delimiter="\n"))
  count_vectorizer = CountVectorizer(stop_words=stopwords)
  uncollapsed_counts = count_vectorizer.fit_transform(speeches['speech'])
  uncollapsed_vocabulary = np.array(
      [k for (k, v) in sorted(count_vectorizer.vocabulary_.items(), 
                              key=lambda kv: kv[1])])
  uncollapsed_author_indices = np.array(
      speeches['speaker_id']).astype(np.int32)
  author_id_to_author = dict(
      enumerate(speeches['speaker_name'].cat.categories))
  uncollapsed_author_map = np.array(list(author_id_to_author.values()))

  # Save uncollapsed results
  if not os.path.exists(tbip_save_dir):
    os.makedirs(tbip_save_dir)

  sparse.save_npz(os.path.join(tbip_save_dir, "counts.npz"),
                  uncollapsed_counts.astype(np.float32))
  np.save(os.path.join(tbip_save_dir, "author_indices.npy"), 
          uncollapsed_author_indices)
  np.savetxt(os.path.join(tbip_save_dir, "vocabulary.txt"), 
             uncollapsed_vocabulary, 
             fmt="%s")
  np.savetxt(os.path.join(tbip_save_dir, "author_map.txt"), 
             uncollapsed_author_map, 
             fmt="%s")

  # Collapse by authors to get in Wordfish format. 
  per_author_counts = utils.bincount_2d(uncollapsed_author_indices, 
                                        uncollapsed_counts.toarray())
  if not os.path.exists(wordfish_save_dir):
    os.makedirs(wordfish_save_dir)

  sparse.save_npz(os.path.join(wordfish_save_dir, "counts.npz"),
                  sparse.csr_matrix(per_author_counts).astype(np.float32))
  np.savetxt(os.path.join(wordfish_save_dir, "vocabulary.txt"), 
             uncollapsed_vocabulary, 
             fmt="%s")
  np.savetxt(os.path.join(wordfish_save_dir, "author_map.txt"), 
             uncollapsed_author_map, 
             fmt="%s")

  # Remove by debates.
  # A 'debate' is here defined as a set of speeches with the same title and 
  # date.
  speeches['title_date'] = (
      speeches['title'] + '_' + speeches['date']).astype('category')
  speeches['debate_id'] = speeches['title_date'].cat.codes

  # Only keep debates with at least 5 speakers.
  debate_size_df = speeches.groupby(
      'debate_id')['speaker'].nunique().reset_index(name="num_debate_speakers")
  speeches = speeches.merge(debate_size_df, 
                            left_on='debate_id', 
                            right_on='debate_id')
  speeches = speeches[speeches["num_debate_speakers"] >= 5]

  # Rename debate ID so dropped debate IDs are refilled.
  speeches['title_date'] = (
      speeches['title'] + '_' + speeches['date']).astype('category')
  speeches['debate_id'] = speeches['title_date'].cat.codes

  speeches['speaker_name_collapsed'] = speeches['speaker_name'].astype(
      'category')
  speeches['speaker_id'] = speeches['speaker_name_collapsed'].cat.codes

  # Collapse speeches for each Senator in each session.
  count_vectorizer = CountVectorizer(stop_words=stopwords)
  collapsed_counts = count_vectorizer.fit_transform(speeches['speech'])
  collapsed_vocabulary = np.array(
      [k for (k, v) in sorted(count_vectorizer.vocabulary_.items(), 
                              key=lambda kv: kv[1])])

  author_indices = np.array(speeches['speaker_id']).astype(np.int32)
  debate_indices = np.array(speeches['debate_id']).astype(np.int32)

  # Generate unique ID for each (author, debate) pair so we can collapse.
  author_and_debate_to_id = {}
  count = 0
  author_debate_indices = np.zeros(len(author_indices))
  for i in range(len(author_indices)):
    existing_keys = list(author_and_debate_to_id.keys())
    if (author_indices[i], debate_indices[i]) not in existing_keys:
      author_and_debate_to_id[(author_indices[i], debate_indices[i])] = count
      count += 1
    author_debate_indices[i] = author_and_debate_to_id[
        (author_indices[i], debate_indices[i])]
  author_debate_indices = author_debate_indices.astype(np.int32)

  collapsed_counts = utils.bincount_2d(author_debate_indices, 
                                       collapsed_counts.toarray())
  sorted_author_debate = sorted(author_and_debate_to_id, 
                                key=author_and_debate_to_id.get)
  collapsed_author_indices, collapsed_debate_indices = list(
      zip(*sorted_author_debate))
  collapsed_debate_indices = np.array(collapsed_debate_indices)
  collapsed_author_indices = np.array(collapsed_author_indices)

  author_id_to_author = dict(
      enumerate(speeches['speaker_name'].cat.categories))
  collapsed_author_map = np.array(list(author_id_to_author.values()))
  debate_id_to_debate = dict(enumerate(speeches['title_date'].cat.categories))
  collapsed_debate_map = np.array(list(debate_id_to_debate.values()))

  # Save Wordshoal results.
  if not os.path.exists(wordshoal_save_dir):
    os.makedirs(wordshoal_save_dir)

  sparse.save_npz(os.path.join(wordshoal_save_dir, "counts.npz"),
                  sparse.csr_matrix(collapsed_counts).astype(np.float32))
  np.save(os.path.join(wordshoal_save_dir, "author_indices.npy"), 
          collapsed_author_indices)
  np.save(os.path.join(wordshoal_save_dir, "debate_indices.npy"), 
          collapsed_debate_indices)
  np.savetxt(os.path.join(wordshoal_save_dir, "vocabulary.txt"), 
             collapsed_vocabulary, 
             fmt="%s")
  np.savetxt(os.path.join(wordshoal_save_dir, "author_map.txt"), 
             collapsed_author_map, 
             fmt="%s")
  np.savetxt(os.path.join(wordshoal_save_dir, "debate_map.txt"), 
             collapsed_debate_map, 
             fmt="%s")
  
  print("...done")


if __name__ == '__main__':
  app.run(main)
