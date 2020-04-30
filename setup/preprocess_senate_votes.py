"""Preprocess votes for vote-based ideal points.

The original raw data can be found at [1]. We need three files: one for votes,
one for members, and one for rollcalls. For example, for Senate session 114,
we would use the files: S114_votes.csv, S114_members.csv, S114_rollcalls.csv.

#### References
[1]: Lewis, Jeffrey B., Keith Poole, Howard Rosenthal, Adam Boche, 
     Aaron Rudkin, and Luke Sonnet (2020). Voteview: Congressional Roll-Call 
     Votes Database. https://voteview.com/

"""
import os
import numpy as np
import pandas as pd

from absl import app
from absl import flags

flags.DEFINE_integer("senate_session",
                     default=114,
                     help="Senate session to preprocess.")
FLAGS = flags.FLAGS


def main(argv):
  del argv
  project_dir = os.path.abspath(
      os.path.join(os.path.dirname(__file__), os.pardir)) 
  source_dir = os.path.join(
      project_dir, "data/senate-votes/{}".format(FLAGS.senate_session))
  data_dir = os.path.join(source_dir, "raw")
  save_dir = os.path.join(source_dir, "clean")

  votes_df = pd.read_csv(
      os.path.join(data_dir, "S{}_votes.csv".format(FLAGS.senate_session)))
  members_df = pd.read_csv(
      os.path.join(data_dir, "S{}_members.csv".format(FLAGS.senate_session)))
  rollcalls_df = pd.read_csv(
      os.path.join(data_dir, "S{}_rollcalls.csv".format(FLAGS.senate_session)))
  df = votes_df.merge(members_df, left_on='icpsr', right_on='icpsr')

  def get_name_and_party(row):
    senator = row['bioname']
    first_name = senator.split(" ")[1].title()
    last_name = senator.split(" ")[0][:-1].title()
    if row['party_code'] == 200:
      return " ".join([first_name, last_name, "(R)"])
    elif row['party_code'] == 100:
      return " ".join([first_name, last_name, "(D)"])
    else:
      return " ".join([first_name, last_name, "(I)"])
  
  # Ignore votes that aren't cast as {1, 2, 3, 4, 5, 6}. 
  def get_vote(row):
    if row['cast_code'] in [1, 2, 3]:
      return 1
    elif row['cast_code'] in [4, 5, 6]:
      return 0
    else:
      return -1
    
  senator = np.array(df.apply(lambda row: get_name_and_party(row), axis=1))
  senator_to_senator_id = dict(
      [(y.title(), x) for x, y in enumerate(sorted(set(senator)))])
  senator_indices = np.array(
      [senator_to_senator_id[s.title()] for s in senator])
  senator_map = np.array(list(senator_to_senator_id.keys()))

  # We also download the first dimension of the fitted DW-Nominate scores.
  first_senator_locations = np.array(
      [np.where(senator == senator_name)[0][0] 
       for senator_name in senator_map])
  nominate_scores = np.array(df['nominate_dim1'])[first_senator_locations]

  # Subtract 1 to zero-index.
  bill_indices = np.array(df.rollnumber) - 1

  votes = np.array(df.apply(lambda row: get_vote(row), axis=1))

  missing_votes = np.where(votes == -1)[0]
  senator_indices = np.delete(senator_indices, missing_votes)
  bill_indices = np.delete(bill_indices, missing_votes)
  votes = np.delete(votes, missing_votes)

  # Get bill information.
  bill_descriptions = np.array(rollcalls_df.vote_desc)
  bill_names = np.array(rollcalls_df.bill_number)

  # Save data.
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
  # `votes.npy` is a [num_total_votes] vector that contains a binary indicator
  # for each vote.
  np.save(os.path.join(save_dir, "votes.npy"), votes)
  # `senator_indices.npy` is a [num_total_votes] vector where each entry is an
  # integer in {0, 1, ..., num_senators - 1}, indicating whose vote is the 
  # corresponding entry in `votes.npy`.
  np.save(os.path.join(save_dir, "senator_indices.npy"), senator_indices)
  # `bill_indices.npy` is a [num_total_votes] vector where each entry is an
  # integer in {0, 1, ..., num_bills - 1}, indicating which bill is being votes 
  # on in the corresponding entry in `votes.npy`.
  np.save(os.path.join(save_dir, "bill_indices.npy"), bill_indices)
  # `nominate_scores.npy` is a [num_senators] vector where each entry contains
  # the pre-fitted DW-Nominate ideal points (using the first dimension).
  np.save(os.path.join(save_dir, "nominate_scores.npy"), nominate_scores)
  # `senator_map.txt` is a file of [num_senators] strings, containing the names
  # of seach senator.
  np.savetxt(os.path.join(save_dir, "senator_map.txt"), senator_map, fmt="%s")
  # `bill_descriptions.txt` is a file of [num_bills] strings, containing
  # descriptions for each bill.
  np.savetxt(os.path.join(save_dir, "bill_descriptions.txt"), 
             bill_descriptions, 
             fmt="%s")
  # `bill_names.txt` is a file of [num_bills] strings, containing the name of
  # each bill.
  np.savetxt(os.path.join(save_dir, "bill_names.txt"), 
             bill_names, 
             fmt="%s")


if __name__ == '__main__':
  app.run(main)
