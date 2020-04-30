"""Analyze TBIP results with Wordfish and Wordshoal.

Specifically, we compare by seeing which ideal points are closest to the
vote-based ideal points learned for the various Senate sessions.
"""

import os
import numpy as np
import analysis_utils as utils


def print_comparisons(senate_session, vote_ideal_point_type):
  project_dir = os.path.abspath(
      os.path.join(os.path.dirname(__file__), os.pardir)) 
  source_dir = os.path.join(project_dir, "data/senate-speech-comparisons")
  vote_source_dir = os.path.join(project_dir, 
                                 "data/senate-votes/{}".format(senate_session))
  wordfish_source_dir =  os.path.join(source_dir, 
                                      "wordfish/{}".format(senate_session))
  wordshoal_source_dir =  os.path.join(source_dir, 
                                       "wordshoal/{}".format(senate_session))
  tbip_source_dir = os.path.join(source_dir, "tbip/{}".format(senate_session))
  # vote_source_dir = "senate-votes/{}/".format(senate_session)

  # Load wordfish data and fits.
  wordfish_data_dir = os.path.join(wordfish_source_dir, "clean")
  wordfish_param_dir = os.path.join(wordfish_source_dir, 
                                    "wordfish-fits/params")
  wordfish_author_map = np.loadtxt(
      os.path.join(wordfish_data_dir, "author_map.txt"),
      dtype=str,
      delimiter="\n")
  wordfish_ideal_points = np.load(
      os.path.join(wordfish_param_dir, "ideal_point_loc.npy"))

  # Load wordshoal data and fits.
  wordshoal_data_dir = os.path.join(wordshoal_source_dir, "clean")
  wordshoal_param_dir = os.path.join(wordshoal_source_dir, 
                                     "wordshoal-fits/params")
  wordshoal_author_map = np.loadtxt(
      os.path.join(wordshoal_data_dir, "author_map.txt"),
      dtype=str,
      delimiter="\n")
  wordshoal_ideal_points = np.load(
      os.path.join(wordshoal_param_dir, "stage_2_ideal_point_loc.npy"))

  # Load TBIP data and fits.
  tbip_data_dir = os.path.join(tbip_source_dir, "clean")
  tbip_param_dir = os.path.join(tbip_source_dir, "tbip-fits/params/")
  tbip_author_map = np.loadtxt(os.path.join(tbip_data_dir, "author_map.txt"),
                               dtype=str,
                               delimiter="\n")
  tbip_ideal_points = np.load(os.path.join(tbip_param_dir, 
                                           "ideal_point_loc.npy"))

  # Load vote data and fits.
  vote_data_dir = os.path.join(vote_source_dir, "clean")
  (_, _, _, voter_map, _, _, 
   vote_ideal_points_dw_nominate) = utils.load_vote_data(vote_data_dir)
  vote_param_dir = os.path.join(vote_source_dir, "fits/params")
  vote_ideal_points_1d = np.load(os.path.join(vote_param_dir, 
                                              "ideal_point_loc.npy"))
  if vote_ideal_point_type == '1D':
    vote_ideal_points = vote_ideal_points_1d
  elif vote_ideal_point_type == 'DW-Nominate':
    vote_ideal_points = vote_ideal_points_dw_nominate
  else:
    raise ValueError("Unrecognized vote ideal point type. "
                     "Must be '1D' or 'DW-Nominate'.")

  # Make sure the author maps for the three speech ideal point models use the
  # same author map.
  matching_author_maps = (np.mean(wordshoal_author_map == tbip_author_map) + 
                          np.mean(wordshoal_author_map == wordfish_author_map))
  if matching_author_maps != 2:
    raise ValueError("Non-matching maps")

  # Since they are all the same, use tbip_author_map as the author map.
  author_map = tbip_author_map

  # Only include Senators in both text and vote datasets.
  wordfish_ideal_points, _, _ = utils.match_authors_with_votes(
      wordfish_ideal_points, vote_ideal_points, author_map, voter_map)
  wordshoal_ideal_points, _, _ = utils.match_authors_with_votes(
      wordshoal_ideal_points, vote_ideal_points, author_map, voter_map)
  (tbip_ideal_points, vote_ideal_points, 
   combined_author_map) = utils.match_authors_with_votes(
      tbip_ideal_points, vote_ideal_points, author_map, voter_map)

  # Standardize and rotate ideal points so all datasets are in same order.
  wordfish_ideal_points, _ = utils.standardize_and_rotate_ideal_points(
      wordfish_ideal_points, vote_ideal_points)
  wordshoal_ideal_points, _ = utils.standardize_and_rotate_ideal_points(
      wordshoal_ideal_points, vote_ideal_points)
  tbip_ideal_points, vote_ideal_points = (
      utils.standardize_and_rotate_ideal_points(tbip_ideal_points, 
                                                vote_ideal_points))

  vote_wordfish_correlation = np.corrcoef(wordfish_ideal_points, 
                                          vote_ideal_points)[0][1]
  vote_wordshoal_correlation = np.corrcoef(wordshoal_ideal_points, 
                                           vote_ideal_points)[0][1]
  vote_tbip_correlation = np.corrcoef(tbip_ideal_points, 
                                      vote_ideal_points)[0][1]

  print("Correlation between Wordfish and {} votes for Session {}: "
        "{:.2f}.".format(
            vote_ideal_point_type, senate_session, vote_wordfish_correlation))
  print("Correlation between Wordshoal and {} votes for Session {}: "
        "{:.2f}.".format(
            vote_ideal_point_type, senate_session, vote_wordshoal_correlation))
  print("Correlation between TBIP and {} votes for Session {}: "
        "{:.2f}.".format(
            vote_ideal_point_type, senate_session, vote_tbip_correlation))

  vote_wordfish_rank_correlation = utils.get_rank_correlation(
      wordfish_ideal_points, vote_ideal_points)
  vote_wordshoal_rank_correlation = utils.get_rank_correlation(
      wordshoal_ideal_points, vote_ideal_points)
  vote_tbip_rank_correlation = utils.get_rank_correlation(
      tbip_ideal_points, vote_ideal_points)

  print("Rank Correlation between Wordfish and {} votes for Session {}: "
        "{:.2f}.".format(vote_ideal_point_type, 
                         senate_session, 
                         vote_wordfish_rank_correlation))
  print("Rank Correlation between Wordshoal and {} votes for Session {}: "
        "{:.2f}.".format(vote_ideal_point_type, 
                         senate_session, 
                         vote_wordshoal_rank_correlation))
  print("Rank Correlation between TBIP and {} votes for Session {}: "
        "{:.2f}.".format(vote_ideal_point_type, 
                         senate_session, 
                         vote_tbip_rank_correlation))


if __name__ == "__main__":
  vote_ideal_point_types = ['1D', 'DW-Nominate']
  senate_sessions = [111, 112, 113]
  for vote_ideal_point_type in vote_ideal_point_types:
    for senate_session in senate_sessions:
      print_comparisons(senate_session, vote_ideal_point_type)
