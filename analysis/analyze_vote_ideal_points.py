"""Analyze vote ideal points."""

import os
import numpy as np
import analysis_utils as utils

project_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)) 
vote_source_dir = os.path.join(project_dir, "data/senate-votes/114")

# Load voting data.
vote_data_dir = os.path.join(vote_source_dir, "clean")
(votes, senator_indices, bill_indices, voter_map, bill_descriptions, 
 bill_names, vote_ideal_points_dw_nominate) = utils.load_vote_data(
    vote_data_dir)

# Load fitted vote ideal points.
vote_param_dir = os.path.join(vote_source_dir, "fits/params")
(polarity_loc, polarity_scale, popularity_loc, popularity_scale, 
 ideal_point_loc, ideal_point_scale) = utils.load_vote_ideal_point_parameters(
    vote_param_dir)

polarity_mean = polarity_loc
popularity_mean = popularity_loc
ideal_point_mean = ideal_point_loc

# Find how extreme Bernie Sanders' ideal point is.
sanders_index = np.where(voter_map == "Bernard Sanders (I)")[0][0]
sanders_rank = np.sum(
    ideal_point_mean <= ideal_point_mean[sanders_index])
sanders_rotated_rank = np.sum(
    ideal_point_mean >= ideal_point_mean[sanders_index])
print("Bernie Sanders has the {}th most extreme vote ideal point "
      "for Senate speeches.".format(np.minimum(sanders_rank, 
                                               sanders_rotated_rank)))

# Find the most liberal ideal point by assuming it is the same direction as 
# Sanders'.
if np.sign(ideal_point_mean[sanders_index]) == -1:
  most_conservative_ideal_point = np.max(ideal_point_mean)
  most_liberal_ideal_point = np.min(ideal_point_mean)
else:
  most_conservative_ideal_point = np.min(ideal_point_mean)
  most_liberal_ideal_point = np.max(ideal_point_mean)

sanders_top_bills = utils.compute_vote_likelihood_ratio(
    "Bernard Sanders (I)",
    ideal_point_mean,
    votes,
    senator_indices,
    bill_indices,
    voter_map,
    popularity_mean,
    polarity_mean,
    null_ideal_point=most_liberal_ideal_point,
    query_size=40)

hr_2048_indices = np.where(bill_names[sanders_top_bills] == "HR2048")[0]
if len(hr_2048_indices) == 0:
  print("The likelihood ratio statistic did not identify any votes about HR "
        "2048 as explicatory of Bernie Sanders' more moderate ideal point.")
else:
  print("Bernie Sanders' vote on HR 2048 is the {}th most explicatory bill of "
        "his more moderate ideal point.".format(hr_2048_indices[0]))
