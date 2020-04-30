"""Analyze TBIP results on 114th Senate speeches."""

import os
import numpy as np
import analysis_utils as utils

project_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)) 
source_dir = os.path.join(project_dir, "data/senate-speeches-114")
vote_source_dir = os.path.join(project_dir, "data/senate-votes/114")

# Load TBIP data.
data_dir = os.path.join(source_dir, "clean")
(counts, vocabulary, author_indices, 
 author_map, raw_documents) = utils.load_text_data(data_dir)

# Load TBIP parameters.
param_dir = os.path.join(source_dir, "tbip-fits/params/")
(document_loc, document_scale, objective_topic_loc, objective_topic_scale, 
 ideological_topic_loc, ideological_topic_scale, ideal_point_loc, 
 ideal_point_scale) = utils.load_tbip_parameters(param_dir)

# Compute means from variational parameters
document_mean = np.exp(document_loc + document_scale ** 2 / 2)
objective_topic_mean = np.exp(objective_topic_loc + 
                              objective_topic_scale ** 2 / 2)
ideological_topic_mean = ideological_topic_loc
ideal_point_mean = ideal_point_loc

# Print topics.
utils.print_topics(objective_topic_loc, 
                   objective_topic_scale,
                   ideological_topic_loc, 
                   ideological_topic_scale, 
                   vocabulary)

# Load voting data. Note that these are NOT in the same order as the TBIP data.
vote_data_dir = os.path.join(vote_source_dir, "clean")
(votes, senator_indices, bill_indices, voter_map, bill_descriptions, 
 bill_names, vote_ideal_points_dw_nominate) = utils.load_vote_data(
    vote_data_dir)

# Load fitted vote ideal points.
vote_param_dir = os.path.join(vote_source_dir, "fits/params")
vote_ideal_points_1d = np.load(os.path.join(vote_param_dir, 
                                            "ideal_point_loc.npy"))

# Match the vote and speech orderings so they are ordered.
(speech_ideal_points, vote_ideal_points_1d, 
 combined_author_map) = utils.match_authors_with_votes(
    ideal_point_mean, vote_ideal_points_1d, author_map, voter_map)

(speech_ideal_points, vote_ideal_points_1d) = (
    utils.standardize_and_rotate_ideal_points(speech_ideal_points, 
                                              vote_ideal_points_1d))

vote_speech_correlation = np.corrcoef(speech_ideal_points, 
                                      vote_ideal_points_1d)[0][1]
print("The correlation between the TBIP speech ideal points and the 1D vote "
      "ideal points is {:.3f}.".format(vote_speech_correlation))

vote_speech_rank_correlation = utils.get_rank_correlation(speech_ideal_points,
                                                          vote_ideal_points_1d)
print("The rank correlation between the TBIP speech ideal points and the 1D "
      "vote ideal points is {:.3f}.\n".format(vote_speech_rank_correlation))

# Find how extreme Bernie Sanders' ideal point is. Note we are now going back
# to using the ideal points found in the TBIP fit, not the merged list.
sanders_index = np.where(author_map == "Bernard Sanders (I)")[0][0]
sanders_tbip_rank = np.sum(
    ideal_point_mean <= ideal_point_mean[sanders_index])
sanders_tbip_rotated_rank = np.sum(
    ideal_point_mean >= ideal_point_mean[sanders_index])
print("Bernie Sanders has the {}st most extreme TBIP ideal point "
      "for Senate speeches".format(np.minimum(sanders_tbip_rank, 
                                              sanders_tbip_rotated_rank)))

# Find influential speeches for Bernie Sanders' ideal point.
verbosity_weights = utils.get_verbosity_weights(counts, author_indices)
sanders_top_indices, sanders_top_words = utils.compute_likelihood_ratio(
    "Bernard Sanders (I)",
    ideal_point_mean,
    counts,
    vocabulary,
    author_indices,
    author_map,
    document_mean,
    objective_topic_mean,
    ideological_topic_mean,
    verbosity_weights,
    null_ideal_point=0.,
    log_counts=True)
sanders_top_speeches = raw_documents[sanders_top_indices]
# The speech used in the paper contains the following quote. We want to find
# where it is ranked among the top ideological words.
paper_quote_substring = "the United States is the only major country on Earth"
match_indices = [index for index, speech in enumerate(sanders_top_speeches) if 
                 paper_quote_substring in speech]
if len(match_indices) == 0:
  print("Paper quote not found in Sanders results.")
else:
  match_index = match_indices[0]
  sanders_snippet_a = sanders_top_speeches[match_index][234:344]
  sanders_snippet_b = sanders_top_speeches[match_index][-933:-738]
  print("An excerpt from a Sanders speech with rank {}: {}... {}\n".format(
      match_index, sanders_snippet_a, sanders_snippet_b))
  
# Find how extreme Jeff Sessions' ideal point is. It is pretty moderate!
sessions_index = np.where(author_map == "Jefferson Sessions (R)")[0][0]
sessions_tbip_rank = np.sum(
    ideal_point_mean <= ideal_point_mean[sessions_index])
sessions_tbip_rotated_rank = np.sum(
    ideal_point_mean >= ideal_point_mean[sessions_index])
print("Jeff Sessions has the {}th most extreme TBIP ideal point "
      "for Senate speeches.".format(np.minimum(sessions_tbip_rank, 
                                               sessions_tbip_rotated_rank)))

# Find influential speeches that explain why Jeff Sessions' ideal point wasn't
# more extreme.
# Find the most conservative ideal point by comparing to Sanders.
if np.sign(ideal_point_mean[sanders_index]) == -1:
  most_conservative_ideal_point = np.max(ideal_point_mean)
  most_liberal_ideal_point = np.min(ideal_point_mean)
else:
  most_conservative_ideal_point = np.min(ideal_point_mean)
  most_liberal_ideal_point = np.max(ideal_point_mean)
sessions_top_indices, sessions_top_words = utils.compute_likelihood_ratio(
    "Jefferson Sessions (R)",
    ideal_point_mean,
    counts,
    vocabulary,
    author_indices,
    author_map,
    document_mean,
    objective_topic_mean,
    ideological_topic_mean,
    verbosity_weights,
    null_ideal_point=most_conservative_ideal_point,
    log_counts=True)
sessions_top_speeches = raw_documents[sessions_top_indices]
# The speech used in the paper contains the following quote. We want to find
# where it is ranked among the top ideological words.
paper_quote_substring = "The President of the United States is giving work"
match_indices = [index for index, speech in enumerate(sessions_top_speeches) if 
                 paper_quote_substring in speech]
if len(match_indices) == 0:
  print("Paper quote not found in Sessions results.\n")
else:
  match_index = match_indices[0]
  sessions_snippet = sessions_top_speeches[match_index][4015:4296]
  print("An excerpt from a Sessions speech with rank {}: {}\n".format(
      match_index, sessions_snippet))

# Find expected word count of 'DACA' using most liberal ideal point and most
# conservative ideal point. Fix the topics at the topics of the DACA speech.
daca_speech_index = sessions_top_indices[match_index]
expected_daca_count_liberal = utils.get_expected_word_count(
    'daca',
    most_liberal_ideal_point,
    document_mean[daca_speech_index], 
    objective_topic_mean, 
    ideological_topic_mean,
    vocabulary)
expected_daca_count_conservative = utils.get_expected_word_count(
    'daca',
    most_conservative_ideal_point,
    document_mean[daca_speech_index], 
    objective_topic_mean, 
    ideological_topic_mean,
    vocabulary)

print("The expected word count for 'DACA' for the most liberal ideal point "
      "using the topics in Sessions' speech is {:.2f}.".format(
          expected_daca_count_liberal))
print("The expected word count for 'DACA' for the most conservative ideal "
      "point using the topics in Sessions' speech is {:.2f}.".format(
          expected_daca_count_conservative))
