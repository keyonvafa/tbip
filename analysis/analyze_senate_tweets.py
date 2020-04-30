"""Analyze TBIP results on 114th Senate tweets."""

import os
import numpy as np
import analysis_utils as utils

project_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)) 
source_dir = os.path.join(project_dir, "data/senate-tweets-114")
vote_source_dir = os.path.join(project_dir, "data/senate-votes/114")

# Load TBIP data.
data_dir = os.path.join(source_dir, "clean")
(counts, vocabulary, author_indices, 
 author_map, raw_documents) = utils.load_text_data(data_dir)

vote_ideal_point_types = ['DW-Nominate', '1D']
for vote_ideal_point_type in vote_ideal_point_types:
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

  # Load Wordfish data.
  wordfish_param_dir = os.path.join(source_dir, "wordfish-fits/params")
  wordfish_ideal_points = np.load(
      os.path.join(wordfish_param_dir, "ideal_point_loc.npy"))

  # Load voting data. Note that these are NOT in the same order as the TBIP data.
  vote_data_dir = os.path.join(vote_source_dir, "clean")
  (votes, senator_indices, bill_indices, voter_map, bill_descriptions, 
   bill_names, vote_ideal_points_dw_nominate) = utils.load_vote_data(
      vote_data_dir)

  # Load fitted vote ideal points.
  vote_param_dir = os.path.join(vote_source_dir, "fits/params")

  if vote_ideal_point_type == '1D':
    vote_ideal_points = np.load(os.path.join(vote_param_dir, 
                                             "ideal_point_loc.npy"))
  elif vote_ideal_point_type == 'DW-Nominate':
    vote_ideal_points = vote_ideal_points_dw_nominate

  # Match the vote and tweet orderings so they are ordered.
  wordfish_ideal_points, _, _ = utils.match_authors_with_votes(
      wordfish_ideal_points, vote_ideal_points, author_map, voter_map)
  (tweet_ideal_points, vote_ideal_points, 
   combined_author_map) = utils.match_authors_with_votes(
      ideal_point_mean, vote_ideal_points, author_map, voter_map)
   
  wordfish_ideal_points, _ = utils.standardize_and_rotate_ideal_points(
      wordfish_ideal_points, vote_ideal_points)
  (tweet_ideal_points, vote_ideal_points) = (
      utils.standardize_and_rotate_ideal_points(tweet_ideal_points, 
                                                vote_ideal_points))

  tbip_correlation = np.corrcoef(tweet_ideal_points, 
                                 vote_ideal_points)[0][1]
  wordfish_correlation = np.corrcoef(wordfish_ideal_points, 
                                     vote_ideal_points)[0][1]
  print("The correlation between the TBIP tweet ideal points and {} vote "
        "ideal points is {:.3f}.".format(vote_ideal_point_type, 
                                         tbip_correlation))
  print("The correlation between the Wordfish tweet ideal points and {} vote "
        "ideal points is {:.3f}.\n".format(vote_ideal_point_type,
                                           wordfish_correlation))

  tbip_rank_correlation = utils.get_rank_correlation(tweet_ideal_points,
                                                     vote_ideal_points)
  wordfish_rank_correlation = utils.get_rank_correlation(wordfish_ideal_points,
                                                         vote_ideal_points)
  print("The rank correlation between the TBIP tweet ideal points and {} "
        "vote ideal points is {:.3f}.".format(vote_ideal_point_type, 
                                              tbip_rank_correlation))
  print("The rank correlation between the Wordfish tweet ideal points and {} "
        "vote ideal points is {:.3f}.\n".format(vote_ideal_point_type,
                                                wordfish_rank_correlation))

# Note that `vote_ideal_points` now refers to the rotated and standardized
# 1D ideal points (and not DW-Nominate), and `tweet_ideal_points` is also
# rotated and standardized. To get the non-rotated or standardized ideal 
# points, we use `ideal_point-mean`.

# Compare Deb Fischer's ideal point to her vote-based ideal point.
fischer_index = np.where(combined_author_map == "Deb Fischer (R)")[0][0]
fischer_vote_ideal_point = vote_ideal_points[fischer_index]
fischer_tweet_ideal_point = tweet_ideal_points[fischer_index]
print("Deb Fischer's tweet ideal point (rotated and standardized) is "
      "{:.3f}.".format(fischer_tweet_ideal_point))
print("Deb Fischer's vote ideal point (rotated and standardized) is "
      "{:.3f}.\n".format(fischer_vote_ideal_point))

# Find how extreme Deb Fischer's ideal point is. Note we are now going back
# to using the ideal points found in the TBIP fit, not the merged list.
fischer_tbip_rank = np.sum(
    ideal_point_mean <= ideal_point_mean[fischer_index])
fischer_tbip_rotated_rank = np.sum(
    ideal_point_mean >= ideal_point_mean[fischer_index])
print("Deb Fischer has the {}th most extreme TBIP ideal point "
      "for Senate tweets.".format(np.minimum(fischer_tbip_rank, 
                                             fischer_tbip_rotated_rank)))

vote_ideal_points = np.load(os.path.join(vote_param_dir, 
                                         "ideal_point_loc.npy"))
fischer_vote_rank = np.sum(
    vote_ideal_points <= vote_ideal_points[fischer_index])
fischer_vote_rotated_rank = np.sum(
    vote_ideal_points >= vote_ideal_points[fischer_index])
print("Deb Fischer has the {}th most extreme vote ideal point.\n".format(
    np.minimum(fischer_vote_rank, fischer_vote_rotated_rank)))


# Find influential tweets that explain why Deb Fischer's ideal point wasn't
# more extreme.
# Find conservative ideal point sign (negative or positive) by comparing to
# Bernie Sanders.
sanders_index = np.where(author_map == "Bernard Sanders (I)")[0][0]
if np.sign(ideal_point_mean[sanders_index]) == -1:
  most_conservative_ideal_point = np.max(ideal_point_mean)
  most_liberal_ideal_point = np.min(ideal_point_mean)
else:
  most_conservative_ideal_point = np.min(ideal_point_mean)
  most_liberal_ideal_point = np.max(ideal_point_mean)

verbosity_weights = utils.get_verbosity_weights(counts, author_indices)
fischer_top_indices, fischer_top_words = utils.compute_likelihood_ratio(
    "Deb Fischer (R)",
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
    log_counts=False,
    query_size=50)
fischer_top_tweets = raw_documents[fischer_top_indices]

# The tweet used in the paper contains the following quote. We want to find
# where it is ranked among the top ideological words.
paper_quote_substring = "FACT: 1963 Equal Pay Act enables women"
match_indices = [index for index, tweet in enumerate(fischer_top_tweets) if 
                 paper_quote_substring in tweet]
if len(match_indices) == 0:
  print("Paper quote not found in Fischer results.\n")
else:
  match_index = match_indices[0]
  fischer_snippet = fischer_top_tweets[match_index]
  print("An excerpt from a Fischer tweet with rank {}: {}\n".format(
      match_index, fischer_snippet))

  paper_quote_substring = "I want to empower women to be their own best "
match_indices = [index for index, tweet in enumerate(fischer_top_tweets) if 
                 paper_quote_substring in tweet]
if len(match_indices) == 0:
  print("Paper quote not found in Fischer results.\n")
else:
  match_index = match_indices[0]
  fischer_snippet = fischer_top_tweets[match_index]
  print("An excerpt from a Fischer tweet with rank {}: {}\n".format(
      match_index, fischer_snippet))

pay_tweet_index = fischer_top_indices[match_index]
keyword = 'women'
expected_pay_count_liberal = utils.get_expected_word_count(
    keyword,
    most_liberal_ideal_point,
    document_mean[pay_tweet_index], 
    objective_topic_mean, 
    ideological_topic_mean,
    vocabulary)
expected_pay_count_conservative = utils.get_expected_word_count(
    keyword,
    most_conservative_ideal_point,
    document_mean[pay_tweet_index], 
    objective_topic_mean, 
    ideological_topic_mean,
    vocabulary)

print("The expected word count for '{}' for the most liberal ideal point "
      "using the topics in Fischer's tweet directly above is {:.3f}.".format(
          keyword, expected_pay_count_liberal))
print("The expected word count for '{}' for the most conservative ideal "
      "point using the topics in Fischer's tweet directly above is "
      "{:.3f}.\n".format(keyword, expected_pay_count_conservative))

keyword = '#equalpay'
expected_pay_count_liberal = utils.get_expected_word_count(
    keyword,
    most_liberal_ideal_point,
    document_mean[pay_tweet_index], 
    objective_topic_mean, 
    ideological_topic_mean,
    vocabulary)
expected_pay_count_conservative = utils.get_expected_word_count(
    keyword,
    most_conservative_ideal_point,
    document_mean[pay_tweet_index], 
    objective_topic_mean, 
    ideological_topic_mean,
    vocabulary)

print("The expected word count for '{}' for the most liberal ideal point "
      "using the topics in Fischer's tweet directly above is {:.3f}.".format(
          keyword, expected_pay_count_liberal))
print("The expected word count for '{}' for the most conservative ideal "
      "point using the topics in Fischer's tweet directly above is "
      "{:.3f}.".format(keyword, expected_pay_count_conservative))

