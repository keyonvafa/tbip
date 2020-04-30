"""Helpful functions for analysis."""

import numpy as np
import os
import scipy.sparse as sparse
from scipy.stats import bernoulli, poisson


def load_text_data(data_dir):
  """Load text data used to train the TBIP.
  
  Args:
    data_dir: Path to directory where data is stored.
  
  Returns:
    counts: A sparse matrix with shape [num_documents, num_words], representing
      the documents in a bag-of-words format.
    vocabulary: An array of strings with shape [num_words].
    author_indices: An array of integeres with shape [num_documents], where
      each entry represents the author who wrote the document.
    author_map: An array of strings with shape [num_authors], containing the
      names of each author.
    raw_documents: A string vector with shape [num_documents] containing the
      raw documents.
  """
  counts = sparse.load_npz(
      os.path.join(data_dir, "counts.npz"))
  vocabulary = np.loadtxt(
      os.path.join(data_dir, "vocabulary.txt"), 
      dtype=str, 
      delimiter="\n",
      comments="<!-")
  author_indices = np.load(
      os.path.join(data_dir, "author_indices.npy")).astype(np.int32) 
  author_map = np.loadtxt(
      os.path.join(data_dir, "author_map.txt"),
      dtype=str,
      delimiter="\n")
  raw_documents = np.loadtxt(
      os.path.join(data_dir, "raw_documents.txt"), 
      dtype=str, 
      delimiter="\n",
      comments="<!-")   
  return counts, vocabulary, author_indices, author_map, raw_documents


def load_tbip_parameters(param_dir):
  """Load the TBIP model parameters from directory where they are stored.
  
  Args:
    param_dir: Path to directory where the TBIP fitted parameters are stored.

  Returns:
    document_loc: Variational lognormal location parameter for the
      document intensities (theta), with shape [num_documents, num_topics].
    document_loc: Variational lognormal scale parameter for the
      document intensities (theta), with shape [num_documents, num_topics].
    objective_topic_loc: Variational lognormal location parameter for the 
      objective topic (beta), with [num_topics, num_words].
    objective_topic_scale: Variational lognormal scale parameter for the 
      objective topic (beta), with shape [num_topics, num_words].
    ideological_topic_loc: Variational Gaussian location parameter for the 
      ideological topic (eta), with shape [num_topics, num_words].
    ideological_topic_scale: Variational Gaussian scale parameter for the 
      ideological topic (eta), with shape [num_topics, num_words].
    ideal_point_loc: Variational Gaussian location parameter for the 
      ideal points (x), with shape [num_authors].
    ideal_point_scale: Variational Gaussian scale parameter for the 
      ideal points (x), with shape [num_authors].
  """ 
  document_loc = np.load(
      os.path.join(param_dir, "document_loc.npy"))
  document_scale = np.load(
      os.path.join(param_dir, "document_scale.npy"))
  objective_topic_loc = np.load(
      os.path.join(param_dir, "objective_topic_loc.npy"))
  objective_topic_scale = np.load(
      os.path.join(param_dir, "objective_topic_scale.npy"))
  ideological_topic_loc = np.load(
      os.path.join(param_dir, "ideological_topic_loc.npy"))
  ideological_topic_scale = np.load(
      os.path.join(param_dir, "ideological_topic_scale.npy"))
  ideal_point_loc = np.load(
      os.path.join(param_dir, "ideal_point_loc.npy"))
  ideal_point_scale = np.load(
      os.path.join(param_dir, "ideal_point_scale.npy"))
  return (document_loc, document_scale, objective_topic_loc, 
          objective_topic_scale, ideological_topic_loc, 
          ideological_topic_scale, ideal_point_loc, ideal_point_scale)


def load_vote_data(vote_data_dir):
  """Load vote data used to train vote-based ideal points for Senators.
  
  Args:
    vote_data_dir: Path to directory where data is stored.
  
  Returns:
    votes: An array with shape [num_votes], containing the binary vote cast
      for all recorded votes.
    senator_indices: An array with shape [num_votes], where each entry is an
      integer in {0, 1, ..., num_voters - 1}, containing the index for the 
      Senator corresponding to the vote in `votes`.
    bill_indices: An array with shape [num_votes], where each entry is an
      integer in {0, 1, ..., num_bills - 1}, containing the index for the 
      bill corresponding to the vote in `votes`.
    voter_map: A string array with length [num_voters] containing the
      name for each voter in the data set.
    bill_descriptions: A string array with length [num_bills] containing 
      descriptions for each bill being voted on.
    bill_descriptions: A string array with length [num_bills] containing 
      the name for each bill being voted on.
    vote_ideal_points_dw_nominate: An array with shape [num_voters], containing
      the DW-Nominate scores for each Senator. Note that these are not trained
      locally; they are the values provided by Voteview.
  """
  votes = np.load(os.path.join(vote_data_dir, "votes.npy"))
  senator_indices = np.load(os.path.join(vote_data_dir, "senator_indices.npy"))
  bill_indices = np.load(os.path.join(vote_data_dir, "bill_indices.npy"))
  voter_map = np.loadtxt(os.path.join(vote_data_dir, "senator_map.txt"),
                         dtype=str, 
                         delimiter="\n")
  bill_descriptions = np.loadtxt(
      os.path.join(vote_data_dir, "bill_descriptions.txt"),
      dtype=str, 
      delimiter="\n")
  bill_names = np.loadtxt(
      os.path.join(vote_data_dir, "bill_names.txt"),
      dtype=str, 
      delimiter="\n")
  vote_ideal_points_dw_nominate = np.load(
      os.path.join(vote_data_dir, "nominate_scores.npy"))
  return (votes, senator_indices, bill_indices, voter_map, bill_descriptions,
          bill_names, vote_ideal_points_dw_nominate)
  

def load_vote_ideal_point_parameters(param_dir):
  """Load the parameters for the 1D vote ideal point model.
  
  Args:
    param_dir: Path to directory containing the vote ideal point parameters.

  Returns:
    polarity_loc: Variational Gaussian location parameter for the polarities 
      (eta), with shape [num_bills].
    polarity_scale: Variational Gaussian scale parameter for the polarities 
      (eta), with shape [num_bills].
    popularity_loc: Variational Gaussian location parameter for the 
      popularities (alpha), with shape [num_bills].
    popularity_scale: Variational Gaussian scale parameter for the 
      popularities (alpha), with shape [num_bills].
    ideal_point_loc: Variational Gaussian location parameter for the 
      ideal points (x), with shape [num_authors].
    ideal_point_scale: Variational Gaussian scale parameter for the 
      ideal points (x), with shape [num_authors].
  """ 
  polarity_loc = np.load(os.path.join(param_dir, "polarity_loc.npy"))
  polarity_scale = np.load(os.path.join(param_dir, "polarity_scale.npy"))
  popularity_loc = np.load(os.path.join(param_dir, "popularity_loc.npy"))
  popularity_scale = np.load(os.path.join(param_dir, "popularity_scale.npy"))
  ideal_point_loc = np.load(os.path.join(param_dir, "ideal_point_loc.npy"))
  ideal_point_scale = np.load(os.path.join(param_dir, "ideal_point_scale.npy"))
  return (polarity_loc, polarity_scale, popularity_loc, 
          popularity_scale, ideal_point_loc, ideal_point_scale)

  
def get_ideological_topic_means(objective_topic_loc, 
                                objective_topic_scale,
                                ideological_topic_loc, 
                                ideological_topic_scale):
  """Returns neutral and ideological topics from variational parameters.
  
  For each (k,v), we want to evaluate E[beta_kv], E[beta_kv * exp(eta_kv)], 
  and E[beta_kv * exp(-eta_kv)], where the expectations are with respect to the 
  variational distributions. Like the paper, beta refers to the obective topic
  and eta refers to the ideological topic.
  
  Dropping the indices and denoting by mu_b the objective topic location and 
  sigma_b the objective topic scale, we have E[beta] = exp(mu + sigma_b^2 / 2),
  using the mean of a lognormal distribution.
  
  Denoting by mu_e the ideological topic location and sigma_e the ideological
  topic scale, we have E[beta * exp(eta)] = E[beta]E[exp(eta)] by the 
  mean-field assumption. exp(eta) is lognormal distributed, so E[exp(eta)] =
  exp(mu_e + sigma_e^2 / 2). Thus, E[beta * exp(eta)] = 
  exp(mu_b + mu_e + (sigma_b^2 + sigma_e^2) / 2).
  
  Finally, E[beta * exp(-eta)] = 
  exp(mu_b - mu_e + (sigma_b^2 + sigma_e^2) / 2).
  
  Because we only care about the orderings of topics, we can drop the exponents
  from the means.
  
  Args:
    objective_topic_loc: Variational lognormal location parameter for the 
      objective topic (beta). Should be shape [num_topics, num_words].
    objective_topic_scale: Variational lognormal scale parameter for the 
      objective topic (beta). Should be positive, with shape 
      [num_topics, num_words].
    ideological_topic_loc: Variational Gaussian location parameter for the 
      ideological topic (eta). Should be shape [num_topics, num_words].
    ideological_topic_scale: Variational Gaussian scale parameter for the 
      ideological topic (eta). Should be positive, with shape 
      [num_topics, num_words].
  
  Returns:
    neutral_mean: A matrix with shape [num_topics, num_words] denoting the
      variational mean for the neutral topics.
    positive_mean: A matrix with shape [num_topics, num_words], denoting the
      variational mean for the ideological topics with an ideal point of +1.
    negative_mean: A matrix with shape [num_topics, num_words], denoting the
      variational mean for the ideological topics with an ideal point of -1.
  """
  neutral_mean = objective_topic_loc + objective_topic_scale ** 2 / 2
  positive_mean = (objective_topic_loc + 
                   ideological_topic_loc + 
                   (objective_topic_scale ** 2 + 
                    ideological_topic_scale ** 2) / 2)
  negative_mean = (objective_topic_loc - 
                   ideological_topic_loc +
                   (objective_topic_scale ** 2 + 
                    ideological_topic_scale ** 2) / 2)
  return neutral_mean, positive_mean, negative_mean


def print_topics(objective_topic_loc, 
                 objective_topic_scale,
                 ideological_topic_loc, 
                 ideological_topic_scale, 
                 vocabulary,
                 words_per_topic=10):
  """Prints neutral and ideological topics from variational parameters.
  
  Args:
    objective_topic_loc: Variational lognormal location parameter for the 
      objective topic (beta). Should be shape [num_topics, num_words].
    objective_topic_scale: Variational lognormal scale parameter for the 
      objective topic (beta). Should be positive, with shape 
      [num_topics, num_words].
    ideological_topic_loc: Variational Gaussian location parameter for the 
      ideological topic (eta). Should be shape [num_topics, num_words].
    ideological_topic_scale: Variational Gaussian scale parameter for the 
      ideological topic (eta). Should be positive, with shape 
      [num_topics, num_words].
    vocabulary: A list of strings with shape [num_words].
    words_per_topic: The number of words to print for each topic.
  """
  
  neutral_mean, positive_mean, negative_mean = get_ideological_topic_means(
      objective_topic_loc, 
      objective_topic_scale,
      ideological_topic_loc, 
      ideological_topic_scale)
  num_topics, num_words = neutral_mean.shape
  
  top_neutral_words = np.argsort(-neutral_mean, axis=1)
  top_negative_words = np.argsort(-negative_mean, axis=1)
  top_positive_words = np.argsort(-positive_mean, axis=1)
  topic_strings = []
  for topic_idx in range(num_topics):
    neutral_start_string = "Neutral {}:".format(topic_idx)
    neutral_row = [vocabulary[word] for word in 
                   top_neutral_words[topic_idx, :words_per_topic]]
    neutral_row_string = ", ".join(neutral_row)
    neutral_string = " ".join([neutral_start_string, neutral_row_string])
    
    positive_start_string = "Positive {}:".format(topic_idx)
    positive_row = [vocabulary[word] for word in 
                    top_positive_words[topic_idx, :words_per_topic]]
    positive_row_string = ", ".join(positive_row)
    positive_string = " ".join([positive_start_string, positive_row_string])
    
    negative_start_string = "Negative {}:".format(topic_idx)
    negative_row = [vocabulary[word] for word in 
                    top_negative_words[topic_idx, :words_per_topic]]
    negative_row_string = ", ".join(negative_row)
    negative_string = " ".join([negative_start_string, negative_row_string])
    
    topic_strings.append(negative_string)
    topic_strings.append(neutral_string)
    topic_strings.append(positive_string)
    topic_strings.append("==========")
    
  print("{}\n".format(np.array(topic_strings)))
  
  
def match_authors_with_votes(text_ideal_points, 
                             vote_ideal_points, 
                             author_map, 
                             voter_map, 
                             verbose=False):
  """Matches the list of Senate authors with vote IDs.
  
  Can work for either Senate speeches or Senate tweets.
  
  Args:
    text_ideal_points: The learned ideal points from the TBIP for Senator
      speeches or tweets, an array with shape [num_authors].
    vote_ideal_points: The vote ideal points for Senators, an array with shape 
      [num_voters].
    author_map: The list of author names used by the TBIP, with shape
      [num_authors].
    voter_map: The list of Senator names used for vote data, with shape
      [num_voters].
    verbose: A boolean that determines whether to print the matched and
      unmatched data.
  
  Returns:
    text_id_to_vote_id: A dictionary where each key is an index in {0, 1, ..., 
      num_authors - 1}, which maps to the corresponding voter index in {0, 1, 
      ..., num_voters -1}. The number of entries in the dictionary is at most
      min(num_authors, num_voters).  
  """
  text_id_to_vote_id = {}
  for author_ind, author in enumerate(author_map):
    last_name_and_party = ' '.join(author.lower().split(" ")[-2:])
    matching_inds = [vote_ind for vote_ind in range(len(voter_map)) 
                     if last_name_and_party in voter_map[vote_ind].lower()]
    if len(matching_inds) == 0 and author == "Joseph Lieberman (I)":
      last_name_and_party = "lieberman (d)"
      matching_inds = [vote_ind for vote_ind in range(len(voter_map)) 
                       if last_name_and_party in voter_map[vote_ind].lower()]
    if len(matching_inds) == 2 and author == "Ben Nelson (D)":
      last_name_and_party = "earl nelson (d)"
      matching_inds = [vote_ind for vote_ind in range(len(voter_map)) 
                       if last_name_and_party in voter_map[vote_ind].lower()]
    if len(matching_inds) == 2 and author == "Bill Nelson (D)":
      last_name_and_party = "clarence nelson (d)"
      matching_inds = [vote_ind for vote_ind in range(len(voter_map)) 
                       if last_name_and_party in voter_map[vote_ind].lower()]
    if len(matching_inds) == 2 and author == "Tom Udall (D)":
      last_name_and_party = "thomas udall (d)"
      matching_inds = [vote_ind for vote_ind in range(len(voter_map)) 
                       if last_name_and_party in voter_map[vote_ind].lower()]
    if len(matching_inds) == 2 and author == "Mark Udall (D)":
      last_name_and_party = "mark udall (d)"
      matching_inds = [vote_ind for vote_ind in range(len(voter_map)) 
                       if last_name_and_party in voter_map[vote_ind].lower()]
    if len(matching_inds) == 0 and author == "Mark Dayton (DFL)":
      last_name_and_party = "mark dayton (d)"
      matching_inds = [vote_ind for vote_ind in range(len(voter_map)) 
                       if last_name_and_party in voter_map[vote_ind].lower()]
    if len(matching_inds) == 2 and author == "Gordon Smith (R)":
      last_name_and_party = "gordon smith (r)"
      matching_inds = [vote_ind for vote_ind in range(len(voter_map)) 
                       if last_name_and_party in voter_map[vote_ind].lower()]
    if len(matching_inds) == 2 and author == "Robert C. Smith (R)":
      last_name_and_party = "robert smith (r)"
      matching_inds = [vote_ind for vote_ind in range(len(voter_map)) 
                       if last_name_and_party in voter_map[vote_ind].lower()]
    if len(matching_inds) == 2 and author == "Lincoln D. Chafee (R)":
      last_name_and_party = "lincoln chafee (r)"
      matching_inds = [vote_ind for vote_ind in range(len(voter_map)) 
                       if last_name_and_party in voter_map[vote_ind].lower()]
    if len(matching_inds) != 1:
      if verbose:
        print("Error with speech author {}: {} matching indices in vote "
              "authors".format(author, len(matching_inds)))
    else:
      text_id_to_vote_id[author_ind] = matching_inds[0]
      if verbose:
        print("Matched {} with {}".format(author, voter_map[matching_inds[0]]))
  if verbose:
    print("Matched {} pairs (out of {} authors and {} voters).\n".format(
        len(text_id_to_vote_id.keys()), len(author_map), len(voter_map)))
  
  matches_found = np.array(list(text_id_to_vote_id.keys()))
  combined_author_map = author_map[matches_found]
  text_ideal_points = text_ideal_points[matches_found]
  vote_ideal_points = np.array([vote_ideal_points[text_id_to_vote_id[text_id]] 
                                for text_id in matches_found])
  return text_ideal_points, vote_ideal_points, combined_author_map


def standardize(x):
  """Standardize a vector x."""
  return (x - np.mean(x)) / np.std(x)


def standardize_and_rotate_ideal_points(text_ideal_points, vote_ideal_points):
  """Standardize and rotate ideal point lists so they have same direction.
  
  Args:
    text_ideal_points: The learned ideal points from the TBIP for Senator
      speeches or tweets, an array with shape [num_combined_senators].
    vote_ideal_points: The vote ideal points for Senators, an array with shape 
      [num_combined_senators]. 
  
  Returns:
    rotated_text_ideal_points: The text ideal points standardized and rotated
      if necessary.
    standardized_vote_ideal_points: The standardized vote ideal points.
  """
  standardized_text_ideal_points = standardize(text_ideal_points)
  standardized_vote_ideal_points = standardize(vote_ideal_points)
  correlation_sign = np.sign(
      np.corrcoef(standardized_text_ideal_points, 
                  standardized_vote_ideal_points)[0][1])
  rotated_text_ideal_points = standardized_text_ideal_points * correlation_sign
  return rotated_text_ideal_points, standardized_vote_ideal_points


def get_rank_correlation(list_a, list_b):
  """Get Spearman's rank correlation between two lists.
  
  Args:
    list_a: A list of values. 
    list_b: A list of values with the same length as `list_a`.
  
  Returns:
    rank_correlation: Spearman's rank correlation between the two lists.
  """
  a_rank = np.array([sorted(list_a).index(v) for v in list_a])
  b_rank = np.array([sorted(list_b).index(v) for v in list_b])
  rank_correlation = np.corrcoef(a_rank, b_rank)[0][1]
  return rank_correlation


def get_verbosity_weights(counts, author_indices):
  """Compute weights to capture author verbosity (as described in Appendix A).
  
  Specifically, if n_s is the average word count over documents for author s,
  we set a weight: w_s = n_s / (1/S sum_s' n_s'). That is, if w_s is 1, the 
  author is as verbose as the average author. If it is more than 1 the author
  is more verbose (i.e. uses more words per document), and if it is less than
  1 the author is less verbose (i.e. uses less words per document).
  
  We find this modification does not make much of a difference for the 
  correlation results, but it helps us interpret the ideal points for the
  qualitative analysis.
  
  Args:
    counts: A matrix of word counts, with shape [num_documents, num_words].
    author_indices: An int-vector of author indices, with shape 
      [num_documents]. Each index is in the set {0, 1, ..., num_authors - 1},
      and indicates the author of the document. 
  
  Returns:
    verbosity_weights: A vector with shape [num_authors], where each entry is a
      weight that captures the author's verbosity. 
  """
  total_counts_per_author = np.bincount(
      author_indices, 
      weights=np.array(np.sum(counts, axis=1)).flatten())
  counts_per_document_per_author = (
      total_counts_per_author / np.bincount(author_indices))
  verbosity_weights = (counts_per_document_per_author / 
                       np.mean(np.sum(counts, axis=1)))
  return verbosity_weights


def compute_likelihood_ratio(name,
                             true_ideal_points,
                             counts,
                             vocabulary,
                             author_indices,
                             author_map,
                             document_mean,
                             objective_topic_mean,
                             ideological_topic_mean,
                             verbosity_weights,
                             null_ideal_point=0.,
                             log_counts=False,
                             query_size=10):
  """Compute top documents and words based on likelihood ratio statistic.
  
  Args:
    name: Name of author to calculate ratio for.
    true_ideal_points: A vector of length [num_authors] containing the learned
      TBIP ideal points.
    counts: A sparse matrix with shape [num_documents, num_words], representing
      the documents in a bag-of-words format.
    vocabulary: An array of strings with shape [num_words].
    author_indices: An array of integeres with shape [num_documents], where
      each entry represents the author who wrote the document.
    author_map: An array of strings with shape [num_authors], containing the
      names of each author.
    document_mean: The learned variational mean for the document intensities
      (theta). A matrix with shape [num_documents, num_words].
    objective_topic_mean: The learned variational mean for the objective
      topics (beta). A matrix with shape [num_words, num_topics].
    ideological_topic_mean: The learned variational mean for the ideological
      topics (eta). A matrix with shape [num_words, num_topics].
    verbosity_weights: A vector with shape [num_authors], where each entry is a
      weight that captures the author's verbosity. 
    raw_documents: A string vector with shape [num_documents] containing the
      raw documents.
    null_ideal_point: The null ideal point for the likelihood ratio test. For
      example, a null ideal point of 0 would capture why the author is away
      from 0 (which documents and words make this author extreme?).
    log_counts: A boolean, whether to use the logged counts as the output data.
    query_size: Number of documents to query.
  
  Returns:
    top_document_indices: A vector of ints with shape [query_size], 
      representing the indices for the top documents identified by the 
      likelihood ratio statistic.
    top_words: A vector of strings with shape [query_size], representing
      the top word in each of the top `query_size` documents identified by the
      likelihood ratio statistic.
  """
  author_index = np.where(author_map == name)[0][0]
  author_documents = np.where(author_indices == author_index)[0]
  true_ideal_point = true_ideal_points[author_index]
  author_weight = verbosity_weights[author_index]
  null_rate = author_weight * np.matmul(
      document_mean[author_documents], 
      objective_topic_mean * np.exp(ideological_topic_mean * null_ideal_point))
  true_rate = author_weight * np.matmul(
      document_mean[author_documents], 
      objective_topic_mean * np.exp(ideological_topic_mean * true_ideal_point))
  counts = counts.toarray()
  if log_counts:
    counts = np.round(np.log(counts + 1))
  null_log_prob = poisson.logpmf(counts[author_documents], null_rate)
  true_log_prob = poisson.logpmf(counts[author_documents], true_rate)
  log_likelihood_ratios = true_log_prob - null_log_prob
  summed_log_likelihood_ratios = np.sum(log_likelihood_ratios, axis=1)
  top_document_indices = np.argsort(-summed_log_likelihood_ratios)[:query_size]
  top_document_indices = author_documents[top_document_indices]
  top_word_indices = np.argmax(log_likelihood_ratios, axis=1)
  top_words = vocabulary[top_word_indices]
  return top_document_indices, top_words


def get_expected_word_count(word,
                            ideal_point,
                            document_mean, 
                            objective_topic_mean, 
                            ideological_topic_mean,
                            vocabulary):
  """Gets expected count for a word and ideal point using fitted topics.
  
  Args:
    word: The word we want to query, a string.
    ideal_point: The ideal point to compute the expectation. 
    document_mean: A vector with shape [num_topics], representing the 
      document intensities for the word count we want to evaluate.
    objective_topic_mean: A matrix with shape [num_topics, num_words],
      representing the fitted objective topics (beta). 
    ideological_topic_mean: A matrix with shape [num_topics, num_words],
      representing the fitted ideological topics (eta). 
    vocabulary: [vocabulary: An array of strings with shape [num_words].
  
  Returns:
    expected_word_count: A scalar representing the expected word count for 
      the queried word, ideal point, and document intensities.
  """
  word_index = np.where(vocabulary == word)[0][0]
  expected_word_count = np.dot(
      document_mean,
      objective_topic_mean[:, word_index] * 
      np.exp(ideal_point * ideological_topic_mean[:, word_index]))
  return expected_word_count


def compute_vote_likelihood_ratio(name,
                                  true_ideal_points,
                                  votes,
                                  senator_indices,
                                  bill_indices,
                                  voter_map,
                                  popularity_mean,
                                  polarity_mean,
                                  null_ideal_point=0.,
                                  query_size=10):
  """Compute top votes based on likelihood ratio statistic.
  
  Args:
    name: Name of Senator to calculate ratio for.
    true_ideal_points: A vector of length [num_voters] containing the learned
      vote ideal points.
    votes: An array with shape [num_votes], containing the binary vote cast
      for all recorded votes.
    senator_indices: An array with shape [num_votes], where each entry is an
      integer in {0, 1, ..., num_voters - 1}, containing the index for the 
      Senator corresponding to the vote in `votes`.
    bill_indices: An array with shape [num_votes], where each entry is an
      integer in {0, 1, ..., num_bills - 1}, containing the index for the 
      bill corresponding to the vote in `votes`.
    voter_map: A string array with length [num_voters] containing the
      name for each voter in the data set.
    popularity_mean: The learned variational mean for the bill popularities
      (alpha). A vector with shape [num_bills].
    polarity_mean: The learned variational mean for the bill polarities (eta).
      A vector with shape [num_bills].
    null_ideal_point: The null ideal point for the likelihood ratio test. For
      example, a null ideal point of 0 would capture why the author is away
      from 0 (which documents and words make this author extreme?).
    query_size: Number of documents to query.
  
  Returns:
    top_indices: A vector of ints with shape [query_size], representing the 
      indices for the top bills identified by the likelihood ratio statistic.
  """
  voter_index = np.where(voter_map == name)[0][0]
  relevant_indices = np.where(senator_indices == voter_index)[0]
  fitted_logit = (true_ideal_points[voter_index] *
                  polarity_mean[bill_indices[relevant_indices]] +
                  popularity_mean[bill_indices[relevant_indices]])
  null_logit = (null_ideal_point *
                polarity_mean[bill_indices[relevant_indices]] +
                popularity_mean[bill_indices[relevant_indices]])
  # Unlike TBIP likelihood ratio statistic, the output distirbution for vote
  # ideal points is Bernoulli. Here we compute the logit.
  fitted_mean = 1. / (1. + np.exp(-fitted_logit))
  null_mean = 1. / (1. + np.exp(-null_logit))
  fitted_log_likelihood = bernoulli.pmf(votes[relevant_indices], fitted_mean)        
  null_log_likelihood = bernoulli.pmf(votes[relevant_indices], null_mean)
  log_likelihood_differences = fitted_log_likelihood - null_log_likelihood
  top_indices = bill_indices[
      relevant_indices[np.argsort(-log_likelihood_differences)[:query_size]]]
  return top_indices
