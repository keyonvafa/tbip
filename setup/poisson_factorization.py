"""Poisson Factorization with coordinate-ascent variational inference.

We use the model and updates from [1].

#### References
[1]: Prem Gopalan, Jake Hofman, David Blei. Scalable Recommendation with 
     Hierarchical Poisson Factorization. In _Conference on Uncertainty in
     Artifical Intelligence_, 2015. https://arxiv.org/abs/1311.1704
"""

import os
import time

from absl import app
from absl import flags
import numpy as np
from scipy.special import digamma, gammaln
import scipy.sparse as sparse

flags.DEFINE_integer("num_topics",
                     default=50,
                     help="Number of topics")
flags.DEFINE_integer("max_steps",
                     default=300,
                     help="Maximum number of steps")
flags.DEFINE_string("data",
                    default="senate-speeches-114",
                    help="Data source being used.")
flags.DEFINE_integer("senate_session",
                     default=113,
                     help="Senate session (used only when data is "
                          "'comparisons').")
flags.DEFINE_integer("seed",
                     default=123,
                     help="Random seed to be used.")
FLAGS = flags.FLAGS


def logsumexp(logits):
  """Applies the LogSumExp function to a tensor of logits of arbitrary rank.

  Args:
    logits: An arbitrary-rank tensor of shape [..., num_codes]. We sum over the
      last dimension.

  Returns:
    normalized_sum: The log of the exponents summed over the last dimension,
      with shape [..., 1].
  """
  max_value = np.max(logits, axis=-1)
  # Add a new axis so max_value is the same rank as logits.
  max_value = max_value[..., np.newaxis]
  unnormalized_sum = np.log(np.sum(np.exp(logits - max_value), axis=-1))
  normalized_sum = max_value + unnormalized_sum[..., np.newaxis]
  return normalized_sum


def geometric_gamma_mean(shape, rate):
  """Returns the geometric mean of a Gamma distributed random variable."""
  geometric_mean = np.exp(digamma(shape)) / rate
  return geometric_mean


def get_expected_gamma_log_prior(variational_shape,
                                 variational_rate,
                                 prior_shape,
                                 prior_rate):
  expected_log_prior = (prior_shape - 1.) * (
      digamma(variational_shape) - np.log(variational_rate)) - (
      prior_rate * (variational_shape / variational_rate))
  return np.sum(expected_log_prior)


def get_expected_log_conditional(variational_shape,
                                 variational_rate,
                                 variational_prior_shape,
                                 variational_prior_rate,
                                 prior_shape,
                                 axis):
  expected_factor = variational_shape / variational_rate
  expected_rate = variational_prior_shape / variational_prior_rate
  expected_log_factor = (digamma(variational_shape) -
                         np.log(variational_rate))
  expected_log_rate = (digamma(variational_prior_shape) -
                       np.log(variational_prior_rate))
  if axis == 0:
    expanded_rate = expected_rate[np.newaxis, :]
    num_topics = variational_shape.shape[0]
  elif axis == 1:
    expanded_rate = expected_rate[:, np.newaxis]
    num_topics = variational_shape.shape[1]
  else:
    raise NotImplementedError("Axis must be 0 or 1.")
  normalizing_term = num_topics * prior_shape * np.sum(expected_log_rate)
  variable_term = (prior_shape - 1.) * np.sum(expected_log_factor)
  exponential_term = np.sum(expected_factor * expanded_rate)
  expected_log_conditional = (normalizing_term + variable_term -
                              exponential_term)
  return expected_log_conditional


def get_expected_data_log_likelihood(counts,
                                     variational_topic_shape,
                                     variational_topic_rate,
                                     variational_document_shape,
                                     variational_document_rate):
  expected_topic = variational_topic_shape / variational_topic_rate
  expected_document = variational_document_shape / variational_document_rate
  negative_rate = -np.sum(np.matmul(expected_document, expected_topic))
  # Don't need to compute other Poisson terms because they cancel out with
  # entropy calculations.
  return negative_rate


def get_gamma_entropy(variational_shape, variational_rate):
  entropy = (variational_shape -
             np.log(variational_rate) +
             gammaln(variational_shape) +
             ((1. - variational_shape) * digamma(variational_shape)))
  entropy = np.sum(entropy)
  return entropy


def get_auxiliary_entropy(normalizer, counts):
  return np.sum(counts * np.log(normalizer))


def get_elbo(counts,
             variational_topic_shape,
             variational_topic_rate,
             variational_document_shape,
             variational_document_rate,
             variational_topic_prior_shape,
             variational_topic_prior_rate,
             variational_document_prior_shape,
             variational_document_prior_rate,
             normalizer,
             prior_shape,
             hyperprior_shape,
             hyperprior_rate):
  expected_topic_rate_log_prior = get_expected_gamma_log_prior(
      variational_topic_prior_shape,
      variational_topic_prior_rate,
      hyperprior_shape,
      hyperprior_rate)
  expected_document_rate_log_prior = get_expected_gamma_log_prior(
      variational_document_prior_shape,
      variational_document_prior_rate,
      hyperprior_shape,
      hyperprior_rate)

  expected_topic_log_conditional = get_expected_log_conditional(
      variational_topic_shape,
      variational_topic_rate,
      variational_topic_prior_shape,
      variational_topic_prior_rate,
      prior_shape,
      axis=0)
  expected_document_log_conditional = get_expected_log_conditional(
      variational_document_shape,
      variational_document_rate,
      variational_document_prior_shape,
      variational_document_prior_rate,
      prior_shape,
      axis=1)

  expected_data_log_likelihood = get_expected_data_log_likelihood(
      counts,
      variational_topic_shape,
      variational_topic_rate,
      variational_document_shape,
      variational_document_rate)

  topic_rate_entropy = get_gamma_entropy(variational_topic_prior_shape,
                                         variational_topic_prior_rate)
  document_rate_entropy = get_gamma_entropy(variational_document_prior_shape,
                                            variational_document_prior_rate)
  topic_entropy = get_gamma_entropy(variational_topic_shape,
                                    variational_topic_rate)
  document_entropy = get_gamma_entropy(variational_document_shape,
                                       variational_document_rate)
  auxiliary_entropy = get_auxiliary_entropy(normalizer,
                                            counts)

  log_joint = (expected_topic_rate_log_prior +
               expected_document_rate_log_prior +
               expected_topic_log_conditional +
               expected_document_log_conditional +
               expected_data_log_likelihood)
  entropy = (topic_rate_entropy +
             document_rate_entropy +
             topic_entropy +
             document_entropy +
             auxiliary_entropy)
  elbo = log_joint + entropy
  return elbo


def update_variational_document_shape(counts,
                                      variational_topic_shape,
                                      variational_topic_rate,
                                      variational_document_shape,
                                      variational_document_rate,
                                      normalizer,
                                      prior_shape):
  """
  Updates the variational document shape (gamma).

  Args:
    counts: A `NumPy` array with shape `[num_documents, num_words]`
      representing the document counts.
    variational_topic_shape: A `NumPy` array with shape `[num_topics,
      num_words]`.
    variational_topic_rate: A `NumPy` array with shape `[num_topics,
      num_words]`.
    variational_document_shape: A `NumPy` array with shape `[num_documents,
      num_topics]`.
    variational_document_rate: A `NumPy` array with shape `[num_documents,
      num_topics]`.
    normalizer: A `NumPy` array with shape `[num_documents, num_topics]`.
    prior_shape: A scalar (denoted by `a` in the paper).

  Returns:
    variational_document_shape: A `NumPy` array with shape `[num_documents,
      num_topics]`.
  """
  normalized_counts = counts / normalizer
  geometric_topic = geometric_gamma_mean(variational_topic_shape,
                                         variational_topic_rate)
  geometric_document = geometric_gamma_mean(variational_document_shape,
                                            variational_document_rate)
  expected_counts = geometric_document * np.matmul(
      normalized_counts,
      geometric_topic.T)
  variational_document_shape = prior_shape + expected_counts
  return variational_document_shape


def update_variational_document_rate(variational_topic_shape,
                                     variational_topic_rate,
                                     variational_document_prior_shape,
                                     variational_document_prior_rate):
  """
  Updates the variational document rate (gamma).

  Args:
    variational_topic_shape: A `NumPy` array with shape `[num_topics,
      num_words]`.
    variational_topic_rate: A `NumPy` array with shape `[num_topics,
      num_words]`.
    variational_document_prior_shape: A `NumPy` array with shape
      `[num_documents]` (denoted by kappa in the paper).
    variational_document_prior_rate: A `NumPy` array with shape
      `[num_documents]` (denoted by kappa in the paper).

  Returns:
    variational_document_rate: A `NumPy` array with shape `[num_documents,
      num_topics]`.
  """
  expected_topic = np.sum(
      variational_topic_shape / variational_topic_rate,
      axis=1)
  expected_prior = (
      variational_document_prior_shape / variational_document_prior_rate)
  variational_document_rate = (expected_prior[:, np.newaxis] +
                               expected_topic[np.newaxis, :])
  return variational_document_rate


def update_variational_document_prior_rate(variational_document_shape,
                                           variational_document_rate,
                                           hyperprior_rate):
  """
  Updates the variational document prior rate (kappa).

  Args:
    variational_document_shape: A `NumPy` array with shape `[num_documents,
      num_topics]`.
    variational_document_rate: A `NumPy` array with shape `[num_documents,
      num_topics]`.
    hyperprior_rate: A scalar (denoted by a'/b' in the paper).

  Returns:
    variational_document_prior_rate: A `NumPy` array with shape
      `[num_documents]`.
  """
  variational_document_mean = np.sum(
      variational_document_shape / variational_document_rate,
      axis=1)
  variational_document_prior_rate = hyperprior_rate + variational_document_mean
  return variational_document_prior_rate


def update_variational_topic_shape(counts,
                                   variational_topic_shape,
                                   variational_topic_rate,
                                   variational_document_shape,
                                   variational_document_rate,
                                   normalizer,
                                   prior_shape):
  """
  Updates the variational topic shape (lambda).

  Args:
    counts: A `NumPy` array with shape `[num_documents, num_words]`
      representing the document counts.
    variational_topic_shape: A `NumPy` array with shape `[num_topics,
      num_words]`.
    variational_topic_rate: A `NumPy` array with shape `[num_topics,
      num_words]`.
    variational_document_shape: A `NumPy` array with shape `[num_documents,
      num_topics]`.
    variational_document_rate: A `NumPy` array with shape `[num_documents,
      num_topics]`.
    normalizer: A `NumPy` array with shape `[num_documents, num_topics]`.
    prior_shape: A scalar (denoted by `a` in the paper).

  Returns:
    variational_topic_shape: A `NumPy` array with shape `[num_topics,
      num_words]`.
  """
  normalized_counts = counts / normalizer
  geometric_topic = geometric_gamma_mean(variational_topic_shape,
                                         variational_topic_rate)
  geometric_document = geometric_gamma_mean(variational_document_shape,
                                            variational_document_rate)
  expected_counts = geometric_topic * np.matmul(
      geometric_document.T,
      normalized_counts)
  variational_topic_shape = prior_shape + expected_counts
  return variational_topic_shape


def update_variational_topic_rate(variational_document_shape,
                                  variational_document_rate,
                                  variational_topic_prior_shape,
                                  variational_topic_prior_rate):
  """
  Updates the variational topic rate (lambda).

  Args:
    variational_document_shape: A `NumPy` array with shape `[num_documents,
      num_topics]`.
    variational_document_rate: A `NumPy` array with shape `[num_documents,
      num_topics]`.
    variational_topic_prior_shape: A `NumPy` array with shape `[num_words]`.
    variational_topic_prior_rate: A `NumPy` array with shape `[num_words]`.

  Returns:
    variational_topic_rate: A `NumPy` array with shape `[num_topics,
      num_words]`.
  """
  expected_topic = np.sum(
      variational_document_shape / variational_document_rate,
      axis=0)
  expected_prior = variational_topic_prior_shape / variational_topic_prior_rate
  variational_topic_rate = (expected_prior[np.newaxis, :] +
                            expected_topic[:, np.newaxis])
  return variational_topic_rate


def update_variational_topic_prior_rate(variational_topic_shape,
                                        variational_topic_rate,
                                        hyperprior_rate):
  """
  Updates the variational topic prior rate (kappa).

  Args:
    variational_topic_shape: A `NumPy` array with shape `[num_topics,
      num_words]`.
    variational_topic_rate: A `NumPy` array with shape `[num_topics,
      num_words]`.
    hyperprior_rate: A scalar (denoted by c'/d' in the paper).

  Returns:
    variational_topic_prior_rate: A `NumPy` array with shape
      `[num_words]`.
  """
  variational_topic_mean = np.sum(
      variational_topic_shape / variational_topic_rate,
      axis=0)
  variational_topic_prior_rate = hyperprior_rate + variational_topic_mean
  return variational_topic_prior_rate


def get_normalizer(variational_topic_shape,
                   variational_topic_rate,
                   variational_document_shape,
                   variational_document_rate):
  """Get normalizing constant sum_k G(theta_ik)G(beta_kj).

  Args:
    variational_topic_shape: Matrix with shape `[num_topics, num_words]`.
    variational_topic_rate: Matrix with shape `[num_topics, num_words]`.
    variational_document_shape: Matrix with shape `[num_documents,
      num_topics].`
    variational_document_rate: Matrix with shape `[num_documents,
      num_topics]`.

  Returns:
    normalizer: A matrix with shape `[num_documents, num_words]`.
  """
  geometric_document = geometric_gamma_mean(variational_document_shape,
                                            variational_document_rate)
  geometric_topic = geometric_gamma_mean(variational_topic_shape,
                                         variational_topic_rate)
  normalizer = np.matmul(geometric_document, geometric_topic)
  return normalizer


def print_topics(topic_scores, vocabulary, words_per_topic=10):
  num_topics, num_words = topic_scores.shape
  top_words = np.argsort(-topic_scores, axis=1)
  res = []
  for topic_idx in range(num_topics):
    row_start_string = "index={}".format(topic_idx)
    row_content = [vocabulary[word] for word in
                   top_words[topic_idx, :words_per_topic]]
    row_content_string = ", ".join(row_content)
    row = " ".join([row_start_string, row_content_string])
    res.append(row)
  print(np.array(res))


def main(argv):
  del argv
  project_dir = os.path.abspath(
      os.path.join(os.path.dirname(__file__), os.pardir)) 
  source_dir = os.path.join(project_dir, "data/{}".format(FLAGS.data))
  if FLAGS.data == "senate-speech-comparisons":
    source_dir = os.path.join(
        source_dir, "tbip/{}".format(FLAGS.senate_session))
  # Data directory must have the following files: counts.npz, vocabulary.txt.
  data_dir = os.path.join(source_dir, "clean")
  save_dir = os.path.join(source_dir, "pf-fits")

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  counts_sparse = sparse.load_npz(os.path.join(data_dir, "counts.npz"))
  counts = np.array(counts_sparse.todense())
  vocabulary = np.loadtxt(os.path.join(data_dir, "vocabulary.txt"),
                          dtype=str,
                          delimiter="\n",
                          comments="<!-")

  num_documents, num_words = counts.shape
  print("{} documents, {} words".format(num_documents, num_words))

  prior_shape = 0.3
  hyperprior_shape = 0.3
  hyperprior_rate = 0.3

  # Initialize variational parameters.
  random_state = np.random.RandomState(FLAGS.seed)
  variational_topic_shape = np.exp(
      0.1 * random_state.randn(FLAGS.num_topics, num_words))
  variational_topic_rate = np.exp(
      0.1 * random_state.randn(FLAGS.num_topics, num_words))
  variational_document_shape = np.exp(
      0.1 * random_state.randn(num_documents, FLAGS.num_topics))
  variational_document_rate = np.exp(
      0.1 * random_state.randn(num_documents, FLAGS.num_topics))
  variational_document_prior_rate = np.exp(
      0.1 * random_state.randn(num_documents))
  variational_topic_prior_rate = np.exp(
      0.1 * random_state.randn(num_words))

  variational_document_prior_shape = (hyperprior_shape +
                                      FLAGS.num_topics * prior_shape)
  variational_topic_prior_shape = (hyperprior_shape +
                                   FLAGS.num_topics * prior_shape)

  for step in range(FLAGS.max_steps):
    start_time = time.time()
    normalizer = get_normalizer(variational_topic_shape,
                                variational_topic_rate,
                                variational_document_shape,
                                variational_document_rate)
    new_variational_document_shape = update_variational_document_shape(
        counts,
        variational_topic_shape,
        variational_topic_rate,
        variational_document_shape,
        variational_document_rate,
        normalizer,
        prior_shape)
    new_variational_document_rate = update_variational_document_rate(
        variational_topic_shape,
        variational_topic_rate,
        variational_document_prior_shape,
        variational_document_prior_rate)
    new_variational_document_prior_rate = (
        update_variational_document_prior_rate(
            new_variational_document_shape,
            new_variational_document_rate,
            hyperprior_rate))
    new_variational_topic_shape = update_variational_topic_shape(
        counts,
        variational_topic_shape,
        variational_topic_rate,
        variational_document_shape,
        variational_document_rate,
        normalizer,
        prior_shape)
    new_variational_topic_rate = update_variational_topic_rate(
        new_variational_document_shape,
        new_variational_document_rate,
        variational_topic_prior_shape,
        variational_topic_prior_rate)
    new_variational_topic_prior_rate = update_variational_topic_prior_rate(
        new_variational_topic_shape,
        new_variational_topic_rate,
        hyperprior_rate)
    (variational_document_shape, variational_document_rate,
     variational_topic_shape, variational_topic_rate,
     variational_document_prior_rate, variational_topic_prior_rate) = (
        new_variational_document_shape, new_variational_document_rate,
        new_variational_topic_shape, new_variational_topic_rate,
        new_variational_document_prior_rate, new_variational_topic_prior_rate)

    # Don't compute ELBO every step because it takes time.
    if step % 50 == 0 or step == FLAGS.max_steps - 1:
      topic_mean = variational_topic_shape / variational_topic_rate
      print_topics(topic_mean, vocabulary)
      elbo = get_elbo(counts,
                      variational_topic_shape,
                      variational_topic_rate,
                      variational_document_shape,
                      variational_document_rate,
                      variational_topic_prior_shape,
                      variational_topic_prior_rate,
                      variational_document_prior_shape,
                      variational_document_prior_rate,
                      normalizer,
                      prior_shape,
                      hyperprior_shape,
                      hyperprior_rate)
      duration = time.time() - start_time
      print("Step: {:>2d}, ELBO: {:.3f} ({:.3f} sec)".format(
          step, elbo, duration))
      np.save(os.path.join(save_dir, "document_shape"), 
              variational_document_shape)
      np.save(os.path.join(save_dir, "document_rate"), 
              variational_document_rate)
      np.save(os.path.join(save_dir, "topic_shape"), variational_topic_shape)
      np.save(os.path.join(save_dir, "topic_rate"), variational_topic_rate)
    elif step % 10 == 0:     
      duration = time.time() - start_time
      print("Step: {:>2d} ({:.3f} sec)".format(step, duration))


if __name__ == '__main__':
  app.run(main)
