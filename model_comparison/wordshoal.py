"""Use Wordshoal model to estimate ideology from Senate speeches.

Wordshoal [1] fits speeches in two stages. First, for speaker i at debate t, 
the  count of word j is denoted by y_{ijt} and is distributed:

y_{ijt} ~ Pois(exp(nu_{it} + lambda_{jt} + chi_{it} * b_{jt})),

where psi_{it} is the ideal point for debate t, b_{jt} is the polarity of 
word j for debate t, and nu_{it} and lambda_{jt} are debate-specific speaker- 
and word-intercepts. 

Wordshoal includes a second stage factor model to aggregate ideal points across
debates. Factorizing the fitted psi_{it}, the model posits:

psi_{it} ~ Normal(alpha_t + x_i * beta_t, tau_i),

where x_i is now the aggregated ideal point and beta_t is the aggregated 
polarity. The model includes Gaussian priors on the real-valued parameters, and
a Gamma prior on tau_i. 

We fit both stages using variational inference with reparameterization
gradients, using a Gaussian variational family for all parameters except for
tau_i, where we use a log-normal due to the positivity constraint. We note that
this is different from the inference procedure used in [1]. 

#### References
[1]  Benjamin E. Lauderdale and Alexander Herzog. Measuring Political Positions
     from Legislative Speech. In _Political Analysis_, 2016.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time

# Dependency imports
from absl import flags
import numpy as np
import scipy.sparse as sparse
import tensorflow as tf
import tensorflow_probability as tfp


flags.DEFINE_float("learning_rate",
                   default=0.01,
                   help="Adam learning rate.")
flags.DEFINE_integer("max_steps",
                     default=20000,
                     help="Number of training steps to run.")
flags.DEFINE_integer("num_samples",
                     default=1,
                     help="Number of samples for ELBO approximation.")
flags.DEFINE_enum("data",
                  default="senate-speech-comparisons",
                  enum_values=["senate-speech-comparisons"],
                  help="Data set used.")
flags.DEFINE_integer("batch_size",
                     default=1024,
                     help="Batch size. Used only for stage 1, because we use"
                          "the full data set for a batch in stage 2.")
flags.DEFINE_integer("senate_session",
                     default=113,
                     help="Senate session.")
flags.DEFINE_integer("print_steps",
                     default=100,
                     help="Number of steps to print and save results.")
flags.DEFINE_integer("seed",
                     default=123,
                     help="Random seed to be used.")
FLAGS = flags.FLAGS


def standardize(x):
  return (x - np.mean(x)) / np.std(x)


def build_input_pipeline(data_dir, batch_size, random_state):
  """Load data and build iterator for minibatches.
  
  Args:
    data_dir: The directory where the data is located. There must be five
      files inside the rep: `counts.npz`, `author_indices.npy`, 
      `debate_indices.npy`, `author_map.txt`, and `vocabulary.txt`.
    batch_size: The batch size to use for training.
    random_state: A NumPy `RandomState` object, used to shuffle the data.
  """
  counts = sparse.load_npz(os.path.join(data_dir, "counts.npz"))
  dataset_size, num_words = counts.shape
  author_indices = np.load(
      os.path.join(data_dir, "author_indices.npy")).astype(np.int32)
  num_authors = np.max(author_indices + 1) 
  debate_indices = np.load(
      os.path.join(data_dir, "debate_indices.npy")).astype(np.int32)
  num_debates = np.max(debate_indices + 1)
  author_map = np.loadtxt(os.path.join(data_dir, "author_map.txt"),
                          dtype=str, 
                          delimiter="\n")
  
  def get_row_py_func(idx):
    def get_row_python(idx_py):
      batch_counts = np.squeeze(np.array(counts[idx_py].todense()), axis=0)
      return batch_counts
    py_func = tf.py_func(get_row_python, [idx], tf.float32, stateful=False)
    py_func.set_shape((num_words,))
    return py_func
  
  indices = random_state.permutation(dataset_size)
  shuffled_author_indices = author_indices[indices]
  shuffled_debate_indices = debate_indices[indices]
  dataset = tf.data.Dataset.from_tensor_slices(
      (indices, shuffled_author_indices, shuffled_debate_indices))
  dataset = dataset.map(lambda index, author, debate: (
      get_row_py_func(index), 
      author,
      debate))
  batches = dataset.repeat().batch(batch_size).prefetch(batch_size)
  iterator = batches.make_one_shot_iterator()
  vocabulary = np.loadtxt(os.path.join(data_dir, "vocabulary.txt"), 
                          dtype=str, 
                          delimiter="\n")
  return (iterator, vocabulary, author_map, num_words, num_authors, 
          num_debates, dataset_size, author_indices, debate_indices)


def print_polarities(polarity_mean, vocabulary):
  """Sort words by polarity for Tensorboard.
  
  Args:
    polarity_mean: The mean of the polarity variational parameter, a NumPy 
      matrix with shape [num_words, num_debates].
    vocabulary: A list of the vocabulary with shape [num_words].

  Returns:
    polarities: A list of the highest polarity words for a sample of debates.
  """
  num_words_to_print = 10
  num_debates_to_print = 50
  # polarity_mean has shape [num_words, num_debates]
  most_negative_words = np.argsort(
      polarity_mean[:, :num_debates_to_print],
      axis=0)[:num_words_to_print].T
  most_positive_words = np.argsort(
      -polarity_mean[:, :num_debates_to_print],
      axis=0)[:num_words_to_print].T
  
  polarities = []
  for debate_idx in range(num_debates_to_print):
    negative_start_string = "Negative {}:".format(debate_idx)
    negative_row = [vocabulary[word] for word in 
                    most_negative_words[debate_idx]]
    negative_row_string = ", ".join(negative_row)
    negative_string = " ".join([negative_start_string, negative_row_string])
    
    positive_start_string = "Positive {}:".format(debate_idx)
    positive_row = [vocabulary[word] for word in 
                    most_positive_words[debate_idx]]
    positive_row_string = ", ".join(positive_row)
    positive_string = " ".join([positive_start_string, positive_row_string])
    
    polarities.append("  \n".join([negative_string, positive_string]))
  return np.array(polarities)


def print_ideal_points(stage_1_author_factor_loc, author_map):
  """Sort authors by ideal points and print for Tensorboard."""
  return ", ".join(author_map[np.argsort(stage_1_author_factor_loc)])


def get_log_prior(samples, prior='normal', scale=1.):
  """Return log prior of samples.
  
  Args:
    samples: A `Tensor` with shape `[num_samples, :, num_debates]`.
    prior: String denoting the type of distribution. Either "normal" or
      "gamma".
    scale: Scale term for normal prior.
  
  Returns:
    log_prior: A `Tensor` with shape `[num_samples]`, with the log priors 
      summed across batch- and word-dimensions.
  """
  if prior == "normal":
    prior_distribution = tfp.distributions.Normal(loc=0., scale=scale)
  elif prior == "gamma":
    prior_distribution = tfp.distributions.Gamma(concentration=1., rate=1.)
  else:
    raise ValueError("Unrecognized prior distribution.")
  if len(samples.shape) == 2:
    axes = [1]
  elif len(samples.shape) == 3:
    axes = [1, 2]
  else:
    raise ValueError("Incorrect shape for log prior samples.")
  log_prior = tf.reduce_sum(prior_distribution.log_prob(samples), axis=axes)
  return log_prior


def get_stage_1_elbo(counts,
                     author_indices,
                     debate_indices,
                     author_factor_distribution,
                     author_intercept_distribution,
                     word_factor_distribution,
                     word_intercept_distribution,
                     dataset_size,
                     batch_size,
                     num_samples=1):
  """Approximate first stage ELBO using reparameterization.
  
  Args:
    counts: A matrix with shape `[batch_size, num_words]`.
    author_indices: An int-vector with shape `[batch_size]`.
    debate_indices: An int-vector with shape `[batch_size]`.
    author_factor_distribution: A real `Distribution` object with parameter 
      shape `[num_authors, num_debates]`.
    author_intercept_distribution: A real `Distribution` object with parameter 
      shape `[num_authors, num_debates]`.
    word_factor_distribution: A real `Distribution` object with parameter shape 
      `[num_words, num_debates]`.
    word_intercept_distribution: A real `Distribution` object with parameter 
      shape `[num_words, num_debates]`.
    dataset_size: The number of rows in the total data set (used to calculate
      log-likelihood scaling factor).
    batch_size: Batch size (used to calculate log-likelihood scaling factor).
    num_samples: Number of Monte-Carlo samples.
  
  Returns:
    elbo: A scalar representing a Monte-Carlo sample of the ELBO. This value is
      averaged across samples and summed across batches.
  """
  ideal_point_samples = author_factor_distribution.sample(num_samples)
  author_intercept_samples = author_intercept_distribution.sample(
      num_samples)
  polarity_samples = word_factor_distribution.sample(num_samples)
  word_intercept_samples = word_intercept_distribution.sample(num_samples)

  # From [1]:
  """
  We place normal priors with mean 0 on all of the sets of the parameters in
  the model, with standard deviation 1 for the debate-specific positions psi
  and 5 for the other model parameters.
  """
  ideal_point_log_prior = get_log_prior(ideal_point_samples)
  author_intercept_log_prior = get_log_prior(author_intercept_samples,
                                             scale=5.)
  polarity_log_prior = get_log_prior(polarity_samples, scale=5.)
  word_intercept_log_prior = get_log_prior(word_intercept_samples, scale=5.)
  log_prior = (ideal_point_log_prior + 
               author_intercept_log_prior + 
               polarity_log_prior + 
               word_intercept_log_prior)
  
  ideal_point_entropy = -tf.reduce_sum(
      author_factor_distribution.log_prob(ideal_point_samples),
      axis=[1, 2])
  author_intercept_entropy = -tf.reduce_sum(
      author_intercept_distribution.log_prob(author_intercept_samples),
      axis=[1, 2])
  polarity_entropy = -tf.reduce_sum(
      word_factor_distribution.log_prob(polarity_samples),
      axis=[1, 2])
  word_intercept_entropy = -tf.reduce_sum(
      word_intercept_distribution.log_prob(word_intercept_samples),
      axis=[1, 2])
  entropy = (ideal_point_entropy + 
             author_intercept_entropy + 
             polarity_entropy + 
             word_intercept_entropy)

  indices_2d = tf.concat(
      [author_indices[:, tf.newaxis], debate_indices[:, tf.newaxis]], 
      axis=1)
  selected_ideal_points = tf.transpose(
      tf.gather_nd(tf.transpose(ideal_point_samples, [1, 2, 0]), indices_2d), 
      [1, 0])
  selected_author_intercepts = tf.transpose(
      tf.gather_nd(tf.transpose(author_intercept_samples, [1, 2, 0]), 
                   indices_2d),
      [1, 0])
  selected_polarities = tf.transpose(
      tf.gather(polarity_samples, debate_indices, axis=2), 
      [0, 2, 1])
  selected_word_intercepts = tf.transpose(
      tf.gather(word_intercept_samples, debate_indices, axis=2), 
      [0, 2, 1])

  rate = tf.exp(
      selected_author_intercepts[:, :, tf.newaxis] + 
      selected_word_intercepts + 
      selected_ideal_points[:, :, tf.newaxis] * 
      selected_polarities)
  count_distribution = tfp.distributions.Poisson(rate=rate)
  count_log_likelihood = count_distribution.log_prob(counts)
  count_log_likelihood = tf.reduce_sum(count_log_likelihood, axis=[1, 2])
  # Adjust for the fact that we're only using a minibatch.
  count_log_likelihood = count_log_likelihood * (dataset_size / batch_size)

  elbo = log_prior + count_log_likelihood + entropy
  elbo = tf.reduce_mean(elbo)

  tf.summary.scalar("stage_1_elbo/elbo", elbo)
  tf.summary.scalar("stage_1_elbo/log_prior", tf.reduce_mean(log_prior))
  tf.summary.scalar("stage_1_elbo/count_log_likelihood", 
                    tf.reduce_mean(count_log_likelihood))
  tf.summary.scalar("stage_1_elbo/entropy", tf.reduce_mean(entropy))
  return elbo


def get_stage_2_elbo(flat_fitted_author_factor,
                     ideal_point_distribution,
                     debate_factor_distribution,
                     debate_intercept_distribution,
                     variance_distribution,
                     author_indices,
                     debate_indices,
                     num_samples=1):
  """Approximate second stage ELBO using reparameterization.
  
  Args:
    fitted_author_factor: A tensor with shape `[dataset_size]`, containing the
      fitted per-debate ideal points from stage 1. This variable does not
      receive gradients, so it is no longer being trained.
    ideal_point_distribution: A real `Distribution` object with parameter 
      shape `[num_authors]`.
    debate_factor_distribution: A real `Distribution` object with parameter 
      shape `[num_debates]`.
    debate_intercept_distribution: A real `Distribution` object with parameter 
      shape  `[num_debates]`.
    variance_distribution: A positive `Distribution` object with parameter 
      shape `[num_authors]`.
    author_indices: A vector with shape `[dataset_size]`.
    debate_indices: A vector with shape `[dataset_size]`.
    num_samples: Number of Monte-Carlo samples.
  
  Returns:
    elbo: A scalar representing a Monte-Carlo sample of the ELBO. This value is
      averaged across samples and summed across batches.
  """
  ideal_point_samples = ideal_point_distribution.sample(num_samples)
  debate_factor_samples = debate_factor_distribution.sample(num_samples)
  debate_intercept_samples = debate_intercept_distribution.sample(num_samples)
  variance_samples = variance_distribution.sample(num_samples)
  
  # Following the prior distributions in [1].
  ideal_point_log_prior = get_log_prior(ideal_point_samples)
  debate_factor_log_prior = get_log_prior(debate_factor_samples, scale=0.5)
  debate_intercept_log_prior = get_log_prior(debate_intercept_samples, 
                                             scale=0.5)
  variance_log_prior = get_log_prior(variance_samples, prior='gamma')
  log_prior = (ideal_point_log_prior + 
               debate_factor_log_prior + 
               debate_intercept_log_prior + 
               variance_log_prior)
  
  ideal_point_entropy = -tf.reduce_sum(
      ideal_point_distribution.log_prob(ideal_point_samples),
      axis=1)
  debate_factor_entropy = -tf.reduce_sum(
      debate_factor_distribution.log_prob(debate_factor_samples),
      axis=1)
  debate_intercept_entropy = -tf.reduce_sum(
      debate_intercept_distribution.log_prob(debate_intercept_samples),
      axis=1)
  variance_entropy = -tf.reduce_sum(
      variance_distribution.log_prob(variance_samples),
      axis=1)
  entropy = (ideal_point_entropy + 
             debate_factor_entropy + 
             debate_intercept_entropy + 
             variance_entropy)

  selected_debate_intercepts = tf.gather(debate_intercept_samples, 
                                         debate_indices, 
                                         axis=1)
  selected_ideal_points = tf.gather(ideal_point_samples, 
                                    author_indices, 
                                    axis=1)
  selected_debate_factors = tf.gather(debate_factor_samples, 
                                      debate_indices, 
                                      axis=1)
  selected_variance_samples = tf.gather(variance_samples, 
                                        author_indices, 
                                        axis=1)

  output_mean = (selected_debate_intercepts + 
                 selected_ideal_points * 
                 selected_debate_factors)
  output_scale = selected_variance_samples
  author_factor_distribution = tfp.distributions.Normal(loc=output_mean,
                                                        scale=output_scale)
  author_factor_log_likelihood = author_factor_distribution.log_prob(
      flat_fitted_author_factor)
  author_factor_log_likelihood = tf.reduce_sum(author_factor_log_likelihood, 
                                               axis=1)

  elbo = log_prior + author_factor_log_likelihood + entropy
  elbo = tf.reduce_mean(elbo)

  tf.summary.scalar("stage_2_elbo/elbo", elbo)
  tf.summary.scalar("stage_2_elbo/log_prior", tf.reduce_mean(log_prior))
  tf.summary.scalar("stage_2_elbo/author_factor_log_likelihood", 
                    tf.reduce_mean(author_factor_log_likelihood))
  tf.summary.scalar("stage_2_elbo/entropy", tf.reduce_mean(entropy))
  return elbo


def main(argv):
  print("Starting Senate Session {}".format(FLAGS.senate_session))
  del argv
  tf.set_random_seed(FLAGS.seed)
  random_state = np.random.RandomState(FLAGS.seed)
  
  project_dir = os.path.abspath(
      os.path.join(os.path.dirname(__file__), os.pardir)) 
  source_dir = os.path.join(project_dir, "data/{}/wordshoal/{}".format(
      FLAGS.data, FLAGS.senate_session))
  
  data_dir = os.path.join(source_dir, "clean")    
  save_dir = os.path.join(source_dir, "wordshoal-fits")
  if tf.gfile.Exists(save_dir):
    tf.logging.warn("Deleting old log directory at {}".format(save_dir))
    tf.gfile.DeleteRecursively(save_dir)
  tf.gfile.MakeDirs(save_dir)
  
  param_save_dir = os.path.join(save_dir, "params/")
  if not tf.gfile.Exists(param_save_dir):
    tf.gfile.MakeDirs(param_save_dir)

  np.warnings.filterwarnings('ignore')  # suppress scipy.sparse warnings.
  (iterator, vocabulary, author_map, num_words, num_authors,
   num_debates, dataset_size, all_author_indices, 
   all_debate_indices) = build_input_pipeline(data_dir, 
                                              FLAGS.batch_size, 
                                              random_state)
  counts, batch_author_indices, batch_debate_indices = iterator.get_next()
  
  stage_1_author_factor_loc = tf.get_variable(
      "stage_1_author_factor_loc",
      shape=[num_authors, num_debates],
      dtype=tf.float32)
  stage_1_author_intercept_loc = tf.get_variable(
      "stage_1_author_intercept_loc",
      shape=[num_authors, num_debates],
      dtype=tf.float32)
  stage_1_word_factor_loc = tf.get_variable( 
      "stage_1_word_factor_loc",
      shape=[num_words, num_debates],
      dtype=tf.float32)
  stage_1_word_intercept_loc = tf.get_variable(
      "stage_1_word_intercept_loc",
      shape=[num_words, num_debates],
      dtype=tf.float32)

  stage_1_author_factor_scale_logit = tf.get_variable(
      "stage_1_author_factor_scale_logit",
      shape=[num_authors, num_debates],
      dtype=tf.float32)
  stage_1_author_intercept_scale_logit = tf.get_variable(
      "stage_1_author_intercept_scale_logit",
      shape=[num_authors, num_debates],
      dtype=tf.float32)
  stage_1_word_factor_scale_logit = tf.get_variable( 
      "stage_1_word_factor_scale_logit",
      shape=[num_words, num_debates],
      dtype=tf.float32)
  stage_1_word_intercept_scale_logit = tf.get_variable(
      "stage_1_word_intercept_scale_logit",
      shape=[num_words, num_debates],
      dtype=tf.float32)

  stage_1_author_factor_scale = tf.nn.softplus(
      stage_1_author_factor_scale_logit)
  stage_1_author_intercept_scale = tf.nn.softplus(
      stage_1_author_intercept_scale_logit)
  stage_1_word_factor_scale = tf.nn.softplus(
      stage_1_word_factor_scale_logit)
  stage_1_word_intercept_scale = tf.nn.softplus(
      stage_1_word_intercept_scale_logit)

  tf.summary.histogram("stage_1_params/author_factor_loc", 
                       stage_1_author_factor_loc)
  tf.summary.histogram("stage_1_params/author_intercept_loc", 
                       stage_1_author_intercept_loc)
  tf.summary.histogram("stage_1_params/word_factor_loc", 
                       stage_1_word_factor_loc)
  tf.summary.histogram("stage_1_params/word_intercept_loc", 
                       stage_1_word_intercept_loc)
  tf.summary.histogram("stage_1_params/author_intercept_scale", 
                       stage_1_author_intercept_scale)
  tf.summary.histogram("stage_1_params/author_factor_scale", 
                       stage_1_author_factor_scale)
  tf.summary.histogram("stage_1_params/word_factor_scale", 
                       stage_1_word_factor_scale)
  tf.summary.histogram("stage_1_params/word_intercept_scale", 
                       stage_1_word_intercept_scale)

  stage_1_author_factor_distribution = tfp.distributions.Normal(
      loc=stage_1_author_factor_loc,
      scale=stage_1_author_factor_scale) 
  stage_1_author_intercept_distribution = tfp.distributions.Normal(
      loc=stage_1_author_intercept_loc,
      scale=stage_1_author_intercept_scale)
  stage_1_word_factor_distribution = tfp.distributions.Normal(
      loc=stage_1_word_factor_loc,
      scale=stage_1_word_factor_scale) 
  stage_1_word_intercept_distribution = tfp.distributions.Normal(
      loc=stage_1_word_intercept_loc,
      scale=stage_1_word_intercept_scale)
    
  stage_1_elbo = get_stage_1_elbo(counts,
                                  batch_author_indices,
                                  batch_debate_indices,
                                  stage_1_author_factor_distribution,
                                  stage_1_author_intercept_distribution,
                                  stage_1_word_factor_distribution,
                                  stage_1_word_intercept_distribution,
                                  dataset_size,
                                  FLAGS.batch_size,
                                  num_samples=FLAGS.num_samples)
  stage_1_loss = -stage_1_elbo
  
  stage_1_optim = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
  stage_1_train_op = stage_1_optim.minimize(
      stage_1_loss,
      var_list=[stage_1_author_factor_loc, stage_1_author_intercept_loc,
                stage_1_word_factor_loc, stage_1_word_intercept_loc,
                stage_1_author_factor_scale_logit, 
                stage_1_author_intercept_scale_logit,
                stage_1_word_factor_scale_logit,
                stage_1_word_intercept_scale_logit])
  
  stage_2_ideal_point_loc = tf.get_variable(
      "stage_2_ideal_point_loc",
      shape=[num_authors],
      dtype=tf.float32)
  stage_2_debate_factor_loc = tf.get_variable( 
      "stage_2_debate_factor_loc",
      shape=[num_debates],
      dtype=tf.float32)
  stage_2_debate_intercept_loc = tf.get_variable(
      "stage_2_debate_intercept_loc",
      shape=[num_debates],
      dtype=tf.float32)
  stage_2_variance_loc = tf.get_variable(
      "stage_2_variance_loc",
      shape=[num_authors],
      dtype=tf.float32)

  stage_2_ideal_point_scale_logit = tf.get_variable(
      "stage_2_ideal_point_scale_logit",
      shape=[num_authors],
      dtype=tf.float32)
  stage_2_debate_factor_scale_logit = tf.get_variable( 
      "stage_2_debate_factor_scale_logit",
      shape=[num_debates],
      dtype=tf.float32)
  stage_2_debate_intercept_scale_logit = tf.get_variable(
      "stage_2_debate_intercept_scale_logit",
      shape=[num_debates],
      dtype=tf.float32)
  stage_2_variance_scale_logit = tf.get_variable(
      "stage_2_variance_scale_logit",
      shape=[num_authors],
      dtype=tf.float32)

  stage_2_ideal_point_scale = tf.nn.softplus(stage_2_ideal_point_scale_logit)
  stage_2_debate_factor_scale = tf.nn.softplus(
      stage_2_debate_factor_scale_logit)
  stage_2_debate_intercept_scale = tf.nn.softplus(
      stage_2_debate_intercept_scale_logit)
  stage_2_variance_scale = tf.nn.softplus(stage_2_variance_scale_logit)

  tf.summary.histogram("stage_2_params/ideal_point_loc", 
                       stage_2_ideal_point_loc)
  tf.summary.histogram("stage_2_params/debate_factor_loc", 
                       stage_2_debate_factor_loc)
  tf.summary.histogram("stage_2_params/debate_intercept_loc", 
                       stage_2_debate_intercept_loc)
  tf.summary.histogram("stage_2_params/variance_loc", 
                       stage_2_variance_loc)
  tf.summary.histogram("stage_2_params/ideal_point_scale", 
                       stage_2_ideal_point_scale)
  tf.summary.histogram("stage_2_params/debate_factor_scale", 
                       stage_2_debate_factor_scale)
  tf.summary.histogram("stage_2_params/debate_intercept_scale", 
                       stage_2_debate_intercept_scale)
  tf.summary.histogram("stage_2_params/variance_scale", 
                       stage_2_variance_scale)

  stage_2_ideal_point_distribution = tfp.distributions.Normal(
      loc=stage_2_ideal_point_loc,
      scale=stage_2_ideal_point_scale) 
  stage_2_debate_factor_distribution = tfp.distributions.Normal(
      loc=stage_2_debate_factor_loc,
      scale=stage_2_debate_factor_scale)
  stage_2_debate_intercept_distribution = tfp.distributions.Normal(
      loc=stage_2_debate_intercept_loc,
      scale=stage_2_debate_intercept_scale)
  stage_2_variance_distribution = tfp.distributions.LogNormal(
      loc=stage_2_variance_loc,
      scale=stage_2_variance_scale)

  fitted_author_factor = stage_1_author_factor_loc
  # We don't want to factorize the entire [num_authors, num_debates] matrix.
  # Not all authors speak at every debate, so we only factorize the non-missing
  # values, rather than the random values.
  # Also, we use the full batch in the second stage becuase it fits into
  # memory. To do this, we need to use `all_author_indices` and 
  # `all_debate_indices` rather than `batch_author_indices` and 
  # `batch_debate_indices`.
  indices_2d = tf.concat(
      [all_author_indices[:, np.newaxis], all_debate_indices[:, np.newaxis]],
      axis=1)
  # We flatten, so `flat_fitted_author_factor` has shape `[dataset_size]`.
  flat_fitted_author_factor = tf.gather_nd(fitted_author_factor, indices_2d)
  
  # We stop the gradient on the fitted author factor to make sure it is not
  # being trained in the second stage.
  stage_2_elbo = get_stage_2_elbo(tf.stop_gradient(flat_fitted_author_factor),
                                  stage_2_ideal_point_distribution,
                                  stage_2_debate_factor_distribution,
                                  stage_2_debate_intercept_distribution,
                                  stage_2_variance_distribution,
                                  all_author_indices,
                                  all_debate_indices,
                                  num_samples=FLAGS.num_samples)
  stage_2_loss = -stage_2_elbo

  stage_2_optim = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
  stage_2_train_op = stage_2_optim.minimize(
      stage_2_loss,
      var_list=[stage_2_ideal_point_loc, stage_2_debate_factor_loc,
                stage_2_debate_intercept_loc, stage_2_variance_loc,
                stage_2_ideal_point_scale_logit, 
                stage_2_debate_factor_scale_logit,
                stage_2_debate_intercept_scale_logit,
                stage_2_variance_scale_logit])

  polarities = tf.py_func(
      functools.partial(print_polarities, vocabulary=vocabulary),
      [stage_1_word_factor_loc],
      tf.string,
      stateful=False)
  tf.summary.text("polarities", polarities)
  ideal_point_list = tf.py_func(
      functools.partial(print_ideal_points, author_map=author_map),
      [stage_2_ideal_point_loc],
      tf.string, stateful=False)
  tf.summary.text("ideal_points", ideal_point_list) 
  
  summary = tf.summary.merge_all()
  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(save_dir, sess.graph)
    sess.run(init)
    # Train Stage 1
    for step in range(FLAGS.max_steps):
      start_time = time.time()
      (_, stage_1_elbo_val) = sess.run([stage_1_train_op, stage_1_elbo])
      duration = time.time() - start_time
      if step % FLAGS.print_steps == 0:
        print("Stage 1: Step: {:>3d} ELBO: {:.3f} ({:.3f} sec)".format(
            step, stage_1_elbo_val, duration))
                     
        summary_str = sess.run(summary)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
      
    # Train Stage 2
    for step in range(FLAGS.max_steps):
      start_time = time.time()
      (_, stage_2_elbo_val) = sess.run([stage_2_train_op, stage_2_elbo])
      duration = time.time() - start_time
      if step % FLAGS.print_steps == 0:
        print("Stage 2: Step: {:>3d} ELBO: {:.3f} ({:.3f} sec)".format(
            step, stage_2_elbo_val, duration))
                     
        summary_str = sess.run(summary)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
      
      if step % 500 == 0:
        (stage_1_author_factor_loc_val, stage_1_author_factor_scale_val,
         stage_2_ideal_point_loc_val, stage_2_ideal_point_scale_val, 
         stage_2_debate_factor_loc_val, stage_2_debate_factor_scale_val, 
         stage_2_variance_loc_val, stage_2_variance_scale_val) = sess.run([
             stage_1_author_factor_loc, stage_1_author_factor_scale,
             stage_2_ideal_point_loc, stage_2_ideal_point_scale, 
             stage_2_debate_factor_loc, stage_2_debate_factor_scale, 
             stage_2_variance_loc, stage_2_variance_scale])
        np.save(os.path.join(param_save_dir, "stage_1_author_factor_loc"), 
                stage_1_author_factor_loc_val)
        np.save(os.path.join(param_save_dir, "stage_1_author_factor_scale"), 
                stage_1_author_factor_scale_val)
        np.save(os.path.join(param_save_dir, "stage_2_ideal_point_loc"), 
                stage_2_ideal_point_loc_val)
        np.save(os.path.join(param_save_dir, "stage_2_ideal_point_scale"), 
                stage_2_ideal_point_scale_val)
        np.save(os.path.join(param_save_dir, "stage_2_debate_factor_loc"), 
                stage_2_debate_factor_loc_val)
        np.save(os.path.join(param_save_dir, "stage_2_debate_factor_scale"), 
                stage_2_debate_factor_scale_val)
        np.save(os.path.join(param_save_dir, "stage_2_variance_loc"), 
                stage_2_variance_loc_val)
        np.save(os.path.join(param_save_dir, "stage_2_variance_scale"), 
                stage_2_variance_scale_val)


if __name__ == "__main__":
  tf.app.run()
