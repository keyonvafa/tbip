"""Use Wordfish model to estimate ideology from Senate speeches.

Unlike the model in [1], we only have a single time period and posit Gaussian
priors on each latent variable, performing variational inference with 
reparameterization gradients. The model is thus

y_{ij} ~ Pois(exp(alpha_i + psi_j + beta_j * x_i))

where x_i is the ideal point, and beta_j is the word polarity. 

We perform inference with variational inference, using reparameterization
gradients to approximate the ELBO. We note that this is different from the
inference procedure used in [1]. 

#### References
[1]: Jonathan B. Slapin, Sven-Oliver Proksch. A Scaling Model for Estimating 
     Time-Series Party Positions from Texts. In _American Journal of Political 
     Science_, July 2008. 
     www.wordfish.org/uploads/1/2/9/8/12985397/slapin_proksch_ajps_2008.pdf
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
                     default=50000,
                     help="Number of training steps to run.")
flags.DEFINE_integer("num_samples",
                     default=1,
                     help="Number of samples for ELBO approximation.")
flags.DEFINE_enum("data",
                  default="senate-speech-comparisons",
                  enum_values=["senate-speech-comparisons", 
                               "senate-tweets-114"],
                  help="Data set used.")
flags.DEFINE_integer("senate_session",
                     default=113,
                     help="Senate session (used only for "
                          "'senate-speech-comparisons').")
flags.DEFINE_integer("print_steps",
                     default=1000,
                     help="Number of steps to print and save results.")
flags.DEFINE_integer("seed",
                     default=123,
                     help="Random seed to be used.")
FLAGS = flags.FLAGS


def standardize(x):
  return (x - np.mean(x)) / np.std(x)


def build_input_pipeline(data_dir, data):
  """Load data and build iterator.
  
  Args:
    data_dir: The directory where the data is located. There must be three
      files inside the rep: `counts.npz`, `author_map.txt`, and 
      `vocabulary.txt`. (If the data source is `senate-tweets-114`, the counts
      data is stored in `wordfish_counts.npz`.)
    data: Name of data source to be used.
    random_state: A NumPy `RandomState` object, used to shuffle the data.
  """
  # We keep both the collapsed (for Wordfish) and uncollapsed (for the TBIP)
  # counts on 'senate-tweets-114/clean/', so we modify the naming convention.
  if data == "senate-tweets-114":
    counts = sparse.load_npz(os.path.join(data_dir, "wordfish_counts.npz"))
  else:
    counts = sparse.load_npz(os.path.join(data_dir, "counts.npz"))
  num_authors, num_words = counts.shape
  author_map = np.loadtxt(os.path.join(data_dir, "author_map.txt"),
                          dtype=str, 
                          delimiter="\n")
  batch_size = num_authors
  
  def get_row_py_func(idx):
    def get_row_python(idx_py):
      batch_counts = np.squeeze(np.array(counts[idx_py].todense()), axis=0)
      return batch_counts
    py_func = tf.py_func(get_row_python, [idx], tf.float32, stateful=False)
    py_func.set_shape((num_words,))
    return py_func
  
  authors = np.arange(num_authors)
  dataset = tf.data.Dataset.from_tensor_slices((authors))
  dataset = dataset.map(lambda author: (author, get_row_py_func(author)))
  batches = dataset.repeat().batch(batch_size).prefetch(batch_size)
  iterator = batches.make_one_shot_iterator()
  vocabulary = np.loadtxt(os.path.join(data_dir, "vocabulary.txt"), 
                          dtype=str, 
                          delimiter="\n",
                          comments="<!-")
  return iterator, vocabulary, author_map, num_words, num_authors


def print_polarities(polarity_mean, vocabulary):
  """Sort words by polarity for Tensorboard.
  
  Args:
    polarity_mean: The mean of the polarity variational parameter, a NumPy 
      array with shape [num_words].
    vocabulary: A list of the vocabulary with shape [num_words].

  Returns:
    polarities: A list of the highest polarity words.
  """
  num_print = 25
  most_negative_words = np.argsort(polarity_mean)[:num_print]
  most_positive_words = np.argsort(-polarity_mean)[:num_print]

  negative_start_string = "Negative:"
  negative_row = [vocabulary[word] for word in most_negative_words]
  negative_row_string = ", ".join(negative_row)
  negative_string = " ".join([negative_start_string, negative_row_string])

  positive_start_string = "Positive:"
  positive_row = [vocabulary[word] for word in most_positive_words]
  positive_row_string = ", ".join(positive_row)
  positive_string = " ".join([positive_start_string, positive_row_string])
  
  polarities = "  \n".join([negative_string, positive_string])
  return np.array(polarities)


def print_ideal_points(ideal_point_loc, author_map):
  """Sort authors by ideal points and print for Tensorboard."""
  return ", ".join(author_map[np.argsort(ideal_point_loc)])


def get_log_prior(samples):
  """Return log prior of sampled Gaussians.
  
  Args:
    samples: A `Tensor` with shape `[num_samples, :]`.
  
  Returns:
    log_prior: A `Tensor` with shape `[num_samples]`, with the log priors 
      summed across batch- and word-dimensions.
  """
  prior_distribution = tfp.distributions.Normal(loc=0., scale=1.)
  log_prior = tf.reduce_sum(prior_distribution.log_prob(samples), axis=[1])
  return log_prior


def get_elbo(counts,
             author_indices,
             ideal_point_distribution,
             author_intercept_distribution,
             polarity_distribution,
             word_intercept_distribution,
             num_authors,
             batch_size,
             num_samples=1):
  """Approximate ELBO using reparameterization.
  
  Args:
    counts: A matrix with shape `[batch_size, num_words]`.
    author_indices: An int-vector with shape `[batch_size]`.
    ideal_point_distribution: A real `Distribution` object with parameter 
      shape `[num_authors]`.
    author_intercept_distribution: A real `Distribution` object with parameter 
      shape `[num_authors]`.
    polarity_distribution: A real `Distribution` object with parameter shape 
      `[num_words]`.
    word_intercept_distribution: A real `Distribution` object with parameter 
      shape `[num_words]`.
    num_authors: The number of authors in the total data set (used to 
      calculate log-likelihood scaling factor).
    batch_size: Batch size (used to calculate log-likelihood scaling factor).
    num_samples: Number of Monte-Carlo samples.
  
  Returns:
    elbo: A scalar representing a Monte-Carlo sample of the ELBO. This value is
      averaged across samples and summed across batches.
  """
  ideal_point_samples = ideal_point_distribution.sample(num_samples)
  author_intercept_samples = author_intercept_distribution.sample(
      num_samples)
  polarity_samples = polarity_distribution.sample(num_samples)
  word_intercept_samples = word_intercept_distribution.sample(num_samples)
  
  ideal_point_log_prior = get_log_prior(ideal_point_samples)
  author_intercept_log_prior = get_log_prior(author_intercept_samples)
  polarity_log_prior = get_log_prior(polarity_samples)
  word_intercept_log_prior = get_log_prior(word_intercept_samples)
  log_prior = (ideal_point_log_prior + 
               author_intercept_log_prior + 
               polarity_log_prior + 
               word_intercept_log_prior)
  
  ideal_point_entropy = -tf.reduce_sum(
      ideal_point_distribution.log_prob(ideal_point_samples),
      axis=1)
  author_intercept_entropy = -tf.reduce_sum(
      author_intercept_distribution.log_prob(author_intercept_samples),
      axis=1)
  polarity_entropy = -tf.reduce_sum(
      polarity_distribution.log_prob(polarity_samples),
      axis=1)
  word_intercept_entropy = -tf.reduce_sum(
      word_intercept_distribution.log_prob(word_intercept_samples),
      axis=1)
  entropy = (ideal_point_entropy + 
             author_intercept_entropy + 
             polarity_entropy + 
             word_intercept_entropy)

  selected_ideal_points = tf.gather(ideal_point_samples, 
                                    author_indices, 
                                    axis=1)
  selected_author_intercepts = tf.gather(author_intercept_samples, 
                                         author_indices, 
                                         axis=1)
  rate = tf.exp(
      selected_author_intercepts[:, :, tf.newaxis] + 
      word_intercept_samples[:, tf.newaxis, :] + 
      selected_ideal_points[:, :, tf.newaxis] * 
      polarity_samples[:, tf.newaxis, :])
  count_distribution = tfp.distributions.Poisson(rate=rate)
  count_log_likelihood = count_distribution.log_prob(counts)
  count_log_likelihood = tf.reduce_sum(count_log_likelihood, axis=[1, 2])
  # Adjust for the fact that we're only using a minibatch.
  count_log_likelihood = count_log_likelihood * (num_authors / batch_size)

  elbo = log_prior + count_log_likelihood + entropy
  elbo = tf.reduce_mean(elbo)

  tf.summary.scalar("elbo/elbo", elbo)
  tf.summary.scalar("elbo/log_prior", tf.reduce_mean(log_prior))
  tf.summary.scalar("elbo/count_log_likelihood", 
                    tf.reduce_mean(count_log_likelihood))
  tf.summary.scalar("elbo/entropy", tf.reduce_mean(entropy))
  return elbo


def main(argv):
  del argv
  tf.set_random_seed(FLAGS.seed)
  
  project_dir = os.path.abspath(
      os.path.join(os.path.dirname(__file__), os.pardir)) 
  source_dir = os.path.join(project_dir, "data/{}".format(FLAGS.data))
  # For model comparisons, we must also specify a Senate session.
  if FLAGS.data == "senate-speech-comparisons":
    source_dir = os.path.join(
        source_dir, "wordfish/{}".format(FLAGS.senate_session))
  
  data_dir = os.path.join(source_dir, "clean")    
  save_dir = os.path.join(source_dir, "wordfish-fits")
  if tf.gfile.Exists(save_dir):
    tf.logging.warn("Deleting old log directory at {}".format(save_dir))
    tf.gfile.DeleteRecursively(save_dir)
  tf.gfile.MakeDirs(save_dir)
  
  param_save_dir = os.path.join(save_dir, "params/")
  if not tf.gfile.Exists(param_save_dir):
    tf.gfile.MakeDirs(param_save_dir)

  np.warnings.filterwarnings('ignore')  # suppress scipy.sparse warnings.
  (iterator, vocabulary, author_map, 
   num_words, num_authors) = build_input_pipeline(data_dir, FLAGS.data)
  batch_size = num_authors
  
  author_indices, counts = iterator.get_next()
  # Initialize variational parameters.
  ideal_point_loc = tf.get_variable(
      "ideal_point_loc",
      shape=[num_authors],
      dtype=tf.float32)
  author_intercept_loc = tf.get_variable(
      "author_intercept_loc",
      shape=[num_authors],
      dtype=tf.float32)
  polarity_loc = tf.get_variable( 
      "polarity_loc",
      shape=[num_words],
      dtype=tf.float32)
  word_intercept_loc = tf.get_variable(
      "word_intercept_loc",
      shape=[num_words],
      dtype=tf.float32)

  ideal_point_scale_logit = tf.get_variable(
      "ideal_point_scale_logit",
      shape=[num_authors],
      dtype=tf.float32)
  author_intercept_scale_logit = tf.get_variable(
      "author_intercept_scale_logit",
      shape=[num_authors],
      dtype=tf.float32)
  polarity_scale_logit = tf.get_variable( 
      "polarity_scale_logit",
      shape=[num_words],
      dtype=tf.float32)
  word_intercept_scale_logit = tf.get_variable(
      "word_intercept_scale_logit",
      shape=[num_words],
      dtype=tf.float32)

  ideal_point_scale = tf.nn.softplus(ideal_point_scale_logit)
  author_intercept_scale = tf.nn.softplus(author_intercept_scale_logit)
  polarity_scale = tf.nn.softplus(polarity_scale_logit)
  word_intercept_scale = tf.nn.softplus(word_intercept_scale_logit)

  tf.summary.histogram("params/ideal_point_loc", ideal_point_loc)
  tf.summary.histogram("params/author_intercept_loc", author_intercept_loc)
  tf.summary.histogram("params/polarity_loc", polarity_loc)
  tf.summary.histogram("params/word_intercept_loc", word_intercept_loc)
  tf.summary.histogram("params/author_intercept_scale", 
                       author_intercept_scale)
  tf.summary.histogram("params/ideal_point_scale", ideal_point_scale)
  tf.summary.histogram("params/polarity_scale", polarity_scale)
  tf.summary.histogram("params/word_intercept_scale", word_intercept_scale)

  ideal_point_distribution = tfp.distributions.Normal(
      loc=ideal_point_loc,
      scale=ideal_point_scale) 
  author_intercept_distribution = tfp.distributions.Normal(
      loc=author_intercept_loc,
      scale=author_intercept_scale)
  polarity_distribution = tfp.distributions.Normal(
      loc=polarity_loc,
      scale=polarity_scale) 
  word_intercept_distribution = tfp.distributions.Normal(
      loc=word_intercept_loc,
      scale=word_intercept_scale)
  
  elbo = get_elbo(counts,
                  author_indices,
                  ideal_point_distribution,
                  author_intercept_distribution,
                  polarity_distribution,
                  word_intercept_distribution,
                  num_authors,
                  batch_size,
                  num_samples=FLAGS.num_samples)
  loss = -elbo
  tf.summary.scalar("loss", loss)

  optim = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
  train_op = optim.minimize(loss)

  polarities = tf.py_func(
      functools.partial(print_polarities, vocabulary=vocabulary),
      [polarity_loc],
      tf.string,
      stateful=False)
  ideal_point_list = tf.py_func(
      functools.partial(print_ideal_points, author_map=author_map),
      [ideal_point_loc],
      tf.string, stateful=False)
  tf.summary.text("polarities", polarities)
  tf.summary.text("ideal_points", ideal_point_list) 
  
  summary = tf.summary.merge_all()
  init = tf.global_variables_initializer()

  with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(save_dir, sess.graph)
    sess.run(init)
    for step in range(FLAGS.max_steps):
      start_time = time.time()
      (_, elbo_val) = sess.run([train_op, elbo])
      duration = time.time() - start_time
      if step % FLAGS.print_steps == 0:
        print("Step: {:>3d} ELBO: {:.3f} ({:.3f} sec)".format(
            step, elbo_val, duration))
                     
        summary_str = sess.run(summary)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
      
      if step % 1000 == 0:
        (ideal_point_loc_val, ideal_point_scale_val, author_intercept_loc_val,
         author_intercept_scale_val, polarity_loc_val, polarity_scale_val, 
         word_intercept_loc_val, word_intercept_scale_val) = sess.run([
             ideal_point_loc, ideal_point_scale, author_intercept_loc, 
             author_intercept_scale, polarity_loc, polarity_scale, 
             word_intercept_loc, word_intercept_scale])
        
        np.save(os.path.join(param_save_dir, "ideal_point_loc"), 
                ideal_point_loc_val)
        np.save(os.path.join(param_save_dir, "ideal_point_scale"), 
                ideal_point_scale_val)
        np.save(os.path.join(param_save_dir, "author_intercept_loc"), 
                author_intercept_loc_val)
        np.save(os.path.join(param_save_dir, "author_intercept_scale"), 
                author_intercept_scale_val)
        np.save(os.path.join(param_save_dir, "polarity_loc"), 
                polarity_loc_val)
        np.save(os.path.join(param_save_dir, "polarity_scale"), 
                polarity_scale_val)
        np.save(os.path.join(param_save_dir, "word_intercept_loc"), 
                word_intercept_loc_val)
        np.save(os.path.join(param_save_dir, "word_intercept_scale"), 
                word_intercept_scale_val)


if __name__ == "__main__":
  tf.app.run()
