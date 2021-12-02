"""Learn ideal points with the text-based ideal point model (TBIP).

Updates from the original code release:
  1. This code supports using Gamma variational families for the document 
     latent variables (theta) and the objective topic variables (beta). When 
     holding the other variational families fixed, the optimal variational
     distributions are available for these parameters in closed-form, using 
     Coordinate-Ascent Variational Inference (CAVI). When CAVI is being used 
     to optimize document and objective topic variational families, the other 
     variational parameters are optimized with SGD. As a result, the 
     optimization procedure converges more quickly. Moreover, the learned 
     topics are more interpretable, and they do not need to be initialized with
     Poisson factorization.
  2. Rather than using pre-determined author weights to normalize each author's
     word counts by their verbosity, here we learn the weights, which was
     shown to improve performance. These weights are referred to as 
     `author_verbosity`.
  3. This code is in Tensorflow 2, which should make it easier to prototype and
     run than Tensorflow 1.

The directory `data/{data_name}/clean/` should have the following four files:
  1. `counts.npz`: a [num_documents, num_words] sparse matrix containing the
     word counts for each document.
  2. `author_indices.npy`: a [num_documents] vector where each entry is an
     integer in the set {0, 1, ..., num_authors - 1}, indicating the author of 
     the corresponding document in `counts.npz`.
  3. `vocabulary.txt`: a [num_words]-length file where each line is a string
     denoting the corresponding word in the vocabulary.
  4. `author_map.txt`: a [num_authors]-length file where each line is a string
     denoting the name of an author in the corpus.

For more details about the base model, refer to our paper [1]. For more details
about the CAVI updates, refer to [2].

#### References
[1]: Keyon Vafa, Suresh Naidu, David Blei. Text-Based Ideal Points. In 
     _Association for Computational Linguistics_, 2020.
     https://www.aclweb.org/anthology/2020.acl-main.475/

[2]: Prem Gopalan, Jake Hofman, David Blei. Scalable Recommendation with 
     Hierarchical Poisson Factorization. In _Conference on Uncertainty in
     Artifical Intelligence_, 2015. https://arxiv.org/abs/1311.1704
"""
import os
import time

from absl import app
from absl import flags

import numpy as np
import scipy.sparse as sparse
import tensorflow as tf
import tensorflow_probability as tfp

flags.DEFINE_float("learning_rate",
                   default=0.01,
                   help="Adam learning rate.")
flags.DEFINE_integer("num_epochs",
                     default=10000,
                     help="Number of training steps to run.")
flags.DEFINE_integer("num_topics",
                     default=50,
                     help="Number of topics.")
flags.DEFINE_integer("batch_size",
                     default=512,
                     help="Batch size.")
flags.DEFINE_integer("num_samples",
                     default=1,
                     help="Number of samples to use for ELBO approximation.")
flags.DEFINE_enum("counts_transformation",
                  default="nothing",
                  enum_values=["nothing", "log"],
                  help="Transformation used on counts data.")
flags.DEFINE_enum("positive_variational_family",
                  default="gamma",
                  enum_values=["gamma", "lognormal"],
                  help="Variational family used for document intensities "
                       "(theta) and objective topics (beta).")
flags.DEFINE_boolean("cavi",
                     default=True,
                     help="Whether to use coordinate-ascent variational "
                          "inference for the positive random variables (theta "
                          "and beta).")
flags.DEFINE_boolean("pre_initialize_parameters",
                     default=False,
                     help="Whether to use pre-initialized document and topic "
                          "intensities (with Poisson factorization).")
flags.DEFINE_string("data",
                    default="senate-speeches-114",
                    help="Data source being used.")
flags.DEFINE_integer("save_every",
                     default=20,
                     help="Number of epochs after which to save and log")
flags.DEFINE_integer("seed",
                     default=123,
                     help="Random seed to be used.")
flags.DEFINE_string("checkpoint_name",
                    default="tmp",
                    help="Name to be used for saving results.")
flags.DEFINE_boolean("load_checkpoint",
                     default=True,
                     help="Whether to load checkpoint (only if it exists).")
FLAGS = flags.FLAGS


def build_input_pipeline(data_dir,
                         batch_size,
                         random_state,
                         counts_transformation="nothing"):
  """Load data and build iterator for minibatches.

  Args:
    data_dir: The directory where the data is located. There must be four
      files inside the rep: `counts.npz`, `author_indices.npy`,
      `author_map.txt`, and `vocabulary.txt`.
    batch_size: The batch size to use for training.
    random_state: A NumPy `RandomState` object, used to shuffle the data.
    counts_transformation: A string indicating how to transform the counts.
      One of "nothing" or "log".
  """
  counts = sparse.load_npz(os.path.join(data_dir, "counts.npz"))
  num_documents, num_words = counts.shape
  author_indices = np.load(
    os.path.join(data_dir, "author_indices.npy")).astype(np.int32)
  author_map = np.loadtxt(os.path.join(data_dir, "author_map.txt"),
                          dtype=str,
                          delimiter="\n")
  documents = random_state.permutation(num_documents)
  shuffled_author_indices = author_indices[documents]
  shuffled_counts = counts[documents]
  if counts_transformation == "nothing":
    count_values = shuffled_counts.data
  elif counts_transformation == "log":
    count_values = np.round(np.log(1 + shuffled_counts.data))
  else:
    raise ValueError("Unrecognized counts transformation.")
  shuffled_counts = tf.SparseTensor(
    indices=np.array(shuffled_counts.nonzero()).T,
    values=count_values,
    dense_shape=shuffled_counts.shape)
  dataset = tf.data.Dataset.from_tensor_slices(
    ({"document_indices": documents,
      "author_indices": shuffled_author_indices}, shuffled_counts))
  dataset = dataset.shuffle(1000, reshuffle_each_iteration=True).batch(
    batch_size)
  vocabulary = np.loadtxt(os.path.join(data_dir, "vocabulary.txt"),
                          dtype=str,
                          delimiter="\n",
                          comments="<!-")
  return dataset, vocabulary, author_map, num_documents, num_words


class VariationalFamily(tf.keras.layers.Layer):
  """Object to represent variational parameters."""

  def __init__(self, 
               family, 
               shape, 
               fitted_shape=None, 
               fitted_rate=None, 
               cavi=False):
    """Initialize variational family.

    Args:
      family: A string repesenting the variational family, one of "gamma", 
        "lognormal", or "normal".
      shape: A list denoting the shape of the variational parameters.
      fitted_shape: The fitted shape parameter from Poisson Factorization, used
        only if pre-initializing with Poisson Factoriation.
      fitted_rate: The fitted rate parameter from Poisson Factorization.
      cavi: Whether the variational parameters will be maximized with CAVI 
        rather than with gradient descent.
    """
    super(VariationalFamily, self).__init__()
    if family in ['normal', 'lognormal']:
      if fitted_shape is None or fitted_rate is None:
        self.location = tf.Variable(
          tf.keras.initializers.GlorotUniform()(shape=shape))
      else:
        self.location = tf.Variable(np.log(fitted_shape / fitted_rate))

      self.scale = tfp.util.TransformedVariable(
        tf.ones(shape),
        bijector=tfp.bijectors.Softplus())
    elif family == 'gamma':
      if fitted_shape is not None:
        if cavi:
          # If we're doing CAVI, the shape doesn't need to be a transformed
          # variable because it's optimized directly rather than with gradient
          # descent.
          self.shape = tf.Variable(fitted_shape)
        else:
          self.shape = tfp.util.TransformedVariable(
            fitted_shape,
            bijector=tfp.bijectors.Softplus())
      else:
        if cavi:
          self.shape = tf.Variable(
            tf.exp(0.5 * tf.keras.initializers.RandomNormal()(shape=shape)))
        else:
          self.shape = tfp.util.TransformedVariable(
            tf.ones(shape),
            bijector=tfp.bijectors.Softplus())

      if fitted_rate is not None:
        if cavi:
          self.rate = tf.Variable(fitted_rate)
        else:
          self.rate = tfp.util.TransformedVariable(
            fitted_rate,
            bijector=tfp.bijectors.Softplus())
      else:
        if cavi:
          self.rate = tf.Variable(
            tf.exp(0.5 * tf.keras.initializers.RandomNormal()(shape=shape)))
        else:
          self.rate = tfp.util.TransformedVariable(
            tf.ones(shape),
            bijector=tfp.bijectors.Softplus())

    if family == 'normal':
      self.distribution = tfp.distributions.Normal(loc=self.location,
                                                   scale=self.scale)
      self.prior = tfp.distributions.Normal(loc=0., scale=1.)
    elif family == 'lognormal':
      self.distribution = tfp.distributions.LogNormal(loc=self.location,
                                                      scale=self.scale)
      self.prior = tfp.distributions.Gamma(concentration=0.3, rate=0.3)
    elif family == 'gamma':
      self.distribution = tfp.distributions.Gamma(concentration=self.shape,
                                                  rate=self.rate)
      self.prior = tfp.distributions.Gamma(concentration=0.3, rate=0.3)
    else:
      raise ValueError("Unrecognized variational family.")
    # NOTE: tf.keras requires formally recognizing TFP variables in order to
    # optimize them. See: https://github.com/tensorflow/probability/issues/946
    self.recognized_variables = self.distribution.variables

  def get_log_prior(self, samples):
    """Compute log prior of samples."""
    # Sum all but first axis.
    log_prior = tf.reduce_sum(self.prior.log_prob(samples),
                              axis=tuple(range(1, len(samples.shape))))
    return log_prior

  def get_entropy(self, samples):
    """Compute entropy of samples from variational distribution."""
    # Sum all but first axis.
    entropy = -tf.reduce_sum(self.distribution.log_prob(samples),
                             axis=tuple(range(1, len(samples.shape))))
    return entropy

  def sample(self, num_samples, seed=None):
    """Sample from variational family using reparameterization."""
    seed, sample_seed = tfp.random.split_seed(seed)
    return self.distribution.sample(num_samples, seed=sample_seed), seed


class TBIP(tf.keras.Model):
  """Tensorflow implementation of the TBIP model."""

  def __init__(self,
               positive_variational_family,
               num_documents,
               num_topics,
               num_words,
               num_authors,
               num_samples,
               cavi=False,
               fitted_document_shape=None,
               fitted_document_rate=None,
               fitted_objective_topic_shape=None,
               fitted_objective_topic_rate=None,):
    """Initialize TBIP model.

    Args:
      positive_variational_family: A string denoting the variational family of
        the positive latent variables (the document intensities `theta` and the
        objective topic `beta`). Either "gamma" or "lognormal".
      num_documents: The number of documents in the corpus.
      num_topics: The number of topics used for the model.
      num_words: The number of words in the vocabulary.
      num_authors: The number of authors in the corpus.
      num_samples: The number of Monte-Carlo samples to use to approximate the
        ELBO.
      cavi: Whether to perform CAVI updates for the positive variational 
        variables (can only be used if a Gamma variational family is used).
      fitted_document_shape: The fitted document shape parameter from Poisson 
        Factorization, used only if pre-initializing with Poisson Factoriation.
      fitted_document_rate: The fitted document rate parameter from Poisson 
        Factorization.
      fitted_objective_topic_shape: The fitted objective topic shape parameter 
        from Poisson Factorization.
      fitted_objective_topic_rate: The fitted objective topic rate parameter 
        from Poisson Factorization.
    """
    super(TBIP, self).__init__()
    self.positive_variational_family = positive_variational_family
    self.num_documents = num_documents
    self.document_distribution = VariationalFamily(
      positive_variational_family,
      [num_documents, num_topics],
      fitted_shape=fitted_document_shape,
      fitted_rate=fitted_document_rate,
      cavi=cavi,)
    self.objective_topic_distribution = VariationalFamily(
      positive_variational_family,
      [num_topics, num_words],
      fitted_shape=fitted_objective_topic_shape,
      fitted_rate=fitted_objective_topic_rate,
      cavi=cavi,)
    self.ideological_topic_distribution = VariationalFamily(
      'normal',
      [num_topics, num_words],)
    self.ideal_point_distribution = VariationalFamily(
      'normal',
      [num_authors],)
    self.author_verbosity_distribution = VariationalFamily(
      'normal', 
      shape=[num_authors],)
    self.num_samples = num_samples
    self.cavi = cavi

  def get_log_prior(self,
                    document_samples,
                    objective_topic_samples,
                    ideological_topic_samples,
                    ideal_point_samples,
                    author_verbosity_samples,):
    """Compute log prior of samples.

    Args:
      document_samples: Samples from the document intensity variational 
        distribution. A tensor with shape [num_samples, num_documents,
        num_topics].
      objective_topic_samples: Samples from the objective topic variational 
        distribution. A tensor with shape [num_samples, num_topics, num_words].
      ideological_topic_samples: Samples from the ideological topic variational
        distribution. A tensor with shape [num_samples, num_topics, num_words].
      ideal_point_samples: Samples from the ideal point variational 
        distribution. A tensor with shape [num_samples, num_authors].
      author_verbosity_samples: Samples from the author verbosity variational
        distribution. A tensor with shape [num_samples, num_authors].

    Returns:
      log_prior: Monte-Carlo estimate of the log prior. A tensor with shape
        [num_samples].
    """
    document_log_prior = self.document_distribution.get_log_prior(
      document_samples)
    objective_topic_log_prior = (
      self.objective_topic_distribution.get_log_prior(objective_topic_samples))
    ideological_topic_log_prior = (
      self.ideological_topic_distribution.get_log_prior(
        ideological_topic_samples))
    ideal_point_log_prior = self.ideal_point_distribution.get_log_prior(
      ideal_point_samples)
    author_verbosity_log_prior = (
      self.author_verbosity_distribution.get_log_prior(
        author_verbosity_samples))
    log_prior = (document_log_prior +
                 objective_topic_log_prior +
                 ideological_topic_log_prior +
                 ideal_point_log_prior + 
                 author_verbosity_log_prior)
    return log_prior

  def get_entropy(self,
                  document_samples,
                  objective_topic_samples,
                  ideological_topic_samples,
                  ideal_point_samples,
                  author_verbosity_samples,):
    """Compute entropy of samples from variational family.

    Args:
      document_samples: Samples from the document intensity variational
        distribution. A tensor with shape [num_samples, num_documents,
        num_topics].
      objective_topic_samples: Samples from the objective topic variational
        distribution. A tensor with shape [num_samples, num_topics, num_words].
      ideological_topic_samples: Samples from the ideological topic variational
        distribution. A tensor with shape [num_samples, num_topics, num_words].
      ideal_point_samples: Samples from the ideal point variational
        distribution. A tensor with shape [num_samples, num_authors].
      author_verbosity_samples: Samples from the author verbosity variational
        distribution. A tensor with shape [num_samples, num_authors].

    Returns:
      entropy: Monte-Carlo estimate of the entropy. A tensor with shape
        [num_samples].
    """
    document_entropy = self.document_distribution.get_entropy(
      document_samples)
    objective_topic_entropy = (
      self.objective_topic_distribution.get_entropy(objective_topic_samples))
    ideological_topic_entropy = (
      self.ideological_topic_distribution.get_entropy(
        ideological_topic_samples))
    ideal_point_entropy = self.ideal_point_distribution.get_entropy(
      ideal_point_samples)
    author_verbosity_entropy = self.author_verbosity_distribution.get_entropy(
      author_verbosity_samples)
    entropy = (document_entropy +
               objective_topic_entropy +
               ideological_topic_entropy +
               ideal_point_entropy +
               author_verbosity_entropy)
    return entropy

  def get_samples(self, seed=None):
    """Get samples from variational families."""
    document_samples, seed = self.document_distribution.sample(
      self.num_samples, seed=seed)
    objective_topic_samples, seed = self.objective_topic_distribution.sample(
      self.num_samples, seed=seed)
    ideological_topic_samples, seed = (
      self.ideological_topic_distribution.sample(self.num_samples, seed=seed))
    ideal_point_samples, seed = self.ideal_point_distribution.sample(
        self.num_samples, seed=seed)
    author_verbosity_samples, seed = self.author_verbosity_distribution.sample(
      self.num_samples, seed=seed)
    samples = [document_samples, objective_topic_samples,
               ideological_topic_samples, ideal_point_samples,
               author_verbosity_samples]
    return samples, seed

  def get_rate_log_prior_entropy(self,
                                 document_indices,
                                 author_indices,
                                 seed=None):
    """Compute Monte-Carlo estimates of ELBO terms.
    
    Args:
      document_indices: Indices of documents in the batch. A tensor with shape
        [batch_size].
      author_indices: Indices of authors in the batch. A tensor with shape
        [batch_size].
      seed: Random seed.
    
    Returns:
      rate: Monte-Carlo estimate of the rate. A tensor with shape [num_samples, 
        batch_size, num_words].
      log_prior: Monte-Carlo estimate of the log prior. A tensor with shape
        [num_samples].
      entropy: Monte-Carlo estimate of the entropy. A tensor with shape
        [num_samples].
      seed: Updated random seed.
    """
    ((document_samples, objective_topic_samples,
     ideological_topic_samples, ideal_point_samples,
     author_verbosity_samples),
     seed) = self.get_samples(seed)
    log_prior = self.get_log_prior(document_samples,
                                   objective_topic_samples,
                                   ideological_topic_samples,
                                   ideal_point_samples,
                                   author_verbosity_samples,)
    entropy = self.get_entropy(document_samples,
                               objective_topic_samples,
                               ideological_topic_samples,
                               ideal_point_samples,
                               author_verbosity_samples,)
    # Compute rate for each document in batch.
    selected_document_samples = tf.gather(document_samples,
                                          document_indices,
                                          axis=1)
    selected_ideal_points = tf.gather(ideal_point_samples,
                                      author_indices,
                                      axis=1)
    selected_author_verbosities = tf.gather(author_verbosity_samples,
                                            author_indices,
                                            axis=1)
    # Compute ideological term, adding the per-author verbosity scaling.
    selected_ideological_topic_samples = tf.exp(
      selected_ideal_points[:, :, tf.newaxis, tf.newaxis] *
      ideological_topic_samples[:, tf.newaxis, :, :] + (
        selected_author_verbosities[:, :, tf.newaxis, tf.newaxis]))
    rate = tf.reduce_sum(
      selected_document_samples[:, :, :, tf.newaxis] *
      objective_topic_samples[:, tf.newaxis, :, :] *
      selected_ideological_topic_samples[:, :, :, :],
      axis=2)
    return rate, log_prior, entropy, seed
  
  def get_ideological_term_geometric_mean(self, author_indices):
    """Compute variational geometric mean of ideological term for CAVI.

    More specifically, we compute:

      exp(E[log(exp(eta_kv * x_{a_d} + w_{a_d}))]) =
      exp(E[eta_kv] * E[x_{a_d}] + E[w_{a_d}]),

    where a_d is the author of document d and w_{a_d} is the author verbosity.
    The geometric mean is used directly for the computation of the auxiliary
    terms. The CAVI updates for theta and beta require the actual mean of the
    ideological term, which cannot be computed in closed form, so we 
    approximate it with the geometric mean.

    Args:
      author_indices: Indices of authors in the batch. A tensor with shape
        [batch_size].
    
    Returns:
      geometric_mean: Geometric mean of the ideological term. A tensor with 
        shape [batch_size, num_topics, num_words].
    """
    ideal_point_loc = tf.gather(self.ideal_point_distribution.location, 
                                author_indices, 
                                axis=0)
    author_verbosity_loc = tf.gather(
      self.author_verbosity_distribution.location, 
      author_indices, 
      axis=0)
    expected_ideological_term = tf.exp(
      self.ideological_topic_distribution.location[tf.newaxis, :, :] * 
      ideal_point_loc[:, tf.newaxis, tf.newaxis] + 
      author_verbosity_loc[:, tf.newaxis, tf.newaxis])
    return expected_ideological_term
  
  def get_cavi_auxiliary_proportions(self, 
                                     document_indices, 
                                     expected_ideological_term):
    """Perform CAVI update for auxiliary proportion variables.
    
    Args:
      document_indices: Indices of documents in the batch. A tensor with shape
        [batch_size].
      expected_ideological_term: Geometric mean of the ideological term. A 
        tensor with shape [batch_size, num_topics, num_words].
    
    Returns:
      auxiliary_proportions: The updated auxiliary proportions. A tensor with
        shape [batch_size, num_topics, num_words]. The tensor is normalized
        across topics, so it can be interpreted as the proportion of each 
        topic belong to each word.
    """
    document_shape = tf.gather(self.document_distribution.shape,
                               document_indices, 
                               axis=0)
    document_rate = tf.gather(self.document_distribution.rate, 
                              document_indices, 
                              axis=0)
    document_geometric_mean = tf.exp(
      tf.math.digamma(document_shape)) / document_rate
    objective_topic_geometric_mean = tf.exp(
      tf.math.digamma(self.objective_topic_distribution.shape)
    ) / self.objective_topic_distribution.rate
    auxiliary_numerator = (document_geometric_mean[:, :, tf.newaxis] * 
                           objective_topic_geometric_mean[tf.newaxis, :, :] * 
                           expected_ideological_term)
    auxiliary_proportions = auxiliary_numerator / tf.reduce_sum(
      auxiliary_numerator, axis=1)[:, tf.newaxis, :]
    return auxiliary_proportions
  
  def get_cavi_document_parameters(self, 
                                   counts, 
                                   expected_ideological_term, 
                                   auxiliary_proportions):
    """Perform CAVI update for document parameters.

    The optimal update requires the variational expectation of the ideological
    term:

      E[exp(eta_kv * x_{a_d} + w_{a_d})],
    
    which is intractable. Instead, we approximate it with the geometric mean.

    Args:
      counts: Counts of words in documents. A tensor with shape
        [batch_size, num_words].
      expected_ideological_term: Geometric mean of the ideological term. A
        tensor with shape [batch_size, num_topics, num_words].
      auxiliary_proportions: The auxiliary proportions. A tensor with shape 
        [batch_size, num_topics, num_words]. 
    
    Returns:
      document_shape: The updated document shape. A tensor with shape
        [batch_size, num_topics].
      document_rate: The updated document rate. A tensor with shape
        [batch_size, num_topics].
    """
    updated_document_shape = 0.3 + tf.reduce_sum(
      auxiliary_proportions * counts[:, tf.newaxis, :], axis=-1)
    expected_objective_topic = (self.objective_topic_distribution.shape / 
                                self.objective_topic_distribution.rate)
    updated_document_rate = 0.3 + tf.reduce_sum(
      expected_objective_topic[tf.newaxis] * expected_ideological_term, -1)
    return updated_document_shape, updated_document_rate
  
  def get_cavi_objective_topic_parameters(self, 
                                          counts, 
                                          expected_ideological_term, 
                                          auxiliary_proportions, 
                                          document_shape, 
                                          document_rate):
    """Perform CAVI update for objective topic parameters.

    The optimal update requires the variational expectation of the ideological
    term, which is intractable, so we approximate it with the geometric mean.

    Args:
      counts: Counts of words in documents. A tensor with shape
        [batch_size, num_words].
      expected_ideological_term: Geometric mean of the ideological term. A
        tensor with shape [batch_size, num_topics, num_words].
      auxiliary_proportions: The auxiliary proportions. A tensor with shape 
        [batch_size, num_topics, num_words].
      document_shape: The variational document shape. A tensor with shape 
        [batch_size, num_topics].
      document_rate: The variational document rate. A tensor with shape
        [batch_size, num_topics].
    
    Returns:
      objective_topic_shape: The updated objective topic shape. A tensor with
        shape [num_topics, num_words].
      objective_topic_rate: The updated objective topic rate. A tensor with
        shape [num_topics, num_words].
    """
    batch_size = tf.shape(counts)[0]
    # We scale to account for the fact that we're only using a minibatch to
    # update the variational parameters of a global latent variable.
    minibatch_scaling = tf.cast(self.num_documents / batch_size, 
                                tf.dtypes.float32)
    updated_objective_topic_shape = 0.3 + minibatch_scaling * tf.reduce_sum(
      auxiliary_proportions * counts[:, tf.newaxis, :], axis=0)
    expected_document = document_shape / document_rate
    updated_objective_topic_rate = 0.3 + minibatch_scaling * tf.reduce_sum(
      expected_document[..., tf.newaxis] * expected_ideological_term, 0)
    return updated_objective_topic_shape, updated_objective_topic_rate
  
  def perform_cavi_updates(self, inputs, outputs, step):
    """Perform CAVI updates for document intensities and objective topics.
    
    Args:
      inputs: A dictionary of input tensors.
      outputs: A sparse tensor containing word counts.
      step: The current training step.
    """
    counts = tf.sparse.to_dense(outputs)
    # The updates all use the geometric mean of the ideological term.
    expected_ideological_term = self.get_ideological_term_geometric_mean(
      inputs['author_indices'])
    # An auxiliary latent variable is required to perform the CAVI updates for
    # the document intensities and objective topics.
    auxiliary_proportions = self.get_cavi_auxiliary_proportions(
      inputs['document_indices'], 
      expected_ideological_term)
    # Update the document intensities.
    updated_document_shape, updated_document_rate = (
      self.get_cavi_document_parameters(counts, 
                                        expected_ideological_term, 
                                        auxiliary_proportions))
    # Update the objective topics.
    updated_objective_topic_shape, updated_objective_topic_rate = (
      self.get_cavi_objective_topic_parameters(counts,
                                               expected_ideological_term,
                                               auxiliary_proportions,
                                               updated_document_shape,
                                               updated_document_rate))
    # The updates above were only for the documents in the batch, so we update
    # the slice of the global latent variables corresponding to the documents
    # in the batch.
    global_document_shape = tf.tensor_scatter_nd_update(
      self.document_distribution.shape, 
      inputs['document_indices'][:, tf.newaxis], 
      updated_document_shape)
    global_document_rate = tf.tensor_scatter_nd_update(
      self.document_distribution.rate, 
      inputs['document_indices'][:, tf.newaxis], 
      updated_document_rate)
    # Because the objective topics are a global variable, stochastic
    # variational inference calls for updating the variational parameters using
    # a convex combination of the previous parameters and the updates. We set
    # the step size to be a decreasing sequence that satisfies the Robbins-
    # Monro condition.
    step_size = tf.math.pow(tf.cast(step, tf.dtypes.float32) + 1, -0.7)
    global_objective_topic_shape = (
      step_size * updated_objective_topic_shape + 
      (1 - step_size) * self.objective_topic_distribution.shape)
    global_objective_topic_rate = (
      step_size * updated_objective_topic_rate + 
      (1 - step_size) * self.objective_topic_distribution.rate)
    self.document_distribution.shape.assign(global_document_shape)
    self.document_distribution.rate.assign(global_document_rate)
    self.objective_topic_distribution.shape.assign(
      global_objective_topic_shape)
    self.objective_topic_distribution.rate.assign(global_objective_topic_rate)

  def get_topic_means(self):
    """Get neutral and ideological topics from variational parameters.
    
    For each (k,v), we want to evaluate E[beta_kv], E[beta_kv * exp(eta_kv)], 
    and E[beta_kv * exp(-eta_kv)], where the expectations are with respect to 
    the variational distributions. Like the paper, beta refers to the obective 
    topic and eta refers to the ideological topic.

    The exact form depends on the variational family (gamma or lognormal).

    Returns:
      negative_mean: A tensor with shape [num_topics, num_words], denoting the
        variational mean for the ideological topics with an ideal point of -1.
      neutral_mean: A tensor with shape [num_topics, num_words] denoting the
        variational mean for the neutral topics.
      positive_mean: A tensor with shape [num_topics, num_words], denoting the
        variational mean for the ideological topics with an ideal point of +1.
    """

    ideological_topic_loc = self.ideological_topic_distribution.location
    ideological_topic_scale = self.ideological_topic_distribution.scale
    if self.positive_variational_family == 'gamma':
      objective_topic_shape = self.objective_topic_distribution.shape
      objective_topic_rate = self.objective_topic_distribution.rate
      neutral_mean = objective_topic_shape / objective_topic_rate
      positive_mean = ((objective_topic_shape / objective_topic_rate) * np.exp(
                       ideological_topic_loc +
                       (ideological_topic_scale ** 2) / 2))
      negative_mean = ((objective_topic_shape / objective_topic_rate) * np.exp(
                       -ideological_topic_loc +
                       (ideological_topic_scale ** 2) / 2))
    elif self.positive_variational_family == 'lognormal':
      objective_topic_loc = self.objective_topic_distribution.location
      objective_topic_scale = self.objective_topic_distribution.scale
      neutral_mean = objective_topic_loc + objective_topic_scale ** 2 / 2
      positive_mean = (objective_topic_loc +
                       ideological_topic_loc +
                       (objective_topic_scale ** 2 +
                        ideological_topic_scale ** 2) / 2)
      negative_mean = (objective_topic_loc -
                       ideological_topic_loc +
                       (objective_topic_scale ** 2 +
                        ideological_topic_scale ** 2) / 2)
    return negative_mean, neutral_mean, positive_mean

  def call(self, inputs, seed):
    """Approximate terms in the ELBO with Monte-Carlo samples.
    
    Args:
      inputs: A dictionary of input tensors.
      seed: A seed for the random number generator.
    
    Returns:
      rate: A tensor with shape [num_samples, batch_size, num_words],
        corresponding to the sampled word count rates.
      negative_log_prior: A scalar tensor, denoting the negative log prior.
      negative_entropy: A scalar tensor, denoting the negative entropy.
      seed: The updated seed.
    """
    document_indices = inputs['document_indices']
    author_indices = inputs['author_indices']
    rate, log_prior, entropy, seed = self.get_rate_log_prior_entropy(
      document_indices, author_indices, seed)
    negative_log_prior = -tf.reduce_mean(log_prior)
    negative_entropy = -tf.reduce_mean(entropy)
    return rate, negative_log_prior, negative_entropy, seed


def print_topics(neutral_mean, negative_mean, positive_mean, vocabulary):
  """Get neutral and ideological topics to be used for Tensorboard.

  Args:
    neutral_mean: The mean of the neutral topics, a tensor with shape
      [num_topics, num_words].
    negative_mean: The mean of the negative topics, a tensor with shape
      [num_topics, num_words].
    positive_mean: The mean of the positive topics, a tensor with shape
      [num_topics, num_words].
    vocabulary: A list of the vocabulary with shape [num_words].

  Returns:
    topic_strings: A list of the negative, neutral, and positive topics.
  """
  num_topics, _ = neutral_mean.shape
  words_per_topic = 10
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
    topic_strings.append("  \n".join(
        [negative_string, neutral_string, positive_string]))
  return np.array(topic_strings)


def print_ideal_points(ideal_point_loc, author_map):
  """Print ideal point ordering for Tensorboard."""
  return ", ".join(author_map[np.argsort(ideal_point_loc)])


def log_static_features(model, vocabulary, author_map, step):
  """Log static features to Tensorboard."""
  negative_mean, neutral_mean, positive_mean = model.get_topic_means()
  ideal_point_list = print_ideal_points(
    model.ideal_point_distribution.location.numpy(),
    author_map)
  topics = print_topics(neutral_mean,
                        negative_mean,
                        positive_mean,
                        vocabulary)
  tf.summary.text("ideal_points", ideal_point_list, step=step)
  tf.summary.text("topics", topics, step=step)

  # The parameters to log depend on the variational families.
  if model.positive_variational_family == 'gamma':
    tf.summary.histogram("params/document_shape",
                         model.document_distribution.shape,
                         step=step)
    tf.summary.histogram("params/document_rate",
                         model.document_distribution.rate,
                         step=step)
    tf.summary.histogram("params/objective_topic_shape",
                         model.objective_topic_distribution.shape,
                         step=step)
    tf.summary.histogram("params/objective_topic_rate",
                         model.objective_topic_distribution.rate,
                         step=step)
  else:
    tf.summary.histogram("params/document_loc",
                         model.document_distribution.location,
                         step=step)
    tf.summary.histogram("params/document_scale",
                         model.document_distribution.scale,
                         step=step)
    tf.summary.histogram("params/objective_topic_loc",
                         model.objective_topic_distribution.location,
                         step=step)
    tf.summary.histogram("params/objective_topic_scale",
                         model.objective_topic_distribution.scale,
                         step=step)
  tf.summary.histogram("params/ideological_topic_loc",
                       model.ideological_topic_distribution.location,
                       step=step)
  tf.summary.histogram("params/ideological_topic_scale",
                       model.ideological_topic_distribution.scale,
                       step=step)
  tf.summary.histogram("params/ideal_point_loc",
                       model.ideal_point_distribution.location,
                       step=step)
  tf.summary.histogram("params/ideal_point_scale",
                       model.ideal_point_distribution.scale,
                       step=step)
  tf.summary.histogram("params/author_verbosity_loc", 
                       model.author_verbosity_distribution.location, 
                       step=step)
  tf.summary.histogram("params/author_verbosity_scale", 
                       model.author_verbosity_distribution.scale, 
                       step=step)


@tf.function
def train_step(model, inputs, outputs, optim, seed, step=None):
  """Perform a single training step.

  Args:
    model: The TBIP model.
    inputs: A dictionary of input tensors.
    outputs: A sparse tensor containing word counts.
    optim: An optimizer.
    seed: The random seed.
    step: The current step.
  
  Returns:
    total_loss: The total loss for the minibatch (the negative ELBO, sampled 
      with Monte-Carlo).
    reconstruction_loss: The reconstruction loss (negative log-likelihood), 
      sampled for the minibatch.
    log_prior_loss: The negative log prior.
    entropy_loss: The negative entropy.
  """
  if model.cavi:
    # Perform CAVI updates.
    model.perform_cavi_updates(inputs, outputs, step)
  with tf.GradientTape() as tape:
    predictions, log_prior_loss, entropy_loss, seed = model(inputs, seed)
    count_distribution = tfp.distributions.Poisson(rate=predictions)
    count_log_likelihood = count_distribution.log_prob(
      tf.sparse.to_dense(outputs))
    count_log_likelihood = tf.reduce_sum(count_log_likelihood, axis=[1, 2])
    # Adjust for the fact that we're only using a minibatch.
    batch_size = tf.shape(outputs)[0]
    count_log_likelihood = count_log_likelihood * tf.dtypes.cast(
      model.num_documents / batch_size, tf.float32)
    reconstruction_loss = -count_log_likelihood
    total_loss = tf.reduce_mean(reconstruction_loss +
                                log_prior_loss +
                                entropy_loss)
  trainable_variables = tape.watched_variables()
  if model.cavi:
    # If we're doing CAVI, the first four parameters are the document and
    # objective topic parameters, which should not be updated with gradients
    # because they're updated with CAVI.
    trainable_variables = trainable_variables[4:]
  grads = tape.gradient(total_loss, trainable_variables)
  optim.apply_gradients(zip(grads, trainable_variables))
  return total_loss, reconstruction_loss, log_prior_loss, entropy_loss, seed


def main(argv):
  del argv
  # Initial random seed for parameter initialization.
  tf.random.set_seed(FLAGS.seed)
  random_state = np.random.RandomState(FLAGS.seed)
  project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.pardir))
  source_dir = os.path.join(project_dir, "data/{}".format(FLAGS.data))

  # As described in the docstring, the data directory must have the following
  # files: counts.npz, author_indices.npy, vocabulary.txt, author_map.txt.
  data_dir = os.path.join(source_dir, "clean")
  save_dir = os.path.join(source_dir, "fits/{}".format(FLAGS.checkpoint_name))

  if FLAGS.cavi:
    if FLAGS.positive_variational_family != 'gamma':
      raise ValueError("Can only do CAVI for gamma variational families.")

  (dataset, vocabulary, author_map,
   num_documents, num_words) = build_input_pipeline(
      data_dir,
      FLAGS.batch_size,
      random_state,
      FLAGS.counts_transformation)
  num_authors = len(author_map)

  fit_dir = os.path.join(source_dir, "pf-fits")
  if FLAGS.pre_initialize_parameters:
    fitted_document_shape = np.load(
        os.path.join(fit_dir, "document_shape.npy")).astype(np.float32)
    fitted_document_rate = np.load(
        os.path.join(fit_dir, "document_rate.npy")).astype(np.float32)
    fitted_topic_shape = np.load(
        os.path.join(fit_dir, "topic_shape.npy")).astype(np.float32)
    fitted_topic_rate = np.load(
        os.path.join(fit_dir, "topic_rate.npy")).astype(np.float32)
  else:
    fitted_document_shape = None
    fitted_document_rate = None
    fitted_topic_shape = None
    fitted_topic_rate = None

  optim = tf.optimizers.Adam(learning_rate=FLAGS.learning_rate)

  model = TBIP(FLAGS.positive_variational_family,
               num_documents,
               FLAGS.num_topics,
               num_words,
               num_authors,
               FLAGS.num_samples,
               FLAGS.cavi,
               fitted_document_shape,
               fitted_document_rate,
               fitted_topic_shape,
               fitted_topic_rate,)

  # Add start epoch so checkpoint state is saved.
  model.start_epoch = tf.Variable(-1)

  checkpoint_dir = os.path.join(save_dir, "checkpoints")
  if os.path.exists(checkpoint_dir) and FLAGS.load_checkpoint:
    pass
  else:
    # If we're not loading a checkpoint, overwrite the existing directory
    # with saved results.
    if os.path.exists(save_dir):
      print("Deleting old log directory at {}".format(save_dir))
      tf.io.gfile.rmtree(save_dir)

  # We keep track of the seed to make sure the random number state is the same
  # whether or not we load a model.
  _, seed = tfp.random.split_seed(FLAGS.seed)
  checkpoint = tf.train.Checkpoint(optimizer=optim,
                                   net=model,
                                   seed=tf.Variable(seed))
  manager = tf.train.CheckpointManager(checkpoint,
                                       checkpoint_dir,
                                       max_to_keep=1)

  checkpoint.restore(manager.latest_checkpoint)

  if manager.latest_checkpoint:
    # Load from saved checkpoint, keeping track of the seed.
    seed = checkpoint.seed
    # Since the dataset shuffles at every epoch and we'd like the runs to be
    # identical whether or not we load a checkpoint, we need to make sure the
    # dataset state is consistent. This is a hack but it will do for now.
    # Here's the issue: https://github.com/tensorflow/tensorflow/issues/48178
    for _ in range(model.start_epoch.numpy() + 1):
      _ = iter(dataset)
    print("Restored from {}".format(manager.latest_checkpoint))
  else:
    print("Initializing from scratch.")

  summary_writer = tf.summary.create_file_writer(save_dir)
  summary_writer.set_as_default()
  start_time = time.time()
  start_epoch = model.start_epoch.numpy()
  for epoch in range(start_epoch + 1, FLAGS.num_epochs):
    for batch_index, batch in enumerate(iter(dataset)):
      batches_per_epoch = len(dataset)
      step = batches_per_epoch * epoch + batch_index
      inputs, outputs = batch
      (total_loss, reconstruction_loss,
       log_prior_loss, entropy_loss, seed) = train_step(
          model, inputs, outputs, optim, seed, tf.constant(step))
      checkpoint.seed.assign(seed)

    sec_per_epoch = (time.time() - start_time) / (epoch - start_epoch)
    sec_per_step = (time.time() - start_time) / step
    print("Epoch: {:>3d} ELBO: {:.3f} Entropy: {:.1f} ({:.3f} sec/step, "
          "{:.3f} sec/epoch)".format(epoch, 
                                     -total_loss.numpy(), 
                                     -entropy_loss.numpy(), 
                                     sec_per_step,
                                     sec_per_epoch))

    # Log to tensorboard at the end of every `save_every` epochs.
    if epoch % FLAGS.save_every == 0:
      tf.summary.scalar('loss', total_loss, step=step)
      tf.summary.scalar("elbo/entropy", -entropy_loss, step=step)
      tf.summary.scalar("elbo/log_prior", -log_prior_loss, step=step)
      tf.summary.scalar("elbo/count_log_likelihood",
                        -tf.reduce_mean(reconstruction_loss),
                        step=step)
      tf.summary.scalar('elbo/elbo', -total_loss, step=step)
      log_static_features(model, vocabulary, author_map, step)
      summary_writer.flush()

      # Save checkpoint too.
      model.start_epoch.assign(epoch)
      save_path = manager.save()
      print("Saved checkpoint for epoch {}: {}".format(epoch, save_path))

      # All model parameters can be accessed by loading the checkpoint, similar
      # to the logic at the beginning of this function. Since that may be 
      # too much hassle, we also save the ideal point model parameters to a 
      # separate file. You can save additional model parameters if you'd like.
      param_save_dir = os.path.join(save_dir, "params/")
      if not os.path.exists(param_save_dir):
        os.makedirs(param_save_dir)
      np.save(os.path.join(param_save_dir, "ideal_point_loc"), 
              model.ideal_point_distribution.location.numpy())
      np.save(os.path.join(param_save_dir, "ideal_point_scale"), 
              model.ideal_point_distribution.scale.numpy())


if __name__ == "__main__":
  app.run(main)
