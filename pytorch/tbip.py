"""PyTorch implementation of the text-based ideal point model (TBIP).

Let y_{dv} denote the counts of word v in document d. Let x_d refer to the 
ideal point of the author of document d. Then we model:

theta, beta ~ Gamma(alpha, alpha)
x, eta ~ N(0, 1)
y_{dv} ~ Pois(sum_k theta_dk beta_kv exp(x_d * eta_kv).

We perform variational inference to provide estimates for the posterior 
distribution of each latent variable. We take reparameterization gradients,
using a lognormal variational family for the positive variables (theta, beta)
and a normal variational family for the real variables (x, eta).

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

We provide more details in our paper [1].

#### References
[1]: Keyon Vafa, Suresh Naidu, David Blei. Text-Based Ideal Points. In 
     _Conference of the Association for Computational Linguistics_, 2020.
"""
import os
import shutil
import time
import torch

from absl import app
from absl import flags
import numpy as np
import scipy.sparse as sparse
from torch.utils.tensorboard import SummaryWriter


flags.DEFINE_float("learning_rate",
                   default=0.01,
                   help="Adam learning rate.")
flags.DEFINE_integer("max_steps",
                     default=1000000,
                     help="Number of training steps to run.")
flags.DEFINE_integer("num_topics",
                     default=50,
                     help="Number of topics.")
flags.DEFINE_integer("batch_size",
                     default=1024,
                     help="Batch size.")
flags.DEFINE_integer("num_samples",
                     default=1,
                     help="Number of samples to use for ELBO approximation.")
flags.DEFINE_enum("counts_transformation",
                  default="nothing",
                  enum_values=["nothing", "log"],
                  help="Transformation used on counts data.")
flags.DEFINE_boolean("pre_initialize_parameters",
                     default=True,
                     help="Whether to use pre-initialized document and topic "
                          "intensities (with Poisson factorization).")
flags.DEFINE_string("data",
                    default="senate-speeches-114",
                    help="Data source being used.")
flags.DEFINE_integer("senate_session",
                     default=113,
                     help="Senate session (used only when data is "
                          "'senate-speech-comparisons'.")
flags.DEFINE_integer("print_steps",
                     default=500,
                     help="Number of steps to print and save results.")
FLAGS = flags.FLAGS


class TBIPDataset(torch.utils.data.Dataset):
  """Dataset object to load corpus and feed to iterator."""
  
  def __init__(self, data_dir, counts_transformation="nothing"):
    """Load data.
    
    Args:
      data_dir: The directory where the data is located. There must be four
        files inside the rep: `counts.npz`, `author_indices.npy`, 
        `author_map.txt`, and `vocabulary.txt`.
      counts_transformation: A string indicating how to transform the counts.
        One of "nothing" or "log".
    """
    counts = sparse.load_npz(os.path.join(data_dir, "counts.npz"))
    num_documents, num_words = counts.shape
    self.num_documents = num_documents
    self.num_words = num_words
    author_indices = np.load(os.path.join(data_dir, "author_indices.npy"))
    self.num_authors = np.max(author_indices + 1)
    self.author_map = np.loadtxt(os.path.join(data_dir, "author_map.txt"),
                                 dtype=str,
                                 delimiter="\n")
    self.vocabulary = np.loadtxt(os.path.join(data_dir, "vocabulary.txt"),
                                 dtype=str,
                                 delimiter="\n",
                                 comments="<!-")
    self.documents = torch.LongTensor(np.random.permutation(num_documents))
    self.author_indices = torch.LongTensor(author_indices[self.documents])
    shuffled_counts = counts[self.documents]
    
    if counts_transformation == "nothing":
      sparse_values = shuffled_counts.toarray()
    elif counts_transformation == "log":
      sparse_values = np.round(np.log(1 + shuffled_counts.toarray()))
    # NOTE: PyTorch doesn't support sparse tensor batching on GPU, so these
    # lines are commented out for now.
    # self.counts = torch.sparse.FloatTensor(
    #     torch.LongTensor(shuffled_counts.nonzero()), 
    #     torch.FloatTensor(sparse_values), 
    #     torch.Size([num_documents, num_words]))
    self.counts = torch.FloatTensor(sparse_values)
    
    total_counts_per_author = np.bincount(
        author_indices,
        weights=np.array(np.sum(counts, axis=1)).flatten())
    counts_per_document_per_author = (
        total_counts_per_author / np.bincount(author_indices))
    # Author weights is how lengthy each author's opinion over average is.
    self.author_weights = (counts_per_document_per_author /
                           np.mean(np.sum(counts, axis=1)))
  
  def __len__(self):
    return self.num_documents
  
  def __getitem__(self, idx):
    return self.documents[idx], self.author_indices[idx], self.counts[idx]


class VariationalFamily(torch.nn.Module):
  """Object to store variational parameters and get sample statistics."""
  
  def __init__(self, device, family, shape, initial_loc=None):
    """Initialize variational family.
    
    Args:
      device: Device where operations take place.
      family: A string representing the variational family, either "normal" or
        "lognormal".
      shape: A list representing the shape of the variational family.
      initial_loc: An optional tensor with shape `shape`, denoting the initial
        location of the variational family.
    """
    super(VariationalFamily, self).__init__()
    if initial_loc is None:
      if len(shape) > 1:
        self.location = torch.nn.init.xavier_uniform_(
            torch.nn.Parameter(torch.ones(shape)))
      else:
        self.location = torch.nn.init.normal_(
            torch.nn.Parameter(torch.ones(shape)), 
            std=0.1)
    else:
      self.location = torch.nn.Parameter(
          torch.FloatTensor(np.log(initial_loc)))
    self.log_scale = torch.nn.Parameter(torch.zeros(shape))
    self.family = family
    if self.family == 'normal':
      self.prior = torch.distributions.Normal(loc=0., scale=1.)
    elif self.family == 'lognormal':
      self.prior = torch.distributions.Gamma(concentration=0.3, rate=0.3)
    else:
      raise ValueError("Unrecognized prior distribution.")
    self.device = device
  
  def scale(self):
    """Constrain scale to be positive using softplus."""
    return torch.nn.functional.softplus(self.log_scale)
  
  def distribution(self):
    """Create variational distribution."""
    if self.family == 'normal':
      distribution = torch.distributions.Normal(
          loc=self.location, 
          scale=self.scale())
    elif self.family == 'lognormal':
      distribution = torch.distributions.LogNormal(
          loc=self.location, 
          scale=self.scale())
    return distribution
  
  def get_log_prior(self, samples):
    """Compute log prior of samples."""
    # Sum all but first axis.
    log_prior = torch.sum(self.prior.log_prob(samples).to(self.device),
                          axis=tuple(range(1, len(samples.shape))))
    return log_prior
  
  def get_entropy(self, samples):
    """Compute entropy of samples from variational distribution."""
    # Sum all but first axis.
    entropy = -torch.sum(self.distribution().log_prob(samples).to(self.device),
                         axis=tuple(range(1, len(samples.shape))))
    return entropy
  
  def sample(self, num_samples):
    """Sample from variational family using reparameterization."""
    return self.distribution().rsample([num_samples])

  
class TBIP(torch.nn.Module):
  """Object to hold model parameters and approximate ELBO."""
  
  def __init__(self, 
               device,
               author_weights, 
               initial_document_loc,
               initial_objective_topic_loc,
               num_samples,
               print_steps,
               summary_writer):
    """Initialize object.
    
    Args:
      device: Device where computations are performed.
      author_weights: A vector with shape [num_authors], constituting how
        lengthy the opinion is above average.
      initial_document_loc: A [num_documents, num_topics] NumPy array 
        containing the initial document intensity means.
      initial_objective_topic_loc: A [num_topics, num_words] NumPy array
        containing the initial objective topic means. 
      num_samples: Number of Monte-Carlo samples.
      print_steps: How often to print summaries to Tensorboard.
      summary_writer: Writer to log entries to Tensorboard.
    """
    super(TBIP, self).__init__()
    self.author_weights = torch.FloatTensor(author_weights).to(device)
    num_documents, num_topics = initial_document_loc.shape
    _, num_words = initial_objective_topic_loc.shape
    num_authors = len(author_weights)
    self.num_documents = num_documents
    self.document_intensities = VariationalFamily(
        device,
        'lognormal', 
        [num_documents, num_topics],
        initial_loc=initial_document_loc)
    self.objective_topics = VariationalFamily(
        device,
        'lognormal',
        [num_topics, num_words],
        initial_loc=initial_objective_topic_loc)
    self.ideological_topics = VariationalFamily(
        device,
        'normal',
        [num_topics, num_words])
    self.ideal_points = VariationalFamily(device, 'normal', [num_authors])
    self.num_samples = num_samples
    self.print_steps = print_steps
    self.summary_writer = summary_writer
  
  def get_samples(self):
    """Return samples from variational distributions."""
    document_samples = self.document_intensities.sample(self.num_samples)
    objective_topic_samples = self.objective_topics.sample(self.num_samples)
    ideological_topic_samples = self.ideological_topics.sample(
        self.num_samples)
    ideal_point_samples = self.ideal_points.sample(self.num_samples)
    samples = [document_samples, objective_topic_samples,
               ideological_topic_samples, ideal_point_samples]
    return samples
  
  def get_log_prior(self, samples):
    """Calculate log prior of variational samples.
    
    Args:
      samples: A list of length 4 containing the document intensity samples,
        the objective topic samples, the ideological samples, and the ideal
        point samples, in that order.
    
    Returns:
      log_prior: A Monte-Carlo approximation of the log prior, summed across 
        latent dimensions and averaged over the number of samples.
    """
    (document_samples, objective_topic_samples,
     ideological_topic_samples, ideal_point_samples) = samples
    document_log_prior = self.document_intensities.get_log_prior(
        document_samples)
    objective_topic_log_prior = self.objective_topics.get_log_prior(
        objective_topic_samples)
    ideological_topic_log_prior = self.ideological_topics.get_log_prior(
        ideological_topic_samples)
    ideal_point_log_prior = self.ideal_points.get_log_prior(
        ideal_point_samples)
    log_prior = (document_log_prior +
                 objective_topic_log_prior +
                 ideological_topic_log_prior +
                 ideal_point_log_prior)
    return torch.mean(log_prior)
  
  def get_entropy(self, samples):
    """Calculate entropy of variational samples.
    
    Args:
      samples: A list of length 4 containing the document intensity samples,
        the objective topic samples, the ideological samples, and the ideal
        point samples, in that order.
    
    Returns:
      log_prior: A Monte-Carlo approximation of the variational entropy, 
        summed across latent dimensions and averaged over the number of 
        samples.
    """
    (document_samples, objective_topic_samples,
     ideological_topic_samples, ideal_point_samples) = samples
    document_entropy = self.document_intensities.get_entropy(
        document_samples)
    objective_topic_entropy = self.objective_topics.get_entropy(
        objective_topic_samples)
    ideological_topic_entropy = self.ideological_topics.get_entropy(
        ideological_topic_samples)
    ideal_point_entropy = self.ideal_points.get_entropy(ideal_point_samples)
    entropy = (document_entropy +
               objective_topic_entropy +
               ideological_topic_entropy +
               ideal_point_entropy)
    return torch.mean(entropy)
  
  def get_count_log_likelihood(self,
                               samples, 
                               document_indices, 
                               author_indices,
                               counts):
    """Approximate log-likelihood term of ELBO using Monte Carlo samples.
    
    Args:
      samples: A list of length 4 containing the document intensity samples,
        the objective topic samples, the ideological samples, and the ideal
        point samples, in that order.
      document_indices: An int-vector with shape [batch_size]. 
      author_indices: An int-vector with shape [batch_size].
      counts: A float-tensor with shape [batch_size, num_words].
    
    Returns:
      count_log_likelihood: A Monte-Carlo approximation of the count 
        log-likelihood, summed across latent dimensions and averaged over the 
        number of samples.
    
    """
    (document_samples, objective_topic_samples,
     ideological_topic_samples, ideal_point_samples) = samples
    selected_document_samples = document_samples[:, document_indices]
    selected_ideal_points = ideal_point_samples[:, author_indices]
    selected_ideological_topic_samples = torch.exp(
        selected_ideal_points[:, :, None, None] *
        ideological_topic_samples[:, None, :, :])
    selected_author_weights = self.author_weights[author_indices]
    selected_ideological_topic_samples = (
        selected_author_weights[None, :, None, None] *
        selected_ideological_topic_samples)
    rate = torch.sum(
        selected_document_samples[:, :, :, None] *
        objective_topic_samples[:, None, :, :] *
        selected_ideological_topic_samples[:, :, :, :],
        axis=2)
    count_distribution = torch.distributions.Poisson(rate=rate)
    count_log_likelihood = count_distribution.log_prob(counts)
    count_log_likelihood = torch.sum(count_log_likelihood, axis=[1, 2])
    # Adjust for the fact that we're only using a minibatch.
    batch_size = len(counts)
    count_log_likelihood = count_log_likelihood * (
        self.num_documents / batch_size)
    return torch.mean(count_log_likelihood)
  
  def get_topic_means(self):
    """Get neutral and ideological topics from variational parameters.
    
    For each (k,v), we want to evaluate E[beta_kv], E[beta_kv * exp(eta_kv)], 
    and E[beta_kv * exp(-eta_kv)], where the expectations are with respect to 
    the variational distributions. Like the paper, beta refers to the obective 
    topic and eta refers to the ideological topic.
    
    Dropping the indices and denoting by mu_b the objective topic location and 
    sigma_b the objective topic scale, we have E[beta] = 
    exp(mu + sigma_b^2 / 2), using the mean of a lognormal distribution.
    
    Denoting by mu_e the ideological topic location and sigma_e the ideological
    topic scale, we have E[beta * exp(eta)] = E[beta]E[exp(eta)] by the 
    mean-field assumption. exp(eta) is lognormal distributed, so E[exp(eta)] =
    exp(mu_e + sigma_e^2 / 2). Thus, E[beta * exp(eta)] = 
    exp(mu_b + mu_e + (sigma_b^2 + sigma_e^2) / 2).
    
    Finally, E[beta * exp(-eta)] = 
    exp(mu_b - mu_e + (sigma_b^2 + sigma_e^2) / 2).
    
    Because we only care about the orderings of topics, we can drop the 
    exponents from the means.
    
    Returns:
      negative_mean: A tensor with shape [num_topics, num_words], denoting the
        variational mean for the ideological topics with an ideal point of -1.
      neutral_mean: A tensor with shape [num_topics, num_words] denoting the
        variational mean for the neutral topics.
      positive_mean: A tensor with shape [num_topics, num_words], denoting the
        variational mean for the ideological topics with an ideal point of +1.
    """
    objective_topic_loc = self.objective_topics.location
    objective_topic_scale = self.objective_topics.scale()
    ideological_topic_loc = self.ideological_topics.location
    ideological_topic_scale = self.ideological_topics.scale()
    
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
  
  def forward(self, document_indices, author_indices, counts, step):
    """Approximate variational Lognormal ELBO using reparameterization.
    
    Args:
      document_indices: An int-vector with shape `[batch_size]`.
      author_indices: An int-vector with shape `[batch_size]`.
      counts: A matrix with shape `[batch_size, num_words]`.
      step: The training step, used to log summaries to Tensorboard.
      
    Returns:
      elbo: A scalar representing a Monte-Carlo sample of the ELBO. This value 
        is averaged across samples and summed across batches.
    """
    samples = self.get_samples()
    log_prior = self.get_log_prior(samples)
    count_log_likelihood = self.get_count_log_likelihood(
        samples,
        document_indices, 
        author_indices, 
        counts)
    entropy = self.get_entropy(samples)
    elbo = count_log_likelihood + log_prior + entropy
    if step % self.print_steps == 0:
      self.summary_writer.add_scalar("elbo/entropy", entropy, step)
      self.summary_writer.add_scalar("elbo/log_prior", log_prior, step)
      self.summary_writer.add_scalar("elbo/count_log_likelihood",
                                     count_log_likelihood,
                                     step)
      self.summary_writer.add_scalar('elbo/elbo', elbo, step)
    return elbo


def print_topics(neutral_mean, negative_mean, positive_mean, vocabulary):
  """Get neutral and ideological topics to be used for Tensorboard.

  Args:
    neutral_mean: The mean of the neutral topics, a NumPy matrix with shape
      [num_topics, num_words].
    negative_mean: The mean of the negative topics, a NumPy matrix with shape
      [num_topics, num_words].
    positive_mean: The mean of the positive topics, a NumPy matrix with shape
      [num_topics, num_words].
    vocabulary: A list of the vocabulary with shape [num_words].

  Returns:
    topic_strings: A list of the negative, neutral, and positive topics.
  """
  num_topics, num_words = neutral_mean.shape
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
  return "  \n  \n".join(topic_strings)

  
def print_ideal_points(ideal_point_loc, author_map):
  """Print ideal point ordering for Tensorboard."""
  return ", ".join(author_map[np.argsort(ideal_point_loc)])  

  
def log_static_features(model, summary_writer, vocabulary, author_map, step):
  """Log static features (i.e. those that do not depend on ELBO calculation).
  
  Args:
    model: An object from class TBIP.
    summary_writer: Writer to log entries to Tensorboard.
    vocabulary: A list of the vocabulary with shape [num_words].
    author_map: A list of the author names with shape [num_authors].
    step: The current training step.
  """
  negative_mean, neutral_mean, positive_mean = model.get_topic_means()
  ideal_point_list = print_ideal_points(
      model.ideal_points.location.cpu().detach(),
      author_map)
  topics = print_topics(neutral_mean.cpu().detach(),
                        negative_mean.cpu().detach(),
                        positive_mean.cpu().detach(),
                        vocabulary)
  summary_writer.add_text("ideal_points", ideal_point_list, step)
  summary_writer.add_text("topics", topics, step)
  summary_writer.add_histogram("params/document_loc",
                               model.document_intensities.location,
                               step)
  summary_writer.add_histogram("params/document_scale",
                               model.document_intensities.scale(),
                               step)
  summary_writer.add_histogram("params/objective_topic_loc",
                               model.objective_topics.location,
                               step)
  summary_writer.add_histogram("params/objective_topic_scale",
                               model.objective_topics.scale(),
                               step)
  summary_writer.add_histogram("params/ideological_topic_loc",
                               model.ideological_topics.location,
                               step)
  summary_writer.add_histogram("params/ideological_topic_scale",
                               model.ideological_topics.scale(),
                               step)
  summary_writer.add_histogram("params/ideal_point_loc",
                               model.ideal_points.location,
                               step)
  summary_writer.add_histogram("params/ideal_point_scale",
                               model.ideal_points.scale(),
                               step)
  
  
def main(argv):
  del argv
  project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.pardir))
  source_dir = os.path.join(project_dir, "data/{}".format(FLAGS.data))

  # As described in the docstring, the data directory must have the following
  # files: counts.npz, author_indices.npy, vocabulary.txt, author_map.txt.
  data_dir = os.path.join(source_dir, "clean")
  save_dir = os.path.join(source_dir, "tbip-pytorch-fits")

  if os.path.exists(save_dir):
    print("Deleting old log directory at {}".format(save_dir))
    shutil.rmtree(save_dir)

  kwargs = ({'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() 
            else {})
  dataset = TBIPDataset(data_dir, 
                        counts_transformation=FLAGS.counts_transformation)
  data_loader = torch.utils.data.DataLoader(
      dataset,
      batch_size=FLAGS.batch_size,
      shuffle=True,
      **kwargs)
  iterator = data_loader.__iter__()

  if FLAGS.pre_initialize_parameters:
    fit_dir = os.path.join(source_dir, "pf-fits")
    fitted_document_shape = np.load(
        os.path.join(fit_dir, "document_shape.npy"))
    fitted_document_rate = np.load(
        os.path.join(fit_dir, "document_rate.npy"))
    fitted_topic_shape = np.load(
        os.path.join(fit_dir, "topic_shape.npy"))
    fitted_topic_rate = np.load(
        os.path.join(fit_dir, "topic_rate.npy"))
    initial_document_loc = fitted_document_shape / fitted_document_rate
    initial_objective_topic_loc = fitted_topic_shape / fitted_topic_rate
  else:
    num_documents, num_words = dataset.counts.shape
    initial_document_loc = np.exp(
        np.random.randn(num_documents, FLAGS.num_topics))
    initial_objective_topic_loc = np.exp(
        np.random.randn(FLAGS.num_topics, num_words))

  summary_writer = SummaryWriter(save_dir)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = TBIP(device, 
               dataset.author_weights,
               initial_document_loc, 
               initial_objective_topic_loc, 
               FLAGS.num_samples,
               FLAGS.print_steps,
               summary_writer).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

  start_time = time.time()
  for step in range(FLAGS.max_steps):
    try:
      document_indices, author_indices, counts = iterator.next()
    except StopIteration:
      iterator = data_loader.__iter__()
      document_indices, author_indices, counts = iterator.next()
    document_indices = document_indices.to(device)
    author_indices = author_indices.to(device)
    counts = counts.to(device)
    optimizer.zero_grad()
    elbo = model(document_indices, author_indices, counts, step)
    loss = -elbo
    loss.backward()
    optimizer.step()
    
    if step % FLAGS.print_steps == 0:
      duration = (time.time() - start_time) / (step + 1)
      print("Step: {:>3d} ELBO: {:.3f} ({:.3f} sec)".format(
            step, -loss, duration))
      summary_writer.add_scalar("loss", loss, step)
      log_static_features(model, 
                          summary_writer, 
                          dataset.vocabulary, 
                          dataset.author_map, 
                          step)
      if step % 1000 == 0 or step == FLAGS.max_steps - 1:
        param_save_dir = os.path.join(save_dir, "params/")
        if not os.path.exists(param_save_dir):
          os.makedirs(param_save_dir)
        
        np.save(os.path.join(param_save_dir, "document_loc"), 
                model.document_intensities.location.cpu().detach())
        np.save(os.path.join(param_save_dir, "document_scale"), 
                model.document_intensities.scale().cpu().detach())
        np.save(os.path.join(param_save_dir, "objective_topic_loc"), 
                model.objective_topics.location.cpu().detach())
        np.save(os.path.join(param_save_dir, "objective_topic_scale"), 
                model.objective_topics.scale().cpu().detach())
        np.save(os.path.join(param_save_dir, "ideological_topic_loc"), 
                model.ideological_topics.location.cpu().detach())
        np.save(os.path.join(param_save_dir, "ideological_topic_scale"), 
                model.ideological_topics.scale().cpu().detach())
        np.save(os.path.join(param_save_dir, "ideal_point_loc"), 
                model.ideal_points.location.cpu().detach())
        np.save(os.path.join(param_save_dir, "ideal_point_scale"), 
                model.ideal_points.scale().cpu().detach())


if __name__ == "__main__":
  app.run(main)
  