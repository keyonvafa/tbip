import numpy as np


def bincount_2d(x, weights):
  """Sum weighted number of occurrences in matrix of non-negative ints.
  
  Args:
    x: An array with shape `[num_documents]` with each entry between 
      `{0, 1, ..., num_classes - 1}`.
    weights: A matrix with shape `[num_documents, num_topics].`
  
  Returns:
    counts: A matrix with shape `[num_classes, num_topics]`. The k'th entry is 
      the sum of all the vectors in `weights` that are assigned class `k`.
  """
  _, num_topics = weights.shape
  num_cases = np.max(x) + 1
  counts = np.array(
      [np.bincount(x, weights=weights[:, topic], minlength=num_cases)
       for topic in range(num_topics)])
  return counts.T


def remove_cooccurring_ngrams(counts, vocabulary):
  """Remove unigrams and n-grams that co-occur from counts and vocabulary.
  
  Our data includes both unigrams and n-grams. Since each n-gram consists of
  multiple unigrams, this will result in undesired co-occuruences. For example,
  if we have a bigram of "health care", a document that includes this will
  also include a count for "health" and a count for "care" (in addition to the
  bigram counts). We fix this by subtracting the corresponding unigram counts
  each time an n-gram appears.
  
  Args:
    counts: A sparse matrix with shape [num_documents, num_words].
    vocabulary: A vector with shape [num_words], containing the vocabulary.
  
  Returns:
    counts_dense: A dense matrix with shape [num_documents, num_words], where
      the co-occuring n-grams are removed.
  """
  # `n_gram_to_unigram` takes as key an index to an n-gram in the vocabulary
  # and its value is a list of the vocabulary indices of the corresponding 
  # unigrams.
  n_gram_indices = np.where(
      np.array([len(word.split(' ')) for word in vocabulary]) > 1)[0]
  n_gram_to_unigrams = {}
  for n_gram_index in n_gram_indices:
    matching_unigrams = []
    for unigram in vocabulary[n_gram_index].split(' '):
      if unigram in vocabulary:
        matching_unigrams.append(np.where(vocabulary == unigram)[0][0])
    n_gram_to_unigrams[n_gram_index] = matching_unigrams

  # `n_grams_to_bigrams` now breaks apart trigrams and higher to find bigrams 
  # as subsets of these words.
  n_grams_to_bigrams = {}
  for n_gram_index in n_gram_indices:
    split_n_gram = vocabulary[n_gram_index].split(' ')
    n_gram_length = len(split_n_gram) 
    if n_gram_length > 2:
      bigram_matches = []
      for i in range(0, n_gram_length - 1):
        bigram = " ".join(split_n_gram[i:(i + 2)])
        if bigram in vocabulary:
          bigram_matches.append(np.where(vocabulary == bigram)[0][0])
      n_grams_to_bigrams[n_gram_index] = bigram_matches

  # Go through counts, and remove a unigram each time a bigram superset 
  # appears. Also remove a bigram each time a trigram superset appears.
  # Note this isn't perfect: if bigrams overlap (e.g. "global health care" 
  # contains "global health" and "health care"), we count them both. This
  # may introduce a problem where we subract a unigram count twice, so we also
  # ensure non-negativity.
  counts_dense = counts.toarray()
  for i in range(len(counts_dense)):
    n_grams_in_doc = np.where(counts_dense[i, n_gram_indices] > 0)[0]
    sub_n_grams = n_gram_indices[n_grams_in_doc]
    for n_gram in sub_n_grams:
      counts_dense[i, n_gram_to_unigrams[n_gram]] -= counts_dense[i, n_gram]
      if n_gram in n_grams_to_bigrams:
        counts_dense[i, n_grams_to_bigrams[n_gram]] -= counts_dense[i, n_gram]
  counts_dense = np.maximum(counts_dense, 0)
  return counts_dense
