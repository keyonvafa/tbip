## Variational Inference with Gamma Distributions

We're updating our inference procedure so that Gamma variational families can be used for the positive variables (the document intensities `theta` and the objective topics `beta`). This allows these variational parameters to be updated directly with coordinate ascent variational inference (CAVI), rather than with reparameterization gradients. As a result, the optimization procedure converges more quickly. Moreover, the learned topics are more interpretable, and they do not need to be initialized with Poisson factorization.

Other updates:
  - Learned author verbosity: In our [original paper](https://aclanthology.org/2020.acl-main.475/), we described a method to set `author weights` so that the ideal points capture political
  preference rather than verbosity. We find that learning the 
  author verbosity improves performance, especially when using Gamma
  variational families.
  - Tensorflow 2 implementation: We've updated the code so it uses
  Tensorflow 2. This should make it easier to prototype and run than the Tensorflow 1 code.

### CAVI details

Using Gamma variational families for `theta` and `beta` allows for the variational parameters to be updated directly with coordinate ascent variational inference (CAVI). The main advantage over stochastic gradient descent is that, keeping the other variational parameters fixed, CAVI finds the true optimum of the ELBO for each update.

We introduce an auxiliary variable to our model to enable CAVI updates. Each word in each document has `K` auxiliary variables (where `K` is the number of topics), each corresponding to the share of the topic that contributes to the word count. For more details, refer to [Scalable Recommendation with Poisson Factorization (Gopalan et al., 2014)](https://arxiv.org/abs/1311.1704).

The variational updates for `theta` and `beta` follow closely from the updates in regular [Poisson factorization](https://arxiv.org/abs/1311.1704). The main difference is that the TBIP requires computing the variational expectation of the ideological term

```
E[exp(eta_kv * x_{a_d} + w_{a_d})]
```
where `a_d` is the author of document `d` and `w_{a_d}` is the author verbosity. This is intractable, so we approximate it with the geometric mean instead:

```
exp(E[log(exp(eta_kv * x_{a_d} + w_{a_d}))]) = exp(E[eta_kv] * E[x_{a_d}] + E[w_{a_d}]).
```

### Learning author verbosity

The only difference on the modeling side is that the per-author verbosity is learned, rather than imputed from the observed counts. We posit a Gaussian prior, `w_a ~ N(0, 1)` for each author `a`. The ideological term becomes 
```
exp(eta_kv * x_{a_d} + w_{a_d})
```
rather than
```
exp(eta_kv * x_{a_d}).
```
We can interpret `w_{a_d}` as a baseline measure of how verbose the author of document `d` is.

We approximate the posterior of `w_a` with variational inference, using a Gaussian variational family. The variational parameters are optimized using reparameterization gradients.

### Running the code

We recommend running the code with

```{bash}
python tbip_with_gamma.py  --data $DATASET_NAME \
  --batch_size=256 --save_every=20 \
  --positive_variational_family=gamma --cavi=True \
  --checkpoint_name=gamma_variational_families_cavi
```
replacing `$DATASET_NAME` with the name of the dataset. The batch size depends on the size of the vocabulary. We recommend using the largest batch size that fits in memory, but 256 is a good default. If you want to initialize the code with Poisson factorization, you can run

```{bash}
python tbip_with_gamma.py  --data $DATASET_NAME \
  --batch_size=256 --save_every=20 \
  --positive_variational_family=gamma --cavi=True \
  --pre_initialize_parameters=True \
  --checkpoint_name=gamma_variational_families_cavi_initialized
```

If you want to perform inference with lognormal variational families, you can run

```{bash}
python tbip_with_gamma.py  --data $DATASET_NAME \
  --batch_size=256 --save_every=20 \
  --positive_variational_family=lognormal --cavi=True \
  --pre_initialize_parameters=True \
  --checkpoint_name=lognormal_variational_families_initialized
```

An added benefit to our code is checkpointing -- you can stop running the code in the middle of training, and it will pick up right where you left off. If you want to re-run without loading from a checkpoint, you can set `--load_checkpoint=False`.

Like the original code, the folder `data/{dataset_name}/clean/` should contain four files:

* `counts.npz`: a `[num_documents, num_words]` [sparse CSR matrix](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.html) containing the word counts for each document.
* `author_indices.npy`: a `[num_documents]` vector where each entry is an integer in the set `{0, 1, ..., num_authors - 1}`, indicating the author of the corresponding document in `counts.npz`.
* `vocabulary.txt`: a `[num_words]`-length file where each line denotes the corresponding word in the vocabulary.
* `author_map.txt`: a `[num_authors]`-length file where each line denotes the name of an author in the corpus.
  
See [data/senate-speeches-114/clean](https://github.com/keyonvafa/tbip/tree/master/data/senate-speeches-114/clean) for an example of what the four files look like for Senate speeches.
