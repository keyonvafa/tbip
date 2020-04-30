# Text-Based Ideal Points
Source code for the paper: Text-Based Ideal Points by Keyon Vafa, Suresh Naidu, and David Blei (ACL 2020).

## Installation for GPU
Configure a virtual environment using Python 3.6+. Inside the virtual environment, use `pip` to install the required packages:

```{bash}
(venv)$ pip install -r requirements.txt
```
The main depednencies are Tensorflow (1.14.0) and Tensorflow Probability (0.7.0).

## Installation for CPU
To run on CPU, a version of Tensorflow that does not use GPU must be installed. In 
[requirements.txt](https://github.com/keyonvafa/tbip/blob/master/requirements.txt),
comment out the line that says `tensorflow-gpu==1.14.0` and uncomment the line that says 
`tensorflow==1.14.0`. Note: the script will be noticeably slower on CPU.

## Data
Preprocessed Senate speech data for the 114th Congress is included in 
[data/senate-speeches-114](https://github.com/keyonvafa/tbip/tree/master/data/senate-speeches-114). 
The original data is from [[1]](#1). Preprocessed 2020 Democratic presidential candidate tweet data is included in
[data/candidate-tweets-2020](https://github.com/keyonvafa/tbip/tree/master/data/candidate-tweets-2020). 

To include a customized data set, first create a repo `data/{dataset_name}/clean/`. The
following four files must be inside this folder:

* `counts.npz`: a `[num_documents, num_words]` 
  [sparse CSR matrix](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.csr_matrix.html) 
  containing the
  word counts for each document.
* `author_indices.npy`: a `[num_documents]` vector where each entry is an
  integer in the set `{0, 1, ..., num_authors - 1}`, indicating the author of 
  the corresponding document in `counts.npz`.
* `vocabulary.txt`: a `[num_words]`-length file where each line denotes
  the corresponding word in the vocabulary.
* `author_map.txt`: a `[num_authors]`-length file where each line denotes
  the name of an author in the corpus.
  
See [data/senate-speeches-114/clean](https://github.com/keyonvafa/tbip/tree/master/data/senate-speeches-114/clean) 
for an example of what the four files look like for Senate speeches. The script 
[setup/senate_speeches_to_bag_of_words.py](https://github.com/keyonvafa/tbip/blob/master/setup/senate_speeches_to_bag_of_words.py) 
contains example code for creating the four files from unprocessed data.

## Learning text-based ideal points
Run [tbip.py](https://github.com/keyonvafa/tbip/blob/master/tbip.py) to produce ideal points.
For the Senate speech data, use the command:
```{bash}
(venv)$ python tbip.py  --data=senate-speeches-114  --batch_size=512
```
You can view Tensorboard while training to see summaries of training (including the learned ideal points
and ideological topics). To run Tensorboard, use the command: 
```{bash}
(venv)$ tensorboard  --logdir=data/senate-speeches-114/tbip-fits/  --port=6006
```
The command should output a link where you can view the Tensorboard results in real time.
The fitted parameters will be stored in `data/senate-speeches-114/tbip-fits/params`.
To perform the above analyses for the 2020 Democratic candidate tweets, replace `senate-speeches-114` 
with `candidate-tweets-2020`.

To run custom data, we recommend training Poisson factorization before running the TBIP script
for best results. If you have custom data stored in `data/{dataset_name}/clean/`, you can run

```{bash}
(venv)$ python setup/poisson_factorization.py  --data={dataset_name}
```
The default number of topics is 50. To use a different number of topics, e.g. 100, use the flag `--num_topics=100`. 
After Poisson factorization finishes, use the following command to run the TBIP:
```{bash}
(venv)$ python tbip.py  --data={dataset_name}
```
You can adjust the batch size, learning rate, number of topics, and number of steps by using the flags
`--batch_size`, `--learning_rate`, `--num_topics`, and `--max_steps`, respectively.
To run the TBIP without initializing from Poisson factorization, use the flag `--pre_initialize_parameters=False`.
To view the results in Tensorboard, run
```{bash}
(venv)$ tensorboard  --logdir=data/{dataset_name}/tbip-fits/  --port=6006
```
Again, the learned parameters will be stored in `data/{dataset_name}/tbip-fits/params`.

## Reproducing Paper Results
The following commands will reproduce all of the paper results. The following data is required:

* Senate votes: The original raw data can be found at [[2]](#2). The paper includes experiments
for Senate sessions 111-114. For each Senate session, we need three files: one for votes,
one for members, and one for rollcalls. For example, for Senate session 114,
we would use the files: `S114_votes.csv`, `S114_members.csv`, `S114_rollcalls.csv`. Make a repo
`data/senate-votes` and store these three files in `data/senate-votes/114/raw/`. Repeat
for Senate sessions 111-113.
* Senate speeches: The original raw data can be found at [[1]](#1). Specifically, we use the `hein-daily` 
data for the 114th Senate session. The files needed are `speeches_114.txt`, `descr_114.txt`, 
and `114_SpeakerMap.txt`. Make sure the relevant files are stored in 
`data/senate-speeches-114/raw/`.
* Senator tweets: The data was provided to us by Voxgov [[3]](#3).
* Senate speech comparisons: We use a separate data set for the Senate speech comparisons
because speech debates must be labeled for Wordshoal. The raw data can be found at [[4]](#4).
The paper includes experiments for Senate sessions 111-113. We need the files 
`speaker_senator_link_file.csv`, `speeches_Senate_111.tab`, `speeches_Senate_112.tab`, 
and `speeches_Senate_113.tab`. These files should all be stored in `data/senate-speech-comparisons/raw/`.
* Democratic presidential candidate tweets: The code requires that a file containing the raw tweets 
(`tweets.csv`) be stored in `data/candidate-tweets-2020/raw/`. We are working on making this data available.

## References
<a id="1">[1]</a> 
Gentzkow, Matthew, Jesse M. Shapiro, and Matt Taddy. Congressional Record for the 43rd-114th Congresses: Parsed Speeches and Phrase Counts. Palo Alto, CA: Stanford Libraries [distributor], 2018-01-16. https://data.stanford.edu/congress_text

<a id="2">[2]</a> 
Lewis, Jeffrey B., Keith Poole, Howard Rosenthal, Adam Boche, Aaron Rudkin, and Luke Sonnet (2020). Voteview: Congressional Roll-Call Votes Database. https://voteview.com/

<a id="3">[3]</a> 
VoxGovFEDERAL, U.S. Senators tweets from the 114th Congress. 2020. https://voxgov.com.  

<a id="4">[4]</a> 
Benjamin E. Lauderdale and Alexander Herzog. Replication Data for: Measuring Political Positions from Legislative Speech. In Harvard Dataverse, 2016. https://doi.org/10.7910/DVN/RQMIV3.
