import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('paper')


def standardize(x):
    """Standardize a vector x."""
    return (x - np.mean(x)) / np.std(x)
  

project_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)) 
source_dir = os.path.join(project_dir, "data")
  
voter_map = np.loadtxt(
    os.path.join(source_dir, "senate-votes/114/clean/senator_map.txt"),
    dtype=str,
    delimiter="\n")
vote_ideal_points = np.load(os.path.join(
    source_dir, 
    "senate-votes/114/fits/params/ideal_point_loc.npy"))
speech_author_map = np.loadtxt(
    os.path.join(source_dir, "senate-speeches-114/clean/author_map.txt"),
    dtype=str,
    delimiter="\n")
speech_ideal_points = np.load(os.path.join(
    source_dir, 
    "senate-speeches-114/tbip-fits/params/ideal_point_loc.npy"))
tweet_author_map = np.loadtxt(
    os.path.join(source_dir, "senate-tweets-114/clean/author_map.txt"),
    dtype=str,
    delimiter="\n")
tweet_ideal_points = np.load(os.path.join(
    source_dir, 
    "senate-tweets-114/tbip-fits/params/ideal_point_loc.npy"))
candidate_author_map = np.loadtxt(
    os.path.join(source_dir, "candidate-tweets-2020/clean/author_map.txt"),
    dtype=str,
    delimiter="\n")
candidate_ideal_points = np.load(os.path.join(
    source_dir, 
    "candidate-tweets-2020/tbip-fits/params/ideal_point_loc.npy"))

# Make Sanders on left for all ideal points so liberals are on left.
sanders_vote_index = np.where(voter_map == "Bernard Sanders (I)")[0][0]
if np.sign(vote_ideal_points[sanders_vote_index]) == 1:
    vote_ideal_points = vote_ideal_points * -1
sanders_speech_index = np.where(
    speech_author_map == "Bernard Sanders (I)")[0][0]
if np.sign(speech_ideal_points[sanders_speech_index]) == +1:
    speech_ideal_points = speech_ideal_points * -1
sanders_tweet_index = np.where(
    tweet_author_map == "Bernard Sanders (I)")[0][0]
if np.sign(tweet_ideal_points[sanders_tweet_index]) == +1:
    tweet_ideal_points = tweet_ideal_points * -1
sanders_candidate_index = np.where(
    candidate_author_map == "Berniesanders")[0][0]
if np.sign(candidate_ideal_points[sanders_candidate_index]) == +1:
    candidate_ideal_points = candidate_ideal_points * -1

# Figure 1: Senate speech ideal points.
sns.set(style="whitegrid")
colors = np.array(["r" if senator[-2] == "R" else "b" 
                   for senator in speech_author_map])
markers = np.array(["x" if senator[-2] == "R" else "o" 
                    for senator in speech_author_map])
sizes = np.array([60 if senator[-2] == "R" else 40 
                  for senator in speech_author_map])
fig = plt.figure(figsize=(12, 2))
ax = plt.axes([0, 0, 1, 1.2], frameon=False)
for index in range(len(speech_ideal_points)):
    ax.scatter(speech_ideal_points[index], 0, c=colors[index], 
               marker=markers[index], s=sizes[index])
ax.set_yticks([])
plt.savefig(os.path.join(project_dir, "analysis/figs/figure_1.pdf"), 
            dpi=300, 
            bbox_inches='tight')

# Figure 2: Vote/Senate speech/Senate tweet ideal point comparison.
standardized_vote_ideal_points = standardize(vote_ideal_points)
standardized_speech_ideal_points = standardize(speech_ideal_points)
standardized_tweet_ideal_points = standardize(tweet_ideal_points)

sns.set(style="whitegrid")
fig = plt.figure(figsize=(12, 2))
ax = plt.axes([0, 0, 1, 1.5], frameon=False)

vote_colors = np.array(["r" if senator[-2] == "R" else "b" 
                        for senator in voter_map])
vote_markers = np.array(["x" if senator[-2] == "R" else "o" 
                         for senator in voter_map])
vote_sizes = np.array([60 if senator[-2] == "R" else 40 for 
                       senator in voter_map])
for index in range(len(voter_map)):
    ax.scatter(standardized_vote_ideal_points[index], 1,
               c=vote_colors[index], marker=vote_markers[index], 
               s=vote_sizes[index])

speech_colors = np.array(["r" if senator[-2] == "R" else "b" 
                          for senator in speech_author_map])
speech_markers = np.array(["x" if senator[-2] == "R" else "o" 
                           for senator in speech_author_map])
speech_sizes = np.array([60 if senator[-2] == "R" else 40 
                         for senator in speech_author_map])
for index in range(len(speech_author_map)):
    ax.scatter(standardized_speech_ideal_points[index], 0,
               c=speech_colors[index], marker=speech_markers[index], 
               s=speech_sizes[index])
    
tweet_colors = np.array(["r" if senator[-2] == "R" else "b" 
                         for senator in tweet_author_map])
tweet_markers = np.array(["x" if senator[-2] == "R" else "o" 
                          for senator in tweet_author_map])
tweet_sizes = np.array([60 if senator[-2] == "R" else 40 
                        for senator in tweet_author_map])
for index in range(len(tweet_author_map)):
    ax.scatter(standardized_tweet_ideal_points[index], -1,
               c=tweet_colors[index], marker=tweet_markers[index], 
               s=tweet_sizes[index])

ax.set_yticks([])
ax.set_xticks([])
plt.savefig(os.path.join(project_dir, "analysis/figs/figure_2.pdf"), 
            dpi=300, 
            bbox_inches='tight')

# Figure 3: Ideal points for 2020 Democratic presidential candidates.
sns.set(style="whitegrid")
fig = plt.figure(figsize=(12, 1))
ax = plt.axes([0, 0, 1, 1], frameon=False)
ax.scatter(candidate_ideal_points, np.zeros(len(candidate_ideal_points)), 
           c='black', s=45)
ax.set_yticks([])
plt.savefig(os.path.join(project_dir, "analysis/figs/figure_3.pdf"), 
            dpi=300, 
            bbox_inches='tight')
