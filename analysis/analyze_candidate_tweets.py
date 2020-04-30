"""Analyze TBIP results on 2020 Democratic candidate tweets."""

import os
import numpy as np
import analysis_utils as utils

project_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)) 
source_dir = os.path.join(project_dir, "data/candidate-tweets-2020")

# Load TBIP data.
data_dir = os.path.join(source_dir, "clean")
(counts, vocabulary, author_indices, 
 author_map, raw_documents) = utils.load_text_data(data_dir)

# Load TBIP parameters.
param_dir = os.path.join(source_dir, "tbip-fits/params/")
(document_loc, document_scale, objective_topic_loc, objective_topic_scale, 
 ideological_topic_loc, ideological_topic_scale, ideal_point_loc, 
 ideal_point_scale) = utils.load_tbip_parameters(param_dir)

# Print topics. Interesting topics are: 4, 43, 47.
utils.print_topics(objective_topic_loc, 
                   objective_topic_scale,
                   ideological_topic_loc, 
                   ideological_topic_scale, 
                   vocabulary)

# Print ideal point orderings.
for i in range(len(author_map)):
  index = np.argsort(ideal_point_loc)[i]
  print("{}: {:.2f}".format(author_map[index], ideal_point_loc[index]))
