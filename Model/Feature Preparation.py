#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import pickle

np.random.seed(42)

# --- Load list from CSV ---
def extract_list_from_csv(file_path, column_index=0):
    df = pd.read_csv(file_path)
    return df.iloc[:, column_index].tolist()

# --- Load feature matrix ---
def extract_feature_matrix(file_path):
    return pd.read_csv(file_path, index_col=0)

# --- Create combined features for entity pairs ---
def create_combined_features(entity1_id, entity2_id, entity1_matrices, entity2_matrices):
    combined_features = []
    for e1_matrix in entity1_matrices:
        for e2_matrix in entity2_matrices:
            combined = np.hstack((e1_matrix.loc[entity1_id].values, e2_matrix.loc[entity2_id].values))
            combined_features.append(combined)
    return combined_features

# --- Load entity info lists ---
entity1_list = extract_list_from_csv('entity1_info.csv')  # e.g., drugsInfo.csv
entity2_list = extract_list_from_csv('entity2_info.csv')  # e.g., diseasesInfo.csv

# --- Load standardized feature matrices ---
entity1_feature_files = [
    'entity1_feature1.csv',
    'entity1_feature2.csv',
    'entity1_feature3.csv',
    'entity1_feature4.csv'
]

entity2_feature_files = [
    'entity2_feature1.csv',
    'entity2_feature2.csv',
    'entity2_feature3.csv'
]

entity1_feature_matrices = [extract_feature_matrix(file) for file in entity1_feature_files]
entity2_feature_matrices = [extract_feature_matrix(file) for file in entity2_feature_files]

# --- Load interaction data ---
interactions_df = pd.read_csv('entity_interactions.csv')  # Columns: Entity1ID, Entity2ID, label

# --- Generate features and labels ---
all_features = []
all_labels = []

for i in range(len(interactions_df)):
    entity1_id = interactions_df['Entity1ID'][i]
    entity2_id = interactions_df['Entity2ID'][i]
    label = interactions_df['label'][i]

    features = create_combined_features(entity1_id, entity2_id, entity1_feature_matrices, entity2_feature_matrices)
    all_features.append(np.array(features))
    all_labels.append(np.array(label))

# --- Save processed data ---
with open("new_features_entity1_entity2.pkl", "wb") as f_feat:
    pickle.dump(all_features, f_feat)

with open("new_labels_entity1_entity2.pkl", "wb") as f_label:
    pickle.dump(all_labels, f_label)

