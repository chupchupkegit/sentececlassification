from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import tensorflow as tf
import pickle
import pandas as pd
import numpy as np


def get_word_to_id(data_path=None):
  word_to_id_path = os.path.join(data_path, "common_words_index.p")
  word_to_id = pickle.load(open(word_to_id_path,"rb"))
  return word_to_id

def get_embedding(data_path=None):
  embedding_path = os.path.join(data_path, "word_embedding_index.p")
  embedding = pickle.load(open(embedding_path,"rb"))
  return embedding

def string_to_word_ids(s, word_to_id):
  return [word_to_id[word] for word in s.split()]

def ptb_raw_data(data_path=None):
  word_to_id = get_word_to_id(data_path)
  train_path = os.path.join(data_path, "cleaned_data/train.csv")
  valid_path = os.path.join(data_path, "cleaned_data/valid.csv")
  test_path = os.path.join(data_path, "cleaned_data/test.csv")

  train_data={}
  df = pd.read_csv(train_path)
  train_data["complaint_labels"] = np.array(df['data_class'].tolist()).astype(dtype='int32')
  complaint_titles = df['cleaned_title'].tolist()
  train_data["complaint_titles"] = np.array([string_to_word_ids(title,word_to_id) for title in complaint_titles]).astype(dtype='int32')

  test_data={}
  df = pd.read_csv(test_path)
  test_data["complaint_labels"] = np.array(df['data_class'].tolist()).astype(dtype='int32')
  complaint_titles = df['cleaned_title'].tolist()
  test_data["complaint_titles"] = np.array([string_to_word_ids(title,word_to_id) for title in complaint_titles]).astype(dtype='int32')

  valid_data={}
  df = pd.read_csv(valid_path)
  valid_data["complaint_labels"] = np.array(df['data_class'].tolist()).astype(dtype='int32')
  complaint_titles = df['cleaned_title'].tolist()
  valid_data["complaint_titles"] = np.array([string_to_word_ids(title,word_to_id) for title in complaint_titles]).astype(dtype='int32')

  return train_data,test_data,valid_data


