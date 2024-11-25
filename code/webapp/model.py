import numpy as np

words = dict()

def add_to_dict(d, filename):
  with open(filename, 'r',encoding='utf-8') as f:
    for line in f.readlines():
      line = line.split(' ')

      try:
        d[line[0]] = np.array(line[1:], dtype=float)
      except:
        continue

add_to_dict(words, 'glove.6B.50d.txt')

import nltk

nltk.download('wordnet')

tokenizer = nltk.RegexpTokenizer(r"\w+")

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def message_to_token_list(s):
  tokens = tokenizer.tokenize(s)
  lowercased_tokens = [t.lower() for t in tokens]
  lemmatized_tokens = [lemmatizer.lemmatize(t) for t in lowercased_tokens]
  useful_tokens = [t for t in lemmatized_tokens if t in words]
  return useful_tokens

def message_to_word_vectors(message, word_dict=words):
  processed_list_of_tokens = message_to_token_list(message)

  vectors = []

  for token in processed_list_of_tokens:
    if token not in word_dict:
      continue

    token_vector = word_dict[token]
    vectors.append(token_vector)

  return np.array(vectors, dtype=float)

def df_to_X_y(dff):
  y = dff['id'].to_numpy().astype(int)

  all_word_vector_sequences = []

  for message in dff['ingredients']:
    message_as_vector_seq = message_to_word_vectors(message)

    if message_as_vector_seq.shape[0] == 0:
      message_as_vector_seq = np.zeros(shape=(1, 50))

    all_word_vector_sequences.append(message_as_vector_seq)

  return all_word_vector_sequences, y






from copy import deepcopy

def pad_X(X, desired_sequence_length=30):
  X_copy = deepcopy(X)

  for i, x in enumerate(X):
    x_seq_len = x.shape[0]
    sequence_length_difference = desired_sequence_length - x_seq_len

    pad = np.zeros(shape=(sequence_length_difference, 50))

    X_copy[i] = np.concatenate([x, pad])

  return np.array(X_copy).astype(float)





from tkinter.constants import X




import torch
import torch.nn as nn


class MyMLP(nn.Module):
  def __init__(self, D,F):
    super(MyMLP, self).__init__()
    self.a1 = nn.Linear(D, 300)
    self.z1 = nn.Tanh()
    self.a2 = nn.Linear(300, 500)
    self.z2 = nn.Tanh()
    self.a3 = nn.Linear(500, F)


  def forward(self, x):
     a = self.a1(x)
     z = self.z1(a)
     a = self.a2(z)
     z = self.z2(a)
     y_hat = self.a3(z)
     return y_hat
  
model = MyMLP(1500,1000)
model.load_state_dict(torch.load("Model1.pth", weights_only=True))

def prediction(text_array):
    test1 = " ".join(text_array)
    test1 = message_to_word_vectors(test1)
    test1 = pad_X([test1])
    test1 = test1.reshape(1, -1)
  
    with torch.no_grad():
        test1 = torch.tensor(test1, dtype=torch.float32)
        y_hat = model(test1)
        topk_values, topk_indices = torch.topk(y_hat, k=3, dim=1)
        top=[]
        for i in range(3):
            top.append(topk_indices[0,i].item())
        
    return top
    

