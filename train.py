##################################################################################
# 
# Recurrent Neural Network to disambiguate instances of 'back' 
# glosses/synsets are from Wordnet 3.0
# training and testing sentences are from the SEMCOR corpus accessed through NLTK
# word encodings done with the GENSIM implementation of the Word2Vec alogorithm
# RNN based on Victor Zhou's Vanilla Neural Network tutorial

import gensim.models
import numpy as np
from rnn import RNN
from nltk.corpus import semcor
import tempfile
import pickle
import random


##################################################################################
# Load preprocessed data
#
# IMPORTANT - there is a tokenization difference between SEMCOR and Word2VEC
# Word2Vec uses simple tokenization - (e.g. 'mail', 'man' would be two words)
# SEMCOR chunks tokens (e.g. 'mail man' would be one word)
# this doesn't significantly impact this project but prevents its extension to general disambiguation tasks

targetword = 'back'           # 'on', 'talk', 'think', 'went', 'in'', 'cost'
print('Loading Data')
# Load Word2Vec word vectors, already created on SEMCOR corpus
with tempfile.NamedTemporaryFile(prefix='gensim-model-', delete=False) as tmp:
  temporary_filepath = 'gensim-model-zviw4kpu'
  model = gensim.models.Word2Vec.load(temporary_filepath)

# Load training / testing data
with open('train_data.p', 'rb') as f:
  train_data = pickle.load(f)
with open('test_data.p', 'rb') as f:
  test_data = pickle.load(f)

# Load glosses of targetword
with open('targetsynset.p', 'rb') as f:
  targetsynset = pickle.load(f)
targetsynsetlist = list(targetsynset)

# load all lemma
# with open('lemma.p', 'rb') as f:
#   lemma = pickle.load(f)

#load sense tags
#taggedsentences = semcor.tagged_sents(tag='both')

# max_train_sent_len = max(len(sentence) for sentence in train_data)
# max_test_sent_len = max(len(sentence) for sentence in test_data)
# max_sent_len = max(max_test_sent_len, max_train_sent_len)

print('Data Loading Done')


##################################################################################
# Initialize RNN

rnn = RNN(300, len(targetsynsetlist))

def softmax(xs):
  # Applies the Softmax Function to the input array.
  return np.exp(xs) / sum(np.exp(xs))

def createInputs(text):
  # Returns an array of word vectors 
  
  inputs = []
  for w in text.split(' '):
    if w in model.wv:
      v = model.wv[w]
      v.shape = (300, 1)
    else:
      v = np.zeros((300, 1))
    inputs.append(v)
  return inputs

def processData(data, backprop=True):
  '''
  Returns the RNN's loss and accuracy for the given data.
  - data is a dictionary mapping sentences to a synset index in targetsynsetlist.
  - backprop determines if the backward phase should be run.
  '''
  items = list(data.items())
  random.shuffle(items)

  loss = 0
  num_correct = 0

  for x, y in items:
    inputs = createInputs(x.lower())
    target = int(y)

    # Forward
    out, _ = rnn.forward(inputs)
    probs = softmax(out)

    # Calculate loss / accuracy
    loss -= np.log(probs[target])
    num_correct += int(np.argmax(probs) == target)

    if backprop:      
      # Build dL/dy
      d_L_d_y = probs
      d_L_d_y[target] -= 1

      # Backward
      rnn.backprop(d_L_d_y)

  return loss / len(data), num_correct / len(data)

# Training loop
for epoch in range(10000):
  train_loss, train_acc = processData(train_data)

  if epoch % 100 == 99:
    print('--- Epoch %d' % (epoch + 1))
    print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

    test_loss, test_acc = processData(test_data, backprop=False)
    print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))

with open('wxh.p', 'wb') as f:
    pickle.dump(rnn.Wxh, f)
with open('whh.p', 'wb') as f:
    pickle.dump(rnn.Whh, f)
with open('why.p', 'wb') as f:
    pickle.dump(rnn.Why, f)
with open('bh.p', 'wb') as f:
    pickle.dump(rnn.bh, f)
with open('by.p', 'wb') as f:
    pickle.dump(rnn.by, f)