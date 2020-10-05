import os
import logging
import json
import random
import time
import torch
from transformers import (
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    BertForTokenClassification,
    DistilBertForTokenClassification,
    BertConfig,
    DistilBertConfig,
    AdamW,
    get_linear_schedule_with_warmup,
    BertTokenizer,
    DistilBertTokenizer
)
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler, SequentialSampler

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from utils import set_all_seeds

logger = logging.getLogger(__name__)

############## define functions for 
# - tokenizing
# - optimizing
# - modeling
# - predicting

set_all_seeds(seed=42)

## 1. tokenization (tokenizing, mapping, padding)

def tokenize_and_preserve_labels(tokenizer, sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels

def plot_max_seq_length(df, tokenizer, max_len_plot = 256):
  # function for plotting distribution of sequence lengths 
  # --> helps to choose appropriate seq max length
  # need tokenizer here!
  
  #store the token length of each review
  token_lens = []
  for txt in df.text:
    tokens = tokenizer.encode(txt, max_length = max_len_plot, truncation = True)
    token_lens.append(len(tokens))

  # plot distribution 
  sns.distplot(token_lens)
  plt.xlim([0, max_len_plot])
  plt.xlabel('Token count')

def token_mapping(sentences, tokenizer, max_len, train = True):
  if train:
    input_ids = []

  # For every sentence...
    for sent in sentences:
    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
      encoded_sent = tokenizer.encode(
                          sent,                      # Sentence to encode.
                          add_special_tokens = True, # Add '[CLS]' and '[SEP]'

                          # This function also supports truncation and conversion
                          # to pytorch tensors, but we need to do padding, so we
                          # can't use these features :( .
                          max_length = max_len,       # Truncate all sentences.
                          #return_tensors = 'pt',     # Return pytorch tensors.
                          truncation = True
                    )
    
      # Add the encoded sentence to the list.
      input_ids.append(encoded_sent)
  else:
    input_ids = []

    # For every sentence...
    for sent in sentences:
    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
      encoded_sent = tokenizer.encode(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        truncation = True
                   )
    
      # Add the encoded sentence to the list.
      input_ids.append(encoded_sent)

  return input_ids

# We'll borrow the `pad_sequences` utility function to do this.

def padding(input_ids, tokenizer, max_len):
  print('\nPadding/truncating all sentences to %d values...' % max_len)
  print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))
  # Pad our input tokens with value 0.
  # "post" indicates that we want to pad and truncate at the end of the sequence,
  # as opposed to the beginning.
  input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", 
                          value=0, truncating="post", padding="post")
  return input_ids

# masking
def masking(input_ids):
  # Create attention masks
  attention_masks = []

  # For each sentence...
  for sent in input_ids:
    
    # Create the attention mask.
    #   - If a token ID is 0, then it's padding, set the mask to 0.
    #   - If a token ID is > 0, then it's a real token, set the mask to 1.
    att_mask = [int(token_id > 0) for token_id in sent]
    
    # Store the attention mask for this sentence.
    attention_masks.append(att_mask)

  return attention_masks

# train/dev split
def split_train_dev(train_df, dev_df, attention_masks, input_ids, labels):
  attention_masks = np.array(attention_masks)
  train_inputs = input_ids[:len(train_df)]
  dev_inputs = input_ids[len(train_df):]
  train_labels = labels[:len(train_df)]
  dev_labels = labels[len(train_df):]
  train_masks = attention_masks[:len(train_df)]
  dev_masks = attention_masks[len(train_df):]

  return train_inputs, train_labels, dev_inputs, dev_labels, train_masks, dev_masks

def create_dataloader(df_inputs, df_masks, df_labels, batch_size, train = True):
  # create DataLoader
  data = TensorDataset(df_inputs, df_masks, df_labels)
  # Create the DataLoader for our training set.
  if train:
      sampler = RandomSampler(data)
  # Create the DataLoader for our validation set.
  else:
      sampler = SequentialSampler(data)

  dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
  
  return dataloader


def get_pretrained_lm_and_params(num_labels, model_type = "BERT", 
                 pretrained_weights = "bert-base-german-cased"):

  if model_type == "BERT":
    model = BertForSequenceClassification.from_pretrained(
      pretrained_weights,
      num_labels = num_labels, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
      output_attentions = False, # Whether the model returns attentions weights.
      output_hidden_states = False # Whether the model returns all hidden-states.
    )

  if model_type == "DistilBERT":
    model = DistilBertForSequenceClassification.from_pretrained(
      pretrained_weights,
      num_labels = num_labels, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
      output_attentions = False, # Whether the model returns attentions weights.
      output_hidden_states = False # Whether the model returns all hidden-states.
    )
  
  # Get all of the model's parameters as a list of tuples.
  named_params = list(model.named_parameters())

  return model, named_params

### 3. modeling

def create_scheduler(train_dataloader, epochs, optimizer):
  # Total number of training steps is number of batches * number of epochs.
  total_steps = len(train_dataloader) * epochs

  # Create the learning rate scheduler.
  scheduler = get_linear_schedule_with_warmup(optimizer, 
                                              num_warmup_steps = 0, # Default value in run_glue.py
                                              num_training_steps = total_steps)
  return scheduler


def flat_accuracy(preds, labels):
    # Function to calculate the accuracy of our predictions vs labels
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

## 4. predictions
def get_predictions(prediction_dataloader, model, device):
  # Tracking variables 
  predictions , true_labels = [], []

  # Predict 
  for batch in prediction_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
  
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
  
    # Telling the model not to compute or store gradients, saving memory and 
    # speeding up prediction
    with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids, #token_type_ids=None, 
                      attention_mask=b_input_mask)

    logits = outputs[0]

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()
  
    # Store predictions and true labels
    predictions.append(logits)
    true_labels.append(label_ids)
  
  return predictions, true_labels, outputs

