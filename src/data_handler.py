import os
import time
import pandas as pd
import numpy as np
import torch
from torch.utils.data import (Dataset, DataLoader, TensorDataset, 
                              RandomSampler, SequentialSampler)
from transformers import (
    BertTokenizer,
    DistilBertTokenizer,
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    BertForTokenClassification,
    DistilBertForTokenClassification,
    BertConfig,
    DistilBertConfig,
    AdamW,
    get_linear_schedule_with_warmup
)
from keras.preprocessing.sequence import pad_sequences
from itertools import chain
from utils import flatten_list
from data_prep import prep_df, bio_tagging_df

def get_sentences_labels(df = None, train_df = None, dev_df = None, task = 'A'):
  """
  Get the lists of sentences and their labels.
  """
  if train_df is not None:
    if dev_df is not None:
      df = pd.concat([train_df, dev_df])
  
  sentences = df.text.values
  labels = []
  if task == 'A':
    class_list = ["false", "true"]
    df['relevance_label'] = df.apply(lambda x:  class_list.index(x['relevance']), axis = 1)
    labels = df.relevance_label.values

  if task == 'B':
    class_list = ["negative", "neutral", "positive"]    
    df['sentiment_label'] = df.apply(lambda x:  class_list.index(x['sentiment']), axis = 1)
    labels = df.sentiment_label.values

  if task == 'C':
    labels = list(df.one_hot_labels.values)

  if task == 'D':
    labels = df.bio_tags.values

  return sentences, labels

### for Subtask D ###
def get_tags_list(df_path):
    """
    get list of BIO tags
    """
    train_df = pd.read_csv(df_path + 'train_df_opinion.tsv', delimiter = '\t')
    dev_df = pd.read_csv(df_path + 'dev_df_opinion.tsv', delimiter = '\t')
    test_syn_df = pd.read_csv(df_path + "test_syn_df_opinion.tsv", delimiter = '\t')
    test_dia_df = pd.read_csv(df_path + "test_dia_df_opinion.tsv", delimiter = '\t')
    # concatenate data frames
    full_df = pd.concat([train_df, dev_df, test_syn_df, test_dia_df])

    # prepare labels
    _, entities = prep_df(full_df)
    full_df = bio_tagging_df(full_df)
    #print(full_df.head())
    _, labels = get_sentences_labels(df = full_df, task = "D")
    labels_unlist = [list(chain.from_iterable(lab)) for lab in labels]
    labels_flat = [flatten_list(lab) for lab in labels_unlist]

    # create tags
    tag_values = [list(set(tag)) for tag in labels_flat]
    tag_values = list(set(flatten_list(tag_values))) # 110 values
    tag_values.append('PAD') # 111
    tag2idx = {t: i for i, t in enumerate(tag_values)}
    #print("Tag values:", tag_values)
    #print("Tag dictionary:", tag2idx)

    return tag_values, tag2idx, entities


############## define functions for 
# - tokenizing
# - optimizing
# - modeling
# - predicting

## 1. tokenization (tokenizing, mapping, padding)
def tokenize_and_preserve_labels(tokenizer, sentence, text_labels, max_len):
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
    
    tokenizer.encode(tokenized_sentence, add_special_tokens=True, truncation = True, max_length=max_len)
    
    return tokenized_sentence, labels

def get_sentences_biotags(tokenizer, df, df2=None, max_len=512):
    if df2 is not None:
      sentences, labels = get_sentences_labels(train_df = df, 
                                             dev_df = df2, task = "D")
    else:
      sentences, labels = get_sentences_labels(df = df, task = "D")
    
    sentences_unlist = [list(chain.from_iterable(sent)) for sent in sentences]
    labels_unlist = [list(chain.from_iterable(lab)) for lab in labels]
    
    sentences_flat = [flatten_list(sent) for sent in sentences_unlist]
    labels_flat = [flatten_list(lab) for lab in labels_unlist]
    

    tokenized_texts_and_labels = [tokenize_and_preserve_labels(tokenizer, sent, labs, max_len) for sent, labs in zip(sentences_flat, labels_flat)]

    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

    print("tokenized_texts", tokenized_texts[0])

    return tokenized_texts, labels


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

def padding(input_ids, tokenizer, max_len):
  # We'll borrow the `pad_sequences` utility function to do this.
  print('\nPadding/truncating all sentences to %d values...' % max_len)
  print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))
  # Pad our input tokens with value 0.
  # "post" indicates that we want to pad and truncate at the end of the sequence,
  # as opposed to the beginning.
  input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", 
                          value=0, truncating="post", padding="post")
  return input_ids

def padding_token(tokenized_texts, tokenizer, max_len, tag2idx, labels):
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen = max_len, value=0.0, padding="post",
                          dtype="long", truncating="post")
    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=max_len, value=tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")
    return input_ids, tags

def masking(input_ids):

    # Create the attention mask.
    #   - If a token ID is 0, then it's padding, set the mask to 0.
    #   - If a token ID is > 0, then it's a real token, set the mask to 1.
    att_masks = [[int(token_id > 0) for token_id in sent] for sent in input_ids]

    return att_masks

def split_train_dev(train_df, dev_df, attention_masks, input_ids, labels):
    ''' train/dev split '''
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

def get_predictions_multilabel(test_dataloader, model, device):
  #track variables
  logit_preds, true_labels, pred_labels, tokenized_texts = [],[],[],[]

  # Predict
  for i, batch in enumerate(test_dataloader):
    batch = tuple(t.to(device) for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_labels = batch
    with torch.no_grad():
      # Forward pass
      outs = model(b_input_ids, attention_mask=b_input_mask)
      b_logit_pred = outs[0]
      pred_label = torch.sigmoid(b_logit_pred)

      b_logit_pred = b_logit_pred.detach().cpu().numpy()
      pred_label = pred_label.to('cpu').numpy()
      b_labels = b_labels.to('cpu').numpy()

    tokenized_texts.append(b_input_ids)
    logit_preds.append(b_logit_pred)
    true_labels.append(b_labels)
    pred_labels.append(pred_label)

  # Flatten outputs
  tokenized_texts = [item for sublist in tokenized_texts for item in sublist]
  pred_labels = [item for sublist in pred_labels for item in sublist]
  true_labels = [item for sublist in true_labels for item in sublist]
  # Converting flattened binary values to boolean values
  true_bools = [tl==1 for tl in true_labels]
  pred_bools = [pl>0.50 for pl in pred_labels] #boolean output after thresholding

  return true_labels, pred_labels, true_bools, pred_bools