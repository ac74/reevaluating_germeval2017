#!/usr/bin/env python3
import pandas as pd
import numpy as np
from itertools import chain
from torch.utils.data import (DataLoader, TensorDataset, 
                              RandomSampler, SequentialSampler)

from utils import flatten_list
from data_prep import prep_df, bio_tagging_df


def get_tags_list(df_path):
    """
    get list of BIO tags.

    Arg:
      df_path: data path 
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
    labels = full_df.bio_tags.values
    labels_unlist = [list(chain.from_iterable(lab)) for lab in labels]
    labels_flat = [flatten_list(lab) for lab in labels_unlist]

    # create tags
    tag_values = [list(set(tag)) for tag in labels_flat]
    tag_values = list(set(flatten_list(tag_values)))
    tag_values.append('PAD')
    tag2idx = {t: i for i, t in enumerate(tag_values)}

    return tag_values, tag2idx, entities


def tokenize_and_preserve_labels(tokenizer, sentence, text_labels, max_len):
    ''' tokenize and preserve labels for token classification (Subtask D) '''
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
    
    tokenizer.encode(tokenized_sentence, add_special_tokens=True, truncation=True, max_length=max_len)
    return tokenized_sentence, labels

def get_sentences_biotags(tokenizer, sentences, labels, max_len):
    '''
    get tokenized flattened sentences and BIO tags.

    Args:
      sentences: text column from data
      labels: label column from data
      max_len: maximal sequence length
    '''
    
    sentences_unlist = [list(chain.from_iterable(sent)) for sent in sentences]
    labels_unlist = [list(chain.from_iterable(lab)) for lab in labels]    
    sentences_flat = [flatten_list(sent) for sent in sentences_unlist]
    labels_flat = [flatten_list(lab) for lab in labels_unlist]
    
    tokenized_texts_and_labels = [tokenize_and_preserve_labels(tokenizer, sent, labs, max_len) for sent, labs in zip(sentences_flat, labels_flat)]
    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    tokenized_labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

    return tokenized_texts, tokenized_labels


def split_train_dev(train_df, dev_df, attention_masks, input_ids, labels):
    ''' train/dev split. '''
    attention_masks = np.array(attention_masks)
    train_inputs = input_ids[:len(train_df)]
    dev_inputs = input_ids[len(train_df):]
    train_labels = labels[:len(train_df)]
    dev_labels = labels[len(train_df):]
    train_masks = attention_masks[:len(train_df)]
    dev_masks = attention_masks[len(train_df):]
    return train_inputs, train_labels, dev_inputs, dev_labels, train_masks, dev_masks


def create_dataloader(df_inputs, df_masks, df_labels, batch_size, train=True):
    ''' create the DataLoader.'''
    data = TensorDataset(df_inputs, df_masks, df_labels)
    if train: # Create the DataLoader for training set
        sampler = RandomSampler(data)
    else: # Create the DataLoader for validation set
        sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader