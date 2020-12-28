#!/usr/bin/env python3
import os
import logging
import argparse
import time
import pickle
import torch
import pandas as pd
import numpy as np
import datetime as dt
from tqdm import tqdm, trange

from keras.preprocessing.sequence import pad_sequences
from transformers import (BertTokenizer, DistilBertTokenizer, BertConfig, DistilBertConfig,
                          BertForSequenceClassification, DistilBertForSequenceClassification,
                          AdamW, get_linear_schedule_with_warmup)
from sklearn.metrics import f1_score, confusion_matrix

from utils import set_all_seeds, initialize_device_settings, format_time
from data_handler import split_train_dev, create_dataloader


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


def train(train_dataloader, model, device, optimizer, scheduler, 
                        max_grad_norm=1.0):
    model.train()
    # Reset the total loss for this epoch.
    tr_loss = 0
    loss_values = []

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Always clear any previously calculated gradients before performing a
        # backward pass. 
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        outputs = model(b_input_ids, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)
        
        # The call to `model` always returns a tuple, so we need to pull the 
        # loss value out of the tuple.
        loss = outputs[0]

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        tr_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = tr_loss / len(train_dataloader)            
    
    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    return model, optimizer, scheduler, loss_values


def eval(sample_dataloader, model, device):
    # After the completion of each training epoch, measure our performance on
    # our validation set.
    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation. Important for reproducible results
    model.eval()
    # Tracking variables 
    pred_labels, true_labels = [], []

    # Predict 
    for batch in sample_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
  
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
  
        # Telling the model not to compute or store gradients, saving memory and 
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids,
                            attention_mask=b_input_mask)

        logits = outputs[0]

        # Move logits and labels to CPU
        label_ids = b_labels.to('cpu').numpy()
        logits = logits.detach().cpu().numpy()
  
        # Store predictions and true labels
        true_labels.append(label_ids)
        pred_labels.append(logits)
    
    true_labels = [item for sublist in true_labels for item in sublist]
    pred_labels = [item for sublist in pred_labels for item in sublist]
    
    true_bools = np.asarray(true_labels)
    pred_bools = np.argmax(pred_labels, axis=1).flatten()

    val_f1_accuracy = f1_score(true_bools, pred_bools, average = "micro")

    return true_bools, pred_bools, val_f1_accuracy


def main():
    """
    main function for conducting Subtask A. Parameters are parsed with argparse.
    Language model should be suitable for German e.g.:
        'bert-base-multilingual-uncased', 
        'bert-base-multilingual-cased',              
        'bert-base-german-cased', 
        'bert-base-german-dbmdz-cased',
        'bert-base-german-dbmdz-uncased',
        'distilbert-base-german-cased',
        'distilbert-base-multilingual-cased'.
    """

    ############################ variable settings #################################
    parser = argparse.ArgumentParser(description='Run Subtask A or B of GermEval 2017 Using Pre-Trained Language Model.')
    parser.add_argument('--task', type=str, default='A', help="The task you want to conduct ('A' or 'B').")
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--lang_model', type=str, default='bert-base-german-dbmdz-uncased', help='The pre-trained language model.')
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=5e-5, help='The learning rate.')
    parser.add_argument('--max_len', type=int, default=256, help='The maximum sequence length of the input text.')
    parser.add_argument('--batch_size', type=int, default=32, help='Your train set batch size.')
    parser.add_argument('--df_path', type=str, default='./data/', help='The data directory.')    
    parser.add_argument('--train_data', type=str, default='train_df.tsv', help='The filename of the input train data.')
    parser.add_argument('--dev_data', type=str, default='dev_df.tsv', help='The filename of the input development data.')
    parser.add_argument('--test_data1', type=str, default='test_syn_df.tsv', help='The filename of the first input test data (synchronic).')
    parser.add_argument('--test_data2', type=str, default='test_dia_df.tsv', help='The filename of the second input test data (diachronic).')
    parser.add_argument('--output_path', type=str, default='./output/subtaskA/', help='The output directory of the model and predictions.')
    parser.add_argument("--train", default=True, action="store_true", help="Flag for training.")
    parser.add_argument("--save_prediction", default=False, action="store_true", help="Flag for saving predictions.")
    args = parser.parse_args()

    ################################################################################
    set_all_seeds(args.seed)
    device, n_gpu = initialize_device_settings(use_cuda=True)

    # Load data
    train_df = pd.read_csv(args.df_path + args.train_data, delimiter = '\t')
    dev_df = pd.read_csv(args.df_path + args.dev_data, delimiter = '\t')
    test_syn_df = pd.read_csv(args.df_path + args.test_data1, delimiter = '\t')
    test_syn_df = test_syn_df.dropna(subset = ["text"])    
    test_dia_df = pd.read_csv(args.df_path + args.test_data2, delimiter = '\t')
    
    # 1. Create a tokenizer
    lower_case = False
    if args.lang_model[-7:] == "uncased":
        lower_case = True

    if args.lang_model[:4] == "bert":
        model_class = "BERT"
        tokenizer = BertTokenizer.from_pretrained(args.lang_model, do_lower_case=lower_case, max_length=args.max_len)
    
    if args.lang_model[:10] == "distilbert":
        model_class = "DistilBERT"
        tokenizer = DistilBertTokenizer.from_pretrained(args.lang_model, do_lower_case=lower_case, max_length=args.max_len)
    
    # get training features
    df = pd.concat([train_df, dev_df])
    sentences = df.text.values
    sentences_syn = test_syn_df.text.values    
    sentences_dia = test_dia_df.text.values
    
    if args.task == 'A':
        class_list = [False, True]
        df['relevance_label'] = df.apply(lambda x:  class_list.index(x['relevance']), axis = 1)
        labels = df.relevance_label.values
        test_syn_df['relevance_label'] = test_syn_df.apply(lambda x:  class_list.index(x['relevance']), axis = 1)
        labels_syn = test_syn_df.relevance_label.values
        test_dia_df['relevance_label'] = test_dia_df.apply(lambda x:  class_list.index(x['relevance']), axis = 1)
        labels_dia = test_dia_df.relevance_label.values

    if args.task == 'B':
        class_list = ["negative", "neutral", "positive"]
        df['sentiment_label'] = df.apply(lambda x:  class_list.index(x['sentiment']), axis = 1)
        labels = df.sentiment_label.values
        test_syn_df['sentiment_label'] = test_syn_df.apply(lambda x:  class_list.index(x['sentiment']), axis = 1)
        labels_syn = test_syn_df.sentiment_label.values
        test_dia_df['sentiment_label'] = test_dia_df.apply(lambda x:  class_list.index(x['sentiment']), axis = 1)
        labels_dia = test_dia_df.sentiment_label.values
    
    num_labels = len(set(labels))
    
    # Tokenize all of the sentences and map the tokens to their word IDs.
    input_ids = [tokenizer.encode(sent, add_special_tokens=True, truncation=True, 
                                  max_length=args.max_len) for sent in sentences]
    input_ids = pad_sequences(input_ids, maxlen=args.max_len, dtype="long", 
                          value=0.0, truncating="post", padding="post")
    # Create attention masks
    attention_masks = [[int(token_id > 0) for token_id in sent] for sent in input_ids]

    # synchronic test data
    input_ids_syn = [tokenizer.encode(sent, add_special_tokens=True, truncation=True) for sent in sentences_syn]
    input_ids_syn = pad_sequences(input_ids_syn, maxlen=args.max_len, dtype="long", 
                          value=0.0, truncating="post", padding="post")
    attention_masks_syn = [[int(token_id > 0) for token_id in sent] for sent in input_ids_syn]
    
    # diachronic test data
    input_ids_dia = [tokenizer.encode(sent, add_special_tokens=True, truncation=True) for sent in sentences_dia]
    input_ids_dia = pad_sequences(input_ids_dia, maxlen=args.max_len, dtype="long", 
                          value=0.0, truncating="post", padding="post")
    attention_masks_dia = [[int(token_id > 0) for token_id in sent] for sent in input_ids_dia]

    # split train, dev
    train_inputs, train_labels, dev_inputs, dev_labels, train_masks, dev_masks = split_train_dev(
        train_df, dev_df, attention_masks, input_ids, labels)

    # transform to torch tensor
    train_inputs = torch.tensor(train_inputs)
    dev_inputs = torch.tensor(dev_inputs)

    train_labels = torch.tensor(train_labels)
    dev_labels = torch.tensor(dev_labels)

    train_masks = torch.tensor(train_masks)
    dev_masks = torch.tensor(dev_masks)

    test_syn_inputs = torch.tensor(input_ids_syn)
    test_syn_labels = torch.tensor(labels_syn)
    test_syn_masks = torch.tensor(attention_masks_syn)

    test_dia_inputs = torch.tensor(input_ids_dia)
    test_dia_labels = torch.tensor(labels_dia)
    test_dia_masks = torch.tensor(attention_masks_dia)

    # Create the DataLoader
    train_dataloader = create_dataloader(train_inputs, train_masks, 
                                     train_labels, args.batch_size, train=True)

    dev_dataloader = create_dataloader(dev_inputs, dev_masks, 
                                   dev_labels, args.batch_size, train=False)

    test_syn_dataloader = create_dataloader(test_syn_inputs, test_syn_masks, 
                                        test_syn_labels, args.batch_size, 
                                        train=False)

    test_dia_dataloader = create_dataloader(test_dia_inputs, test_dia_masks, 
                                        test_dia_labels, args.batch_size, 
                                        train=False)

    # 4. Create model
    if args.train:
        if model_class == "BERT":
            config = BertConfig.from_pretrained(args.lang_model, num_labels=num_labels)   
            config.hidden_dropout_prob = 0.1
            model = BertForSequenceClassification.from_pretrained(
                args.lang_model,
                num_labels = num_labels,
                output_attentions = False,
                output_hidden_states = False
            )

        if model_class == "DistilBERT":
            config = DistilBertConfig.from_pretrained(args.lang_model, num_labels=num_labels)   
            config.hidden_dropout_prob = 0.1 
            model = DistilBertForSequenceClassification.from_pretrained(
                args.lang_model,
                num_labels = num_labels,
                output_attentions = False,
                output_hidden_states = False
            )
        model.cuda()


        # Create an optimizer
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                'weight_decay_rate': 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.lr,
            eps=1e-8
        )

        # Total number of training steps = number of batches * number of epochs
        total_steps = len(train_dataloader) * args.epochs
        # Create the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
    
        # train model
        # Main Loop
        print("=================== Train ================")
        print("##### Language Model:", args.lang_model, ",", "learning rate:", args.lr)
        print()

        track_time = time.time()
        # trange is a tqdm wrapper around the normal python range
        for epoch in trange(args.epochs, desc="Epoch"):
            print("Epoch: %4i"%epoch, dt.datetime.now())

            model, optimizer, scheduler, tr_loss = train(
                train_dataloader, 
                model=model, 
                device=device, 
                optimizer=optimizer, 
                scheduler=scheduler
            )
            # EVALUATION: TRAIN SET
            true_bools_train, pred_bools_train, f1_train = eval(
                train_dataloader, model=model, device=device)
            print("TRAIN: micro F1 %.4f"%(f1_train)) # here: same as accuracy
            print(confusion_matrix(true_bools_train,pred_bools_train))
            
            # EVALUATION: DEV SET
            true_bools_dev, pred_bools_dev, f1_dev = eval(
                dev_dataloader, model=model, device=device)
            print("EVAL: micro F1 %.4f"%(f1_dev))
            print(confusion_matrix(true_bools_dev,pred_bools_dev))
        

        print("  Training and validation took in total: {:}".format(format_time(time.time()-track_time)))

        # EVALUATION: TEST SYN SET
        true_bools_syn, pred_bools_syn, f1_test_syn = eval(
            test_syn_dataloader, model=model, device=device)
        print("TEST SYN: micro F1 %.4f"%(f1_test_syn))
        print(confusion_matrix(true_bools_syn,pred_bools_syn))

        # EVALUATION: TEST DIA SET
        true_bools_dia, pred_bools_dia, f1_test_dia = eval(
            test_dia_dataloader, model=model, device=device)
        print("TEST DIA: micro F1 %.4f"%(f1_test_dia))
        print(confusion_matrix(true_bools_dia, pred_bools_dia))

        if args.save_prediction:
            if args.task == 'A':
                test_syn_df["relevance_pred"] = flat_predictions_syn
                test_dia_df["relevance_pred"] = flat_predictions_dia
            if args.task == 'B':                
                test_syn_df["sentiment_pred"] = flat_predictions_syn
                test_dia_df["sentiment_pred"] = flat_predictions_dia
            
            test_syn_df.to_csv(args.output_path+args.lang_model+"_eval_test_syn.tsv", sep="\t", index = False, 
                header = True, encoding = "utf-8-sig")
            test_dia_df.to_csv(save_dir+args.lang_model+"_eval_test_dia.tsv", sep="\t", index = False, 
                header = True, encoding = "utf-8-sig")


if __name__ == "__main__":
    main()