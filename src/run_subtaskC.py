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

from torch.nn import BCEWithLogitsLoss, BCELoss
from keras.preprocessing.sequence import pad_sequences
from transformers import (BertTokenizer, DistilBertTokenizer, BertConfig, DistilBertConfig,
                          BertForSequenceClassification, DistilBertForSequenceClassification,
                          AdamW, get_linear_schedule_with_warmup)

from sklearn.metrics import (f1_score, accuracy_score, 
                            multilabel_confusion_matrix, classification_report)

from utils import set_all_seeds, initialize_device_settings, format_time
from data_handler import (split_train_dev, create_dataloader)                     


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


def train_multilabel(train_dataloader, model, device, optimizer, scheduler, 
                        num_labels, max_grad_norm=1.0):

    # Set our model to training mode (as opposed to evaluation mode)
    model.train()
    # Tracking variables
    tr_loss = 0 # running loss
    nb_tr_examples, nb_tr_steps = 0, 0
    train_loss_set = []
    # Train the data for one epoch
    for step, batch in enumerate(train_dataloader):
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Clear out the gradients (by default they accumulate)
        model.zero_grad()

        # Forward pass for multilabel classification(!)
        outputs = model(b_input_ids, attention_mask=b_input_mask)
        logits = outputs[0]
        loss_func = BCEWithLogitsLoss() # combined Sigmoid layer and binary cross entropy loss
        loss = loss_func(logits.view(-1, num_labels), 
                         b_labels.type_as(logits).view(-1, num_labels)) #convert labels to float for calculation
        train_loss_set.append(loss.item())    

        # Backward pass
        loss.backward()
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)      
        # Update parameters and take a step using the computed gradient
        optimizer.step()
        scheduler.step()
        # Update tracking variables
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

    print("Train loss: {}".format(tr_loss/nb_tr_steps))

    return model, optimizer, scheduler, tr_loss


def eval_multilabel(sample_dataloader, model, device):
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()
    # Variables to gather full output
    logit_preds, true_labels, pred_labels = [],[],[]

    # Predict
    for i, batch in enumerate(sample_dataloader):
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

      logit_preds.append(b_logit_pred)
      true_labels.append(b_labels)
      pred_labels.append(pred_label)

    # Flatten outputs
    pred_labels = [item for sublist in pred_labels for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]

    # Calculate Accuracy
    pred_bools = [pl>0.50 for pl in pred_labels]
    true_bools = [tl==1 for tl in true_labels]
    val_f1_accuracy = f1_score(true_bools, pred_bools, average='micro')

    return pred_bools, true_bools, val_f1_accuracy


def main():
    """
    main function for conducting Subtask C. Parameters are parsed with argparse.
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
    parser = argparse.ArgumentParser(description='Run Subtask C of GermEval 2017 Using Pre-Trained Language Model.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--lang_model', type=str, default='bert-base-german-dbmdz-uncased', help='The pre-trained language model.')
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs for training.')
    parser.add_argument('--lr', type=float, default=5e-5, help='The learning rate.')
    parser.add_argument('--max_len', type=int, default=256, help='The maximum sequence length of the input text.')
    parser.add_argument('--batch_size', type=int, default=32, help='Your train set batch size.')
    parser.add_argument('--df_path', type=str, default='./data/', help='The data directory.')    
    parser.add_argument('--train_data', type=str, default='train_df_cat.tsv', help='The filename of the input train data.')
    parser.add_argument('--dev_data', type=str, default='dev_df_cat.tsv', help='The filename of the input development data.')
    parser.add_argument('--test_data1', type=str, default='test_syn_df_cat.tsv', help='The filename of the first input test data (synchronic).')
    parser.add_argument('--test_data2', type=str, default='test_dia_df_cat.tsv', help='The filename of the second input test data (diachronic).')
    parser.add_argument('--output_path', type=str, default='./output/subtaskC/', help='The output directory of the model and predictions.')
    parser.add_argument("--train", default=True, action="store_true", help="Flag for training.")
    parser.add_argument("--save_prediction", default=False, action="store_true", help="Flag for saving predictions.")
    parser.add_argument("--save_cr", default=False, action="store_true", help="Flag for saving confusion matrix.")
    parser.add_argument("--exclude_general", default=False, action="store_true", help="Flag for excluding category Allgemein.")
    parser.add_argument("--exclude_neutral", default=False, action="store_true", help="Flag for excluding neutral polarity.")
    parser.add_argument("--exclude_general_neutral", default=False, action="store_true", help="Flag for excluding category Allgemein:neutral.")
    args = parser.parse_args()
    ################################################################################
    set_all_seeds(args.seed)
    device, n_gpu = initialize_device_settings(use_cuda=True)
    
    # Load data
    train_df = pd.read_csv(args.df_path + args.train_data, delimiter = '\t')
    dev_df = pd.read_csv(args.df_path + args.dev_data, delimiter = '\t')
    test_syn_df = pd.read_csv(args.df_path + args.test_data1, delimiter = '\t')
    test_dia_df = pd.read_csv(args.df_path + args.test_data2, delimiter = '\t')
    
    # Create a tokenizer
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
    cats = train_df.columns[5:]
    end = "full"
    # exclude categories if required
    if (args.exclude_general):
        cats = [i for i in list(cats) if "Allgemein" not in i]
        end = "excl_gen"
    if (args.exclude_neutral):
        cats = [i for i in list(cats) if "neutral" not in i]
        end = "excl_neu"
    if (args.exclude_general_neutral):
        cats = [i for i in list(cats) if "Allgemein:neutral" not in i]
        end = "excl_genneu"
    
    num_labels = len(list(cats))

    # create one hot labels
    train_df['one_hot_labels'] = list(train_df[list(cats)].values)
    dev_df['one_hot_labels'] = list(dev_df[list(cats)].values)
    test_syn_df['one_hot_labels'] = list(test_syn_df[list(cats)].values)
    test_dia_df['one_hot_labels'] = list(test_dia_df[list(cats)].values)

    # retrieve sentences and labels
    df = pd.concat([train_df, dev_df])
    sentences = df.text.values
    labels = list(df.one_hot_labels.values) 

    sentences_syn = test_syn_df.text.values
    labels_syn = list(test_syn_df.one_hot_labels.values)

    sentences_dia = test_dia_df.text.values
    labels_dia = list(test_dia_df.one_hot_labels.values)
        
    print("number of categories:", len(list(cats)))

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
    test_syn_masks = torch.tensor(attention_masks_syn)
    test_syn_labels = torch.tensor(labels_syn)

    test_dia_inputs = torch.tensor(input_ids_dia)
    test_dia_masks = torch.tensor(attention_masks_dia)
    test_dia_labels = torch.tensor(labels_dia)

    # Create the DataLoader
    train_dataloader = create_dataloader(train_inputs, train_masks, 
                                     train_labels, args.batch_size, train = True)

    dev_dataloader = create_dataloader(dev_inputs, dev_masks, 
                                   dev_labels, args.batch_size, train = False)

    test_syn_dataloader = create_dataloader(test_syn_inputs, test_syn_masks, 
                                        test_syn_labels, args.batch_size, 
                                        train = False)

    test_dia_dataloader = create_dataloader(test_dia_inputs, test_dia_masks, 
                                        test_dia_labels, args.batch_size, 
                                        train = False)

    # Create model
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
            eps = 1e-8
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

            model, optimizer, scheduler, tr_loss = train_multilabel(
                train_dataloader=train_dataloader, 
                model=model, 
                device=device, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                num_labels=num_labels
            )
            # EVALUATION: TRAIN SET
            pred_bools_train, true_bools_train, f1_train = eval_multilabel(
                train_dataloader, model=model, device=device)
            print("TRAIN: micro F1 %.3f"%(f1_train))
            
            # EVALUATION: DEV SET
            pred_bools_dev, true_bools_dev, f1_dev = eval_multilabel(
                dev_dataloader, model=model, device=device)
            print("EVAL: micro F1 %.3f"%(f1_dev))
        

        print("  Training and validation took in total: {:}".format(format_time(time.time()-track_time)))

        # EVALUATION: TEST SYN SET
        pred_bools_syn, true_bools_syn, f1_test_syn = eval_multilabel(
            test_syn_dataloader, model=model, device=device)
        print("TEST SYN: micro F1 %.4f"%(f1_test_syn))

        # classification report
        clf_report_syn = classification_report(true_bools_syn, pred_bools_syn, target_names=cats, digits=3)
        print(clf_report_syn)


        # EVALUATION: TEST DIA SET
        pred_bools_dia, true_bools_dia, f1_test_dia = eval_multilabel(
            test_dia_dataloader, model=model, device=device
        )
        print("TEST DIA: micro F1 %.4f"%(f1_test_dia))

        # classification report
        clf_report_dia = classification_report(true_bools_dia, pred_bools_dia, target_names=cats, digits=3)
        print(clf_report_dia)
        
        if args.save_cr:
            pickle.dump(clf_report_syn, open(args.output_path+'clf_report_'+args.lang_model+'_test_syn_'+str(num_labels)+end+'.txt','wb'))
            pickle.dump(clf_report_dia, open(args.output_path+'clf_report_'+args.lang_model+'_test_dia_'+str(num_labels)+end+'.txt','wb'))


        if args.save_prediction:
            test_syn_df["category_pred"] = pred_bools_syn
            test_dia_df["category_pred"] = pred_bools_dia
            test_syn_df.category_pred.to_csv(args.output_path+args.lang_model+'_test_syn_'+str(num_labels)+end+".tsv", 
            sep="\t", index = False, header = True, encoding = "utf-8-sig")
            test_dia_df.category_pred.to_csv(args.output_path+args.lang_model+'_test_dia_'+str(num_labels)+end+".tsv", 
            sep="\t", index = False, header = True, encoding = "utf-8-sig")
    

if __name__ == "__main__":
    set_all_seeds()
    main()