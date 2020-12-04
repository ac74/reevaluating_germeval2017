import logging
from pathlib import Path
import pickle

import torch

# install torchcrf with:
# sudo -H pip3 install git+https://github.com/kmkurn/pytorch-crf#egg=pytorch_crf
from torchcrf import CRF

from data_handler import *
from modeling import *
from train import train_validate_model_token
from utils import * 
from test import *

from itertools import chain
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from seqeval_labeling import *

from modeling_CRF import TokenBERT

logger = logging.getLogger(__name__)
set_all_seeds()

def main(train_df, dev_df, test_syn_df = None, test_dia_df = None, lang_model = "bert-base-german-dbmdz-uncased", use_crf = False):
    # main function for subtask D
    # Args:
    # train_df: train dataframe (tsv format)
    # dev_df: development dataframe (tsv format) for validation
    # test_syn_df: synchronic test data (tsv format)
    # test_dia_df: diachronic test data (tsv format)
    # data sets sholud contain following variables
    #   - text: text data
    #   - 
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    #ml_logger = MLFlowLogger(tracking_uri="https://public-mlflow.deepset.ai/")
    #ml_logger.init_experiment(experiment_name="Public_FARM", run_name="Run_subtaskD")

    ############################ variable settings #################################
    seed = 42
    epochs = 1 # 4
    batch_size = 16
    max_len = 512
    lr = 5e-5 # try also 2e-5, 1e-5?
    eps = 1e-8
    weight_decay = 0.01
    max_grad_norm = 1.0
    ################################################################################
    
    device, n_gpu = initialize_device_settings(use_cuda=True)
    df_path = "/home/ubuntu/masterthesis_germeval2017/data/"
    
    # 1. Create a tokenizer
    do_lower_case = False
    if lang_model[-7:] == "uncased":
        do_lower_case = True

    if lang_model[:4] == "bert":
        model_class = "BERT"
        tokenizer = BertTokenizer.from_pretrained(lang_model, do_lower_case = do_lower_case, max_length = max_len)
    
    if lang_model[:10] == "distilbert":
        model_class = "DistilBERT"
        tokenizer = DistilBertTokenizer.from_pretrained(lang_model, do_lower_case = do_lower_case, max_length=max_len)

    # get training features
    tokenized_texts, labels = get_sentences_biotags(tokenizer, train_df, dev_df)
    print(len(tokenized_texts))
    print(len(labels))
    
    # get tag values and dictionary
    tag_values, tag2idx, entities = get_tags_list(df_path)
    
    # pad input_ids and tags, create attention masks
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen = max_len, dtype="long", value=0.0,
                          truncating="post", padding="post")
    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=max_len, value=tag2idx["PAD"], padding="post",
                     dtype="long", truncating="post")
    
    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

    # split train, dev
    train_inputs, train_labels, dev_inputs, dev_labels, train_masks, dev_masks = split_train_dev(
        train_df, dev_df, attention_masks, input_ids, tags)
    print(len(train_inputs))
    print(len(train_labels))
    print(len(dev_inputs))
    print(len(dev_labels))
    print(train_inputs[:5])
    print(dev_labels[:5])

    # transform to torch tensor
    train_inputs = torch.tensor(train_inputs)
    dev_inputs = torch.tensor(dev_inputs)

    train_labels = torch.tensor(train_labels)
    dev_labels = torch.tensor(dev_labels)

    train_masks = torch.tensor(train_masks)
    dev_masks = torch.tensor(dev_masks)
    # create DataLoader
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, num_workers = 0)

    dev_data = TensorDataset(dev_inputs, dev_masks, dev_labels)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size, num_workers = 0)

    # 4. Create model
    if model_class == "BERT":
        model = BertForTokenClassification.from_pretrained(
            lang_model,
            num_labels = len(tag2idx),
            output_attentions = False,
            output_hidden_states = False
        )

    if model_class == "DistilBERT":
        model = DistilBertForTokenClassification.from_pretrained(
            lang_model,
            num_labels = len(tag2idx),
            output_attentions = False,
            output_hidden_states = False
        )

    if use_crf:
        model = TokenBERT(
            model_name=lang_model, 
            num_labels=len(tag2idx), 
            output_hidden_states=False, 
            output_attentions = False,
            use_crf=use_crf)
    else:
        crf = None
    
    model.cuda()

    # 5. Create an optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW( # Lee et al. hat bessere Ergebnisse mit adadelta erzielt...
        optimizer_grouped_parameters,
        lr=lr,
        eps=eps
    )
    # Total number of training steps = number of batches * number of epochs
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    #model, optimizer, lr_schedule = initialize_optimizer(
     #   model=model,
      #  learning_rate=1e-5, # or 5e-5
       # n_batches=len(data_silo.loaders["train"]),
        #n_epochs=n_epochs,
        #device=device,
    #)
    
    # 6. train model
    track_time = time.time()
    
    set_all_seeds(seed)
    outputs, pred_tags, dev_tags, eval_loss = train_validate_model_token(
        model = model,
        epochs = epochs,
        train_dataloader = train_dataloader,
        dev_dataloader = dev_dataloader,
        optimizer = optimizer,
        scheduler = scheduler,
        device = device,
        tag_values = tag_values,
        crf = crf
    )
    print("  Training and validation took in total: {:}".format(format_time(time.time()-track_time)))

    print("Classification Report for Subtask D (exact): ")
    print(seq_classification_report(pred_tags, dev_tags, digits = 3))
    print("Classification Report for Subtask D (overlap): ")
    print(seq_classification_report(pred_tags, dev_tags, digits = 3,overlap = True))

    # 8. Store it
    save_dir = "saved_models/subtaskD/"
    #model.save(save_dir)
    #processor.save(save_dir)
    #torch.save(model.state_dict(), save_dir+"bert_model_D")

    # 9. test it
    tokenized_texts_syn, labels_syn = get_sentences_biotags(tokenizer, test_syn_df)
    tokenized_texts_dia, labels_dia = get_sentences_biotags(tokenizer, test_dia_df)
    
    # 9.1 synchronic test data
    # padding
    input_ids_syn = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts_syn],
                          maxlen = max_len, dtype="long", value=0.0,
                          truncating="post", padding="post")

    tags_syn = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels_syn], 
                         maxlen=max_len, value=tag2idx["PAD"], padding="post", 
                         dtype="long", truncating="post")

    # Create attention masks
    attention_masks_syn = [[float(i != 0.0) for i in ii] for ii in input_ids_syn]

    # Convert to tensors
    test_syn_inputs = torch.tensor(input_ids_syn)
    test_syn_masks = torch.tensor(attention_masks_syn)
    test_syn_labels = torch.tensor(tags_syn)
    # Create the DataLoader
    test_syn_data = TensorDataset(test_syn_inputs, test_syn_masks, test_syn_labels)
    test_syn_sampler = SequentialSampler(test_syn_data)
    test_syn_dataloader = DataLoader(test_syn_data, sampler=test_syn_sampler, batch_size=batch_size, num_workers = 0)
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    true_labels_syn, pred_labels_syn = test_biotagging(
        test_dataloader = test_syn_dataloader, model = model, device = device)
    
    pred_tags_syn = [tag_values[p_i] for p, l in zip(pred_labels_syn, true_labels_syn) for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    true_tags_syn = [tag_values[l_i] for l in true_labels_syn for l_i in l if tag_values[l_i] != "PAD"]

    # Print and save classification report
    print('Test Syn F1 Accuracy: ', seq_f1_score(true_tags_syn, pred_tags_syn, average = "micro"))
    print('Test Syn Flat Accuracy: ', seq_accuracy_score(true_tags_syn, pred_tags_syn),'\n') # beachtet auch O tags, deshalb so hoch
    cr_report_syn = seq_classification_report(true_tags_syn, pred_tags_syn, digits = 3)
    print(cr_report_syn)
    #pickle.dump(cr_report_syn, open(save_dir+'classification_report_'+lang_model+'_test_syn_exact.txt','wb'))

    ############ overlap evaluation ################
    print('Test Syn F1 Accuracy: ', seq_f1_score(true_tags_syn, pred_tags_syn, average = "micro", overlap = True))
    #print('Test Syn Flat Accuracy: ', seq_accuracy_score(true_tags_syn, pred_tags_syn),'\n')
    cr_report_syn_overlap = seq_classification_report(true_tags_syn, pred_tags_syn, digits = 3, overlap = True)
    print(cr_report_syn_overlap)
    #pickle.dump(cr_report_syn_overlap, open(save_dir+'classification_report_'+lang_model+'_test_syn_overlap.txt','wb'))
    ################################################
    
    # 9.2 diachronic test data
    input_ids_dia = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts_dia],
                          maxlen = max_len, dtype="long", value=0.0,
                          truncating="post", padding="post")

    tags_dia = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels_dia], 
                         maxlen=max_len, value=tag2idx["PAD"], padding="post", 
                         dtype="long", truncating="post")

    # Create attention masks
    attention_masks_dia = [[float(i != 0.0) for i in ii] for ii in input_ids_dia]

    # Convert to tensors
    test_dia_inputs = torch.tensor(input_ids_dia)
    test_dia_masks = torch.tensor(attention_masks_dia)
    test_dia_labels = torch.tensor(tags_dia)
    # Create the DataLoader
    test_dia_data = TensorDataset(test_dia_inputs, test_dia_masks, test_dia_labels)
    test_dia_sampler = SequentialSampler(test_dia_data)
    test_dia_dataloader = DataLoader(test_dia_data, sampler=test_dia_sampler, batch_size=batch_size, num_workers = 0)
    # Put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    true_labels_dia, pred_labels_dia = test_biotagging(
        test_dataloader = test_dia_dataloader, model = model, device = device)
    pred_tags_dia = [tag_values[p_i] for p, l in zip(pred_labels_dia, true_labels_dia)
                             for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    true_tags_dia = [tag_values[l_i] for l in true_labels_dia for l_i in l if tag_values[l_i] != "PAD"]

    # Print and save classification report
    print('Test Dia F1 Accuracy: ', seq_f1_score(true_tags_dia, pred_tags_dia, average = "micro"))
    print('Test Dia Flat Accuracy: ', seq_accuracy_score(true_tags_dia, pred_tags_dia),'\n')
    cr_report_dia = seq_classification_report(true_tags_dia, pred_tags_dia, digits = 3)
    print(cr_report_dia)
    #pickle.dump(cr_report_dia, open(save_dir+'classification_report_'+lang_model+'_test_dia_exact.txt','wb')) #save report

    ############ overlap evaluation ################
    print('Test Dia F1 Accuracy: ', seq_f1_score(true_tags_dia, pred_tags_dia, average = "micro", overlap = True))
    #print('Test Flat Accuracy: ', seq_accuracy_score(true_tags_syn, pred_tags_syn),'\n')
    cr_report_dia_overlap = seq_classification_report(true_tags_dia, pred_tags_dia, digits = 3, overlap = True)
    print(cr_report_dia_overlap)
    #pickle.dump(cr_report_dia_overlap, open(save_dir+'classification_report_'+lang_model+'_test_dia_overlap.txt','wb'))

    # interpret it
    
    # return anything?


#if __name__ == "__main__":
    # load data
df_path = "/home/ubuntu/masterthesis_germeval2017/data/"
_, train_df, _, _, _ = sample_to_tsv(df_path, "train-2017-09-15.xml", bio_tagging=True)
_, dev_df, _, _, _ = sample_to_tsv(df_path, "dev-2017-09-15.xml", bio_tagging=True)
_, test_syn_df, _, _, _ = sample_to_tsv(df_path, "test_syn-2017-09-15.xml", bio_tagging=True)
_, test_dia_df, _, _, _ = sample_to_tsv(df_path, "test_dia-2017-09-15.xml", bio_tagging=True)

    #train_df = pd.read_csv(df_path+"train_df_opinion.tsv", delimiter="\t")
    #dev_df = pd.read_csv(df_path+"dev_df_opinion.tsv", delimiter="\t")

    # define models
lm_bert = ('distilbert-base-german-cased',
           'distilbert-base-multilingual-cased',
           'bert-base-german-cased', 
           'bert-base-multilingual-uncased', 
           'bert-base-multilingual-cased',
           'bert-base-german-dbmdz-cased',
           'bert-base-german-dbmdz-uncased')

    # run models
    #for lang_model in lm_bert:
    #print("===================Train: ", lang_model, " ================")
main(train_df, dev_df, test_syn_df, test_dia_df, 'distilbert-base-german-cased', use_crf = False)
    # check reproducibility
main(train_df, dev_df, test_syn_df, test_dia_df, 'distilbert-base-german-cased', use_crf = False)