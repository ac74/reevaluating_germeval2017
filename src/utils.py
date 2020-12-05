import time
import datetime
import random
import os
#import sys
import logging
#import json
import torch
import tensorflow as tf
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def set_all_seeds(seed=42, deterministic_cudnn=True):
    """
    Setting multiple seeds to make runs reproducible.
    Important: Enabling `deterministic_cudnn` gives full reproducibility with CUDA, but might slow down training
    :param seed:number to use as seed
    :type deterministic_cudnn: bool
    :return: None
    """    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG']= ":4096:8" #oder ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_deterministic(True)

def initialize_device_settings(use_cuda=True, local_rank=-1, use_amp=None):
    if not use_cuda:
        device = torch.device("cpu")
        n_gpu = 0
    elif local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            n_gpu = 0
        else:
            n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, automatic mixed precision training: {}".format(
            device, n_gpu, bool(local_rank != -1), use_amp
        )
    )
    return device, n_gpu

#def data_loading(df_path, df_filename):
  # load data
  #df = pd.read_csv(df_path + df_filename, delimiter = '\t')

  # df['text'] = df['text'].str.replace('\r', "")
  #return df

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def combine_labels(labs, pred = True):
  labs = [item for sublist in labs for item in sublist]
  if pred:
    labs = np.argmax(labs, axis=1).flatten()
  return labs

def flatten_list(lists):
  final = []
  for i in lists:
    if type(i) is list:
      for j in i:
        final.append(j)
    else:
      final.append(i)
  return final