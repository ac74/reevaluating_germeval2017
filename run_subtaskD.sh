#!/bin/bash
python3 src/run_subtaskD.py \
    --lang_model="bert-base-german-cased"

python3 src/run_subtaskD.py \
    --lang_model="bert-base-german-dbmdz-cased"

python3 src/run_subtaskD.py \
    --lang_model="bert-base-german-dbmdz-uncased"

python3 src/run_subtaskD.py \
    --lang_model="bert-base-multilingual-cased"

python3 src/run_subtaskD.py \
    --lang_model="bert-base-multilingual-uncased"

python3 src/run_subtaskD.py \
    --lang_model="distilbert-base-german-cased"

python3 src/run_subtaskD.py \
    --lang_model="distilbert-base-multilingual-cased"


python3 src/run_subtaskD.py \
    --lang_model="bert-base-german-cased" \
    --max_len=512 \
    --batch_size=16

python3 src/run_subtaskD.py \
    --lang_model="bert-base-german-dbmdz-cased" \
    --max_len=512 \
    --batch_size=16

python3 src/run_subtaskD.py \
    --lang_model="bert-base-german-dbmdz-uncased" \
    --max_len=512 \
    --batch_size=16

python3 src/run_subtaskD.py \
    --lang_model="distilbert-base-german-cased" \
    --max_len=512 \
    --batch_size=16

python3 src/run_subtaskD.py \
    --lang_model="distilbert-base-multilingual-cased" \
    --max_len=512 \
    --batch_size=16


python3 src/run_subtaskD.py \
    --lang_model="bert-base-german-cased" \
    --max_len=512 \
    --batch_size=8

python3 src/run_subtaskD.py \
    --lang_model="bert-base-german-dbmdz-cased" \
    --max_len=512 \
    --batch_size=8

python3 src/run_subtaskD.py \
    --lang_model="bert-base-german-dbmdz-uncased" \
    --max_len=512 \
    --batch_size=8

python3 src/run_subtaskD.py \
    --lang_model="bert-base-multilingual-cased" \
    --max_len=512 \
    --batch_size=8

python3 src/run_subtaskD.py \
    --lang_model="bert-base-multilingual-uncased" \
    --max_len=512 \
    --batch_size=8

python3 src/run_subtaskD.py \
    --lang_model="distilbert-base-german-cased" \
    --max_len=512 \
    --batch_size=8
    
python3 src/run_subtaskD.py \
    --lang_model="distilbert-base-multilingual-cased" \
    --max_len=512 \
    --batch_size=8