#!/bin/bash
python3 src/run_subtaskD.py \
    --lang_model="bert-base-german-dbmdz-uncased" \
    --batch_size=16 \
    --epochs=1

python3 src/run_subtaskD.py \
    --lang_model="bert-base-german-dbmdz-uncased" \
    --batch_size=16 \
    --epochs=1