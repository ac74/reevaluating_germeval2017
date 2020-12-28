#!/bin/bash
python3 src/run_subtaskAB.py \
    --lang_model="bert-base-german-dbmdz-uncased" \
    --task="A"

python3 src/run_subtaskAB.py \
    --lang_model="bert-base-german-dbmdz-uncased" \
    --task="B"

python3 src/run_subtaskC.py \
    --lang_model="bert-base-german-dbmdz-uncased" \
    --train_data="train_df_cat_pol.tsv" \
    --dev_data="dev_df_cat_pol.tsv" \
    --test_data1="test_syn_df_cat_pol.tsv" \
    --test_data2="test_dia_df_cat_pol.tsv"

python3 src/run_subtaskC.py \
    --lang_model="bert-base-german-dbmdz-uncased" \

python3 src/run_subtaskD.py \
    --lang_model="bert-base-german-dbmdz-uncased" \
    --use_crf