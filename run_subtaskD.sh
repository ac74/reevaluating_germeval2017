#example run for training and evaluating our best models
#!/usr/bin/sh
python3 src/run_subtaskD.py \
    --pretrained_weights="bert-base-german-dbmdz-uncased" \
    --max_sequence_length=512 \
    --batch_size=16 \
    --crf \
    --save_prediction_test_syn \
    --save_prediction_test_dia \