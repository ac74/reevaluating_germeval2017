# reevaluating_germeval2017
Accompanying repository for [research paper](https://arxiv.org/abs/2102.12330) "Re-Evaluating GermEval17 Using German Pre-Trained Language Models" from February 24, 2021 and [master's thesis](https://epub.ub.uni-muenchen.de/75190/) "Re-Evaluating GermEval 2017: Document-Level and Aspect-Based Sentiment Analysis Using Pre-Trained Language Models" from January 4, 2021.

Contact persons: 
- Alessandra Corvonato, [A.Corvonato@campus.lmu.de](mailto:A.Corvonato@campus.lmu.de)
- Matthias Aßenmacher, [matthias@stat.uni-muenchen.de](mailto:matthias@stat.uni-muenchen.de)
- Christian Heumann, [chris@stat.uni-muenchen.de](mailto:chris@stat.uni-muenchen.de)

## Description
Re-Evaluation of [GermEval 2017](https://sites.google.com/view/germeval2017-absa/home) using following BERT and DistilBERT models from Hugging Face's [transformers](https://huggingface.co/transformers/):
- bert-base-german-cased,
- bert-base-german-dbmdz-cased,
- bert-base-german-dbmdz-uncased,
- bert-base-multilingual-cased,
- bert-base-multilingual-uncased,
- distilbert-base-german-cased,
- distilbert-base-multilingual-cased.

Our implementation is based on the [tutorial by Chris McCormick](https://colab.research.google.com/drive/1Y4o3jh3ZH70tl6mCd76vz_IxX23biCPP). For Subtask D, we also took a CRF layer into account (BERT-CRF / DistilBERT-CRF), based on [this](https://github.com/trtm/AURC) repository.

### Main Results
- all of the models outperformed the participants' results from 2017
- bert-base-german-dbmdz-uncased performs best on all four subtasks (with use of CRF on Subtask D)

## Setup
```bash
pip install -r requirements.txt
```
The code was run using Python 3.8.7 64-bit.

## Data
The organizers provide
- a train dataset,
- a development dataset,
- a synchronic test dataset,
- a diachronic test dataset.

We used the latest versions of the datasets (2017-09-15) in XML format which can be downloaded [here](http://ltdata1.informatik.uni-hamburg.de/germeval2017/). We evaluated the models on both test datasets.

### Data preparation
```bash
python3 src/data_prep.py
```
Make sure to put the data provided by the organizers to a folder "data". When running `data_prep.py`, the pre-processed subtask-specific data will also be saved in "data".

## Example run
```bash
sh run_example.sh
```

Note: `run_SubtaskD.py` should return reproducible results, but for some reason, it does not. Therefore, the micro F1 score may fluctuate between +/-0.01. This may slightly change the ranking of the language models for Subtask D. 

## Citation

```
@article{assenmacher2021re,
  title={Re-Evaluating GermEval17 Using German Pre-Trained Language Models},
  author={A{\ss}enmacher, Matthias and Corvonato, Alessandra and Heumann, Christian},
  journal={arXiv preprint arXiv:2102.12330},
  year={2021}
}
```
