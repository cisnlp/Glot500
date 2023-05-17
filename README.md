# Glot500: Scaling Multilingual Corpora and Language Models to 500+ Languages

[**Model**](https://huggingface.co/cis-lmu/Glot500) |
[**Paper**]()

# Introduction

This repository contains information about Glot500, code for Glot2000-c and Glot500-m, and implementations of Glot500-m evaluation.

# Prerequisites

We use two settings due to package conflict:

- Major: Python 3.9, `requirements.txt`
- Sentence retrieval: Python 3.6, `evaluation/retrieval/requirements.txt`

# Glot2000-c

## Data collection

## Data cleaning

## Data preparation

For training both tokenizer and model of Glot500-m, we need to prepare a **balanced** corpus covering all languages.

Go to 'preprocessing/' and run:

```
bash merge_files.sh
```

We set `--scale 1` for tokenizer, `--scale 30` for model.

# Glot500-m

## Vocabulary Extension

## Continued Pretraining

```
bash train_bash.sh
```

# Evaluation

## Perplexity

## Sentence Retrieval

## Sequence Labeling

## Round Trip Alignment

## Text Classification

# Citation

# Acknowledgements

This repository is built on top of [transformers](https://github.com/huggingface/transformers) and [xtreme](https://github.com/google-research/xtreme).

