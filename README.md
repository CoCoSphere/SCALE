# SCALE: Semantic Causal ALignment for ECPEC

Official PyTorch implementation of **SCALE** (**S**emantic **C**ausal **AL**ignment for **E**CPEC), a unified framework for Emotion-Cause Pair Extraction in Conversations.

## 📖 Introduction

**Emotion-Cause Pair Extraction in Conversations (ECPEC)** aims to identify causal relations between emotion utterances and their triggering causes in a dialogue. Current methods often treat this as an independent pairwise classification task, which overlooks the distinct semantic roles of "emotion diffusion" and "cause explanation" and fails to capture many-to-many causalities.

**SCALE** addresses these limitations by:

**Semantic Decoupling**: Disentangling emotion-oriented and cause-oriented semantics into two complementary representation spaces.

**Global Alignment**: Formulating the task as a global alignment problem between emotion-side and cause-side representations.

**Optimal Transport**: Utilizing an optimal transport (OT) framework to enable globally consistent and many-to-many emotion-cause matching.


## 🛠️ Model Architecture

The SCALE framework consists of four main components:

**Feature Extraction**: Encodes utterances using a pretrained **RoBERTa** model to obtain contextual embeddings.


**Graph Construction**: Represents each conversation as a graph capturing global context, local temporal dynamics, and intra-speaker dependencies.


**Representation Learning**: Employs two independent graph encoders ( and ) to induce emotion-aware and cause-aware node representations.


**Graph Alignment**: Solves a global transport plan using the **Sinkhorn scheme** to find correspondences between the two semantic spaces.



### Data Preparation

We evaluate SCALE on three benchmark datasets:

* **RECCON-DD / RECCON-IE**
* **ECF (Emotion-Cause in Friends)**


## 📊 Experimental Results

SCALE achieves state-of-the-art performance across all major benchmarks:

| Dataset       | Precision | Recall | F1-score  |
| ------------- | --------- | ------ | --------- |
| **RECCON-DD** | 56.31     | 61.60  | **58.83** |
| **RECCON-IE** | 42.54     | 34.69  | **34.69** |
| **ECF**       | 55.01     | 60.67  | **57.70** |

Note: SCALE's soft optimal transport alignment favors higher recall by encouraging broader semantic matching.

## 📂 Project Structure

```text
.
├── models/          # SCALE core components (GNN, OT Solver)
├── data/            # Dataset directory (RECCON, ECF)
├── utils/           # Preprocessing and evaluation metrics
├── train.py         # Main training script
└── README.md

```


---

*This repository is for the anonymous submission to ACL.*