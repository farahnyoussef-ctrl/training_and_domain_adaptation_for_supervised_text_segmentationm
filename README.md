# Text Segmentation on Wiki-727K with Adapter-based RoBERTa

A PyTorch model that finds **topic (section) boundaries** in English Wikipedia articles. For a window of 16 consecutive sentences, it predicts which sentences start a new section.

The architecture is inspired by **HITS**, the hierarchical, adapter-augmented segmentation model of Glavaš, Ganesh and Somasundaran (2021), *Training and Domain Adaptation for Supervised Text Segmentation*, Proceedings of the 16th Workshop on Innovative Use of NLP for Building Educational Applications (BEA), pages 110–116.

## Overview

| | |
|---|---|
| **Task** | Text segmentation (sentence-level boundary prediction) |
| **Dataset** | Wiki-727K ([Koshorek et al., 2018](https://arxiv.org/abs/1803.09337)), loaded from [Kaggle](https://www.kaggle.com/datasets/duongtran1909/wiki727k) |
| **Base model** | `roberta-base` (Hugging Face Transformers) |
| **Framework** | PyTorch |
| **Training** | Google Colab |

## Data preparation

1. Split every article at its section markers and take the sentences of each section.
2. Label each sentence: **1** if it is the first sentence of a section, **0** otherwise.
3. Use a subset of the training split: the first third of the training documents, then the first half of the resulting sentences.
4. Build overlapping windows of **16 sentences** (stride 8) and tokenize every sentence to **128 tokens** with the RoBERTa tokenizer.
5. Save each window as a JSON file (`input_ids`, `attention_mask`, `labels`). The last 10,000 windows are held out for validation.

## Model

1. **Sentence vectors:** the frozen RoBERTa embedding layer encodes each sentence's tokens, and the token embeddings are mean-pooled into one 768-d vector per sentence.
2. **Lower stack (12 layers):** RoBERTa-base layers with frozen pretrained weights, plus a trainable **bottleneck adapter** (size 64, GELU, residual connection; [Houlsby et al., 2019](https://arxiv.org/abs/1902.00751)) applied after the attention and feed-forward blocks. Only the adapters are trained in this stack.
3. **Upper stack (6 layers):** six RoBERTa layers initialized from the pretrained layers 7 to 12 and fully trainable, followed by mean pooling and a linear layer with 16 outputs (one per sentence in the window), passed through a softmax.

### Differences from the paper

This is a simplified re-implementation, not an exact reproduction of HITS:

- HITS runs the 12-layer RoBERTa (with adapters) **over the tokens of each sentence** and uses the start-token vector as the sentence embedding. Here, sentence vectors come from mean-pooled token embeddings, and the adapter layers run over the sequence of sentence vectors.
- HITS initializes the upper sentence-level transformer randomly; here it starts from pretrained RoBERTa layers.
- HITS classifies each sentence separately and averages the predictions from overlapping windows at inference; here one softmax over the 16 window outputs is used, with no averaging across windows.

## Training

| | |
|---|---|
| Loss | Binary cross-entropy |
| Optimizer | Adam |
| Learning rate | 1e-5 |
| Batch size | 32 |
| Model selection | best validation loss on the held-out windows |

## Evaluation

Segmentation quality is measured with **Pk** (Beeferman et al., 1999; lower is better): the share of sentence pairs a distance `k` apart where the model and the reference disagree about whether both sentences are in the same section. It is implemented in [`metrics.py`](metrics.py), with `k` set to half the average reference segment length.

## Limitations

- Trained on a **subset** of the training data because of Colab limits.
- English only.
- The window size (16 sentences) and the 128-token sentence limit are fixed in the code.

## How to run

The notebook was written for Google Colab and uses Google Drive paths, so adjust them for your setup.

```bash
pip install -r requirements.txt
```

1. Open `notebooks/wiki727k_text_segmentation.ipynb` in Colab and mount Google Drive.
2. Run the data-preparation cells to create the window JSON files.
3. Run the model and training cells.
4. Evaluate with `metrics.py`.

## Repository structure

```
.
├── README.md
├── metrics.py
├── notebooks/
│   └── wiki727k_text_segmentation.ipynb
├── requirements.txt
└── .gitignore
```

## Tools

Python, PyTorch, Hugging Face Transformers (RoBERTa), pandas, NumPy, scikit-learn, Google Colab

## Author

**Farah Youssef** · [LinkedIn](https://linkedin.com/in/farah-yousef-66031a328)
** Adel Dwaedari**
