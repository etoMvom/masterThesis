---
# Biomedical Text Classification Using LSTM, GRU, and Bahdanau Attention
---

## 📘 **Overview**


This project explores the application of deep learning techniques for classifying biomedical abstracts into disease-related categories. Given the complexity of biomedical language and the presence of long sentences in abstracts, we utilize deep learning architectures such as **LSTM**, **GRU**, and **GRU + Bahdanau attention** to effectively capture nuances and dependencies within these texts.

## 🧪 **Motivation & Core Research Themes**

- _Model Effectiveness & Efficiency_: Evaluating the performance of different neural architectures (LSTM, GRU, GRU + Bahdanau attention) in terms of training speed, inference time, parameter complexity, and predictive performance.
- _Impact of Embedding Representations_: Assessing the impact of various embedding strategies (raw input, GloVe, Fasttext, BioBert, PubMedBERT) on model performance.
- _Addressing Class Imbalance_: Investigating the effectiveness of techniques to address class imbalance.
- _Generalizability & Few-Shot Potential_: Exploring the potential of these models to generalize to underrepresented diseases and laying the foundation for future few-shot learning strategies.

## 📚 **Dataset**

Binary Classification Task

- _Labels_: Malaria vs. Non-Malaria (Alzheimer’s & Dengue)
- _Size_: 29,997 abstracts (imbalanced)
- _Source_: PubMed abstracts from 1950–2024
- _Objective_: Evaluate model behavior in a binary scenario with class imbalance

Multiclass Classification Task

- _Labels_: 9 disease classes (e.g., tuberculosis, cholera, lupus, cystic fibrosis)
- _Size_: 42,879 abstracts (imbalanced)
-  _Source_: PubMed abstracts from 1950–2024;
- _Objective_: Evaluate model behavior in a multi-class scenario with class imbalance.

## 💾 **Data Collection & Curation**

Biomedical abstracts were extracted from _PubMed_, covering a wide range of literature types, including research articles, clinical trials, systematic reviews, and epidemiological reports. All texts are academic, publicly available, and _de-identified_, containing no clinical or patient-level data.

## 🧠 **Model Architectures**

#### Binary Classification Task

- _LSTM_: Captures long-term sequential dependencies
- _GRU_: More efficient alternative to LSTM, with fewer parameters
- _GRU + Bahdanau Attention_: Dynamically focuses on the most relevant parts of the sequence using Bahdanau attention, enhancing interpretability and accuracy

#### Multiclass Classification Task

- _LSTM_: Captures long-term sequential dependencies
- _GRU_: More efficient alternative to LSTM, with fewer parameters
- _CNN + LSTM_: Combines convolutional layers with LSTM to capture both local and long-term dependencies
- _CNN + GRU_: Combines convolutional layers with GRU to capture both local and sequential dependencies
- _GRU + Bahdanau Attention_: Dynamically focuses on the most relevant parts of the sequence using Bahdanau attention, enhancing interpretability and accuracy

## 🧬 **Embedding Strategies**
To represent biomedical language effectively, we tested:

- _No Embedding_: Raw tokenized input (baseline)
- _GloVe and Fasttext (300d)_: General-purpose static embeddings
- _PubMedBERT and BioBert_: Transformer-based contextual embeddings trained on biomedical corpora

## ⚖️ **Class Imbalance & Handling Techniques**

In both the _binary_ and _multiclass settings_, class imbalance was addressed with:

- _SMOTE / Borderline-SMOTE_ (resampling techniques)
- _Weight Class_ (assigning different weights to classes in the loss function)

## 📈 **Performance Monitoring**

All experiments were tracked using _Weights & Biases_, including live training curves, evaluation metrics per epoch, confusion matrices, and model comparison dashboards.

## 🔍 **Key Insights**

- The GRU + Bahdanau attention architecture effectively captures nuances and dependencies in biomedical texts.
- PubMedBERT and BioBert consistently outperform static embeddings like GloVe in biomedical contexts.
- Addressing class imbalance is crucial in both binary and multiclass settings.
- Few-shot learning remains a promising future direction to improve classification for rare diseases.
