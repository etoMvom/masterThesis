

---

# 🧠 Biomedical Text Classification with Deep Learning

**LSTM • GRU • Bahdanau Attention • GloVe • Fasttext • PubMedBERT • BioBERT • Resampling(SMOTE - Bordrline-SMOTE - Weight Class)**

---

## 📘 Overview

This project explores the use of deep learning for classifying biomedical abstracts, primarily from PubMed, into disease-related categories. Given the domain-specific complexity of biomedical language, models such as **LSTM**, **GRU**, and **attention mechanism** (soft-attention) is employed to extract meaningful temporal and semantic information from scientific texts. The study emphasizes both **binary and multiclass classification tasks**, with a focus on the role of **pretrained embeddings** and **resampling techniques** to address class imbalance.

---

## 🧪 Motivation & Core Research Themes

* **Model Effectiveness & Efficiency**
  Evaluating how different neural architectures (LSTM, GRU, attention) compare in terms of training speed, inference time, parameter complexity, and predictive performance.

* **Impact of Embedding Representations**
  Assessing how performance shifts across models when using raw input, **GloVe and Fasttext embeddings**, and **BioBert and PubMedBERT**, which is specifically trained on biomedical literature.

* **Addressing Class Imbalance**
  Investigating whether class imbalance negatively impacts performance — and measuring the effectiveness of resampling techniques such as **SMOTE**, **Borderline-SMOTE**, and **Weight class**.

* **Generalizability & Few-Shot Potential**
  Exploring whether these models can generalize to underrepresented diseases, and laying the foundation for future few-shot learning strategies in biomedical text mining.

---

## 📚 Datasets

### Binary Classification Task

* **Labels**: Malaria (infectious) vs. Non-Malaria (Alzheimer’s & Dengue)
* **Size**: 29,997 abstracts (balanced: 9,999 per class)
* **Source**: PubMed abstracts from 1950–2024
* **Objective**: Evaluate model behavior in a balanced, binary scenario

### Multiclass Classification Task

* **Labels**: 9 disease classes (e.g., tuberculosis, cholera, lupus, cystic fibrosis)
* **Size**: 42,879 abstracts
* **Challenge**: Highly imbalanced distribution of classes
* **Objective**: Classify each abstract into the correct disease group

---

## 💾 Data Collection & Curation

Biomedical abstracts were extracted from **PubMed**, covering a wide span of literature types, including:

* Research articles
* Clinical trials
* Systematic reviews
* Epidemiological reports

All texts are academic, publicly available, and **de-identified**, containing no clinical or patient-level data.

---

## 🧠 Model Architectures

Three core architectures were implemented:

* **LSTM**: Captures long-term sequential dependencies
* **GRU**: More efficient alternative to LSTM, with fewer parameters
* **LSTM/GRU + Bahdanau Attention**: Dynamically focuses on the most relevant parts of the sequence, enhancing interpretability and accuracy, especially on long texts
(hybrid models CNN-GRU and CNN-LSTM used on multi-class dataset)
Each model is trained under different embedding conditions, enabling fair comparison across various levels of linguistic representation.

---

## 🧬 Embedding Strategies

To represent biomedical language effectively, we tested:

* **No Embedding**: Raw tokenized input (baseline)
* **GloVe and Fasttext (300d)**: General-purpose static embeddings
* **PubMedBERT and BioBert**: Transformer-based contextual embeddings trained on biomedical corpora

These embeddings significantly affect model behavior, especially in nuanced disease classification.

---

## ⚖️ Class Imbalance & Resampling Techniques

In the **multiclass setting**, class imbalance was addressed with:

* **Random Oversampling**
* **Random Undersampling**
* **SMOTE / Borderline-SMOTE**
* **Weight Class**

Results show substantial gains in **F1-score**, **balanced accuracy**, and **recall** for minority classes after resampling.

In contrast, the **binary classification task**, despite an underlying imbalance, yielded excellent performance across models without requiring resampling — raising questions about when and where imbalance matters most.

---

## 📈 Performance Monitoring

All experiments were tracked using **Weights & Biases (wandb.ai)**, including:

* Live training curves
* Evaluation metrics per epoch
* Confusion matrices
* Model comparison dashboards

---

## 🔍 Key Insights

* Attention mechanisms significantly boost interpretability and performance on longer abstracts.
* PubMedBERT aand BioBert consistently outperforms static embeddings like GloVe in biomedical contexts.
* Resampling is crucial in the multiclass setting but has negligible impact in well-performing binary scenarios.
* Few-shot learning remains a promising future direction to improve classification for rare diseases.

---
