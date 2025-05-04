Biomedical Text Classification Using LSTM, GRU, and Bahdanau Attention

📘 Overview
This project explores the application of deep learning techniques for classifying biomedical abstracts into disease-related categories. Given the complexity of biomedical language and the presence of long sentences in abstracts, we utilize deep learning architectures such as _LSTM_, _GRU_, and _GRU + Bahdanau attention_ to effectively capture nuances and dependencies within these texts.

🧪 Motivation & Core Research Themes
- _Model Effectiveness & Efficiency_: Evaluating the performance of different neural architectures (LSTM, GRU, GRU + Bahdanau attention) in terms of training speed, inference time, parameter complexity, and predictive performance.
- _Impact of Embedding Representations_: Assessing the impact of various embedding strategies (raw input, GloVe, Fasttext, BioBert, PubMedBERT) on model performance.
- _Addressing Class Imbalance_: Investigating the effectiveness of resampling techniques (SMOTE, Borderline-SMOTE, Weight class) in addressing class imbalance.
- _Generalizability & Few-Shot Potential_: Exploring the potential of these models to generalize to underrepresented diseases and laying the foundation for future few-shot learning strategies.

📚 Datasets
Binary Classification Task
- _Labels_: Malaria vs. Non-Malaria (Alzheimer’s & Dengue)
- _Size_: 29,997 abstracts (balanced: 9,999 per class)
- _Source_: PubMed abstracts from 1950–2024
- _Objective_: Evaluate model behavior in a balanced, binary scenario

Multiclass Classification Task
- _Labels_: 9 disease classes (e.g., tuberculosis, cholera, lupus, cystic fibrosis)
- _Size_: 42,879 abstracts
- _Challenge_: Highly imbalanced distribution of classes
- _Objective_: Classify each abstract into the correct disease group

💾 Data Collection & Curation
Biomedical abstracts were extracted from _PubMed_, covering a wide range of literature types, including research articles, clinical trials, systematic reviews, and epidemiological reports. All texts are academic, publicly available, and _de-identified_, containing no clinical or patient-level data.

🧠 Model Architectures
Three core architectures were implemented:

- _LSTM_: Captures long-term sequential dependencies
- _GRU_: More efficient alternative to LSTM, with fewer parameters
- _GRU + Bahdanau Attention_: Dynamically focuses on the most relevant parts of the sequence using Bahdanau attention, enhancing interpretability and accuracy

🧬 Embedding Strategies
To represent biomedical language effectively, we tested:

- _No Embedding_: Raw tokenized input (baseline)
- _GloVe and Fasttext (300d)_: General-purpose static embeddings
- _PubMedBERT and BioBert_: Transformer-based contextual embeddings trained on biomedical corpora

⚖️ Class Imbalance & Resampling Techniques
In the _multiclass setting_, class imbalance was addressed with:

- _SMOTE / Borderline-SMOTE_
- _Weight Class_

📈 Performance Monitoring
All experiments were tracked using _Weights & Biases_, including live training curves, evaluation metrics per epoch, confusion matrices, and model comparison dashboards.

🔍 Key Insights
- The GRU + Bahdanau attention architecture effectively captures nuances and dependencies in biomedical texts.
- PubMedBERT and BioBert consistently outperform static embeddings like GloVe in biomedical contexts.
- Resampling is crucial in the multiclass setting but has negligible impact in well-performing binary scenarios.
- Few-shot learning remains a promising future direction to improve classification for rare diseases.
