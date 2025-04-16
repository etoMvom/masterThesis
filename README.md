---
# Classification of Biomedical Texts with Deep Learning: LSTM, GRU, and Soft-Attention

---

Advancements in disease prediction and medical knowledge extraction increasingly rely on processing vast amounts of biomedical literature. Scientific texts from sources like PubMed contain rich information on diseases, treatments, and biomedical mechanisms, but their unstructured nature and domain-specific terminology pose significant challenges for automated classification.

Recurrent Neural Networks (RNNs), particularly Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU), have been widely used to capture temporal and sequential dependencies in text. However, these models face limitations in representing complex relationships within the language of scientific texts. Attention-based models, such as those incorporating self-attention mechanisms, offer a compelling alternative by allowing the model to focus on the most relevant parts of the text contextually.

This work explores how LSTM, GRU, and self-attention models can enhance classification performance by extracting temporal and semantic patterns from biomedical literature, particularly in identifying disease categories from PubMed articles.

## Data Description

To evaluate these models, we use a **Multiclass Dataset** derived from PubMed, consisting of 42,879 biomedical texts covering 9 disease categories, both infectious and non-infectious. The task is to classify these texts into the appropriate disease category based on their content. The dataset spans a range of articles from research, clinical trials, scientific reviews, and epidemiological studies.

### Data Collection

The dataset used in this study is composed of **biomedical texts** collected from **PubMed** between 1950 and 2024. These are *scientific articles published in medical or biological journals*. They deal with **research, clinical trials, scientific reviews, epidemiological studies**, and more. Authored by researchers, these texts are written for an academic audience and do **not** contain clinical records or real patient data.

By using curated summaries and structured segments from these publications, we create a text-based corpus suitable for multiclass disease classification tasks.

## Model Implementation

### Word and Contextual Embeddings

To enhance the representation of biomedical text data, we employ a combination of static and contextual embeddings:

- **GloVe 300d**: A pre-trained word embedding model that captures semantic relationships between words.
- **PubMedBERT**: A contextualized transformer-based model specifically trained on biomedical literature, allowing for deeper understanding of domain-specific terminology.

### LSTM and GRU

LSTM and GRU architectures are implemented to analyze the sequential nature of biomedical abstracts and article excerpts. These models effectively capture long-term dependencies in the text, enabling more accurate classification of disease-related literature.

### Bahdanau Attention Mechanism
To further improve performance, a Bahdanau attention mechanism is integrated. Unlike LSTM/GRU, which process text sequentially, Bahdanau attention allows the model to dynamically weigh the importance of different parts of the input by learning a set of attention scores. This enables the model to focus on the most relevant parts of the input sequence, improving both interpretability and accuracy, particularly in longer or more complex texts. The attention mechanism provides a way to capture dependencies between words regardless of their distance in the sequence, ensuring that the model can better understand the relationships and context within the text.

## Conclusion

This study demonstrates the comparative strengths of LSTM, GRU, and self-attention models in biomedical text classification tasks. By leveraging structured content from scientific literature—rather than clinical patient data—and combining it with advanced deep learning techniques such as GloVe and PubMedBERT, we aim to improve disease classification performance, ultimately contributing to more efficient biomedical knowledge extraction.

