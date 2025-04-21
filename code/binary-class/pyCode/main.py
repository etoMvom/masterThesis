import torch
import time
import random
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel

from src.utils import *
from torch.utils.data import DataLoader, ConcatDataset

from src.model import PubMedBERT_GRU_Attention
from sklearn.model_selection import train_test_split
from src.training import train_model, epoch_time, count_parameters
from src.TextDataset import PubMedBERTBinaryDataset, search_pubmed, fetch_abstracts, preprocess_text


def main():
    SEED = 200
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    BATCH_SIZE = 16
    num_epochs = 4
    learning_rate = 9e-5
    weight_decay = 1e-5
    loss_gap_ratio = 1.25
    ax_accuracy_gap = 5
    model_name = "pubmed_bert_model"
    saved_once = False

    print()
    print("------------------------Collecting data ...------------------------")
    diseases = {
        "Tuberculosis": '"tuberculosis" AND (1950:2024[DP])',
        "Cholera": '"cholera" AND (1950:2024[DP])',
        "Leprosy": '"leprosy" AND (1950:2024[DP])',
        "Ebola": '"Ebola virus disease" AND (1950:2024[DP])',
        "Leukemia": '"leukemia" AND (1950:2024[DP])',
        "Asthma": '"asthma" AND (1950:2024[DP])',
        "Parkinson": '"Parkinson disease" AND (1950:2024[DP])',
        "Lupus": '"systemic lupus erythematosus" AND (1950:2024[DP])',
        "Cystic Fibrosis": '"cystic fibrosis" AND (1950:2024[DP])'
    }

    num_classes = len(diseases)
    dfs = []
    for label, (disease, query) in enumerate(diseases.items()):
        print(f"Searching for {disease}...")
        ids = search_pubmed(query)
        data = fetch_abstracts(ids)
        df = pd.DataFrame(data)
        df["Cleaned_Abstract"] = df["Abstract"].apply(preprocess_text)
        df["Label"] = label
        df["Disease"] = disease
        dfs.append(df)

    print()
    print("------------------------Merging all disease data, shuffling, and removing incomplete entries...------------------------")
    print()
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    combined_df.replace(["", "None", "null"], np.nan, inplace=True)

    combined_df = combined_df.dropna()
    combined_df = combined_df.reset_index(drop=True)

    print(combined_df.info())
    print(combined_df['Label'].value_counts())
    print()

    print("------------------------Splitting data into train/val/test sets...------------------------")
    X_train, X_temp, y_train, y_temp = train_test_split(
        combined_df["Cleaned_Abstract"].values,
        combined_df["Label"].values,
        test_size=0.3,
        random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    pubmedbert_model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    tokenizer = AutoTokenizer.from_pretrained(pubmedbert_model_name)
    pubmedbert = AutoModel.from_pretrained(pubmedbert_model_name)

    train_dataset = PubMedBERTBinaryDataset(X_train.tolist(), y_train, tokenizer)
    val_dataset = PubMedBERTBinaryDataset(X_val.tolist(), y_val, tokenizer)
    test_dataset = PubMedBERTBinaryDataset(X_test.tolist(), y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    combined_loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True)

    bert_dim = 768
    hidden_dim = 256
    model = PubMedBERT_GRU_Attention(bert_dim, hidden_dim, num_classes=9, num_layers=2, dropout_prob=0.6)

    print("------------------------Initializing model ...------------------------")
    print(model)
    print(f"{count_parameters(model)} model parameters")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.CrossEntropyLoss()

    print("------------------------Beginning training and evaluation per epoch ...------------------------")
    print()
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_acc = train_model(model, combined_loader, optimizer, loss_function, device)
        test_loss, test_acc, f1_test, balanced_accuracy_test, recall_test, precision_test = test_model(
            model, test_loader, loss_function, device
        )
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%} | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2%} | "
            f"F1: {f1_test:.2%}, Bal Acc: {balanced_accuracy_test:.2%}, "
            f"Recall: {recall_test:.2%}, Precision: {precision_test:.2%} | "
            f"Time: {epoch_mins}m {epoch_secs}s"
        )

        if (test_loss > train_loss * loss_gap_ratio or
            (train_acc > test_acc and train_acc - test_acc > ax_accuracy_gap / 100)) and not saved_once:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            torch.save(model.state_dict(), f"{model_name}_early_stop.pth")
            saved_once = True
            break

        if epoch == num_epochs - 1 and not saved_once:
            print("Saving final model.")
            torch.save(model.state_dict(), f"{model_name}_final.pth")


if __name__ == "__main__":
    main()
