import torch
import torch.nn as nn
import time
import random
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader

from src.utils import *
from src.model import PubMedBERT_GRU_Attention
from src.training import train_model, test_model, epoch_time, count_parameters
from src.TextDataset import PubMedBERTDataset, search_pubmed, fetch_abstracts, preprocess_text
from sklearn.model_selection import train_test_split


def main():
    # ------------------------ Reproducibility ------------------------
    SEED = 200
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # ------------------------ Hyperparameters ------------------------
    BATCH_SIZE = 16
    num_epochs = 4
    learning_rate = 9e-5
    weight_decay = 1e-5
    loss_gap_ratio = 1.25
    acc_gap_threshold = 5
    model_name = "pubmed_bert_binary_model"
    saved_once = False

    print("\n------------------------ Collecting data ------------------------")
    diseases_queries = {
        "malaria": '("malaria" OR "Plasmodium" OR "malarial infection") AND (1950:2024[DP])',
        "parasitic": '"parasitic diseases" AND (1950:2024[DP])',
        "other": '"disease" NOT "malaria" AND NOT "parasitic diseases" AND (1950:2024[DP])'
    }

    dfs = []
    for disease, query in diseases_queries.items():
        print(f"Searching for {disease}...")
        ids = search_pubmed(query)
        data = fetch_abstracts(ids)
        df = pd.DataFrame(data)
        df["Cleaned_Abstract"] = df["Abstract"].apply(preprocess_text)
        df["Label"] = 1 if disease == "malaria" else 0
        df["Disease"] = disease
        dfs.append(df)

    print("\n------------------------ Merging, shuffling, cleaning ------------------------")
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    combined_df.replace(["", "None", "null"], np.nan, inplace=True)
    combined_df.dropna(inplace=True)
    combined_df.reset_index(drop=True, inplace=True)

    print(combined_df.info())
    print(combined_df['Label'].value_counts(), "\n")

    # ------------------------ Split Dataset ------------------------
    print("------------------------ Splitting data ------------------------")
    X_train, X_temp, y_train, y_temp = train_test_split(
        combined_df["Cleaned_Abstract"].values,
        combined_df["Label"].values,
        test_size=0.3,
        random_state=42,
        stratify=combined_df["Label"].values
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp
    )

    # ------------------------ Load PubMedBERT tokenizer and model ------------------------
    pubmedbert_model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    tokenizer = AutoTokenizer.from_pretrained(pubmedbert_model_name)
    pubmedbert = AutoModel.from_pretrained(pubmedbert_model_name)

    # ------------------------ Prepare custom dataset objects ------------------------
    train_dataset = PubMedBERTDataset(X_train.tolist(), y_train, tokenizer)
    val_dataset = PubMedBERTDataset(X_val.tolist(), y_val, tokenizer)
    test_dataset = PubMedBERTDataset(X_test.tolist(), y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ------------------------ Initialize model ------------------------
    bert_dim = 768
    hidden_dim = 256
    model = PubMedBERT_GRU_Attention(bert_dim, hidden_dim, num_layers=2, dropout_prob=0.6)

    print("\n------------------------ Initializing model ------------------------")
    print(model)
    print(f"{count_parameters(model)} model parameters\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.BCEWithLogitsLoss()

    print("------------------------ Training ------------------------\n")
    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_acc = train_model(model, train_loader, optimizer, loss_function, device)
        test_loss, test_acc, f1_test, bal_acc_test, recall_test, precision_test = test_model(
            model, test_loader, loss_function, device
        )

        epoch_mins, epoch_secs = epoch_time(start_time, end_time=time.time())

        # Print epoch metrics
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%} | "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2%} | "
            f"F1: {f1_test:.2%}, Bal Acc: {bal_acc_test:.2%}, "
            f"Recall: {recall_test:.2%}, Precision: {precision_test:.2%} | "
            f"Time: {epoch_mins}m {epoch_secs}s"
        )

        # Early stopping condition: stop if test loss is too high compared to train loss or large accuracy gap
        if (
            test_loss > train_loss * loss_gap_ratio or
            (train_acc > test_acc and (train_acc - test_acc) * 100 > acc_gap_threshold)
        ) and not saved_once:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            torch.save(model.state_dict(), f"{model_name}_early_stop.pth")
            saved_once = True
            break

        # Save the final model if training completes normally
        if epoch == num_epochs - 1 and not saved_once:
            print("Saving final model.")
            torch.save(model.state_dict(), f"{model_name}_final.pth")


if __name__ == "__main__":
    main()
