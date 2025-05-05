import os
import wfdb
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
import joblib
import argparse

PTBXL_CSV = "data/ptbxl/ptbxl_database.csv"
PTBXL_RECORDS_PATH = "data/ptbxl/records500/"

SEGMENT_LENGTH = 2500  # ~10 seconds at 500 Hz
NUM_LEADS = 12

def load_metadata():
    df = pd.read_csv(PTBXL_CSV)
    df["scp_codes"] = df["scp_codes"].apply(eval)  # convert string dict to real dict
    return df

def extract_labels(df, threshold=100):
    """Extract high-frequency diagnostic labels from metadata."""
    scp_statements = pd.read_csv("data/ptbxl/scp_statements.csv", index_col=0)
    scp_statements = scp_statements[scp_statements["diagnostic"] == 1]
    valid_labels = set(scp_statements.index)
    
    df["diagnostic_labels"] = df["scp_codes"].apply(
        lambda code_dict: [code for code in code_dict if code in valid_labels]
    )
    
    # Filter out rare labels
    all_labels = df["diagnostic_labels"].explode()
    label_counts = all_labels.value_counts()
    frequent_labels = label_counts[label_counts >= threshold].index.tolist()
    
    df["filtered_labels"] = df["diagnostic_labels"].apply(
        lambda labels: [label for label in labels if label in frequent_labels]
    )
    
    df = df[df["filtered_labels"].map(len) > 0]  # keep rows with at least 1 valid label
    return df, frequent_labels

def load_waveform(record_name):
    record_path = os.path.join("data/ptbxl", record_name)
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal.T  # shape: (12, time)
    
    if signal.shape[1] >= SEGMENT_LENGTH:
        signal = signal[:, :SEGMENT_LENGTH]
    else:
        pad_width = SEGMENT_LENGTH - signal.shape[1]
        signal = np.pad(signal, ((0, 0), (0, pad_width)), mode='constant')
        
    return signal  # shape: (12, 2500)

def normalize_signals(signals):
    """Normalize each lead separately across the training set."""
    normalized = np.zeros_like(signals)
    scalers = []
    
    for i in range(NUM_LEADS):
        scaler = StandardScaler()
        normalized[:, i, :] = scaler.fit_transform(signals[:, i, :])
        scalers.append(scaler)
    
    return normalized, scalers

def apply_scalers(signals, scalers):
    normalized = np.zeros_like(signals)
    for i in range(NUM_LEADS):
        normalized[:, i, :] = scalers[i].transform(signals[:, i, :])
    return normalized

def main():
    print("\n Loading metadata...")
    df = load_metadata()
    df, class_labels = extract_labels(df)

    print(f"✅ Using {len(class_labels)} diagnostic classes after filtering\\n")
    
    df["first_label"] = df["filtered_labels"].apply(lambda x: x[0])
    train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df["first_label"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["first_label"], random_state=42)

    for split_name, split_df in zip(["train", "val", "test"], [train_df, val_df, test_df]):
        print(f"🔄 Loading waveforms for {split_name} set...")
        signals = np.stack([
            load_waveform(record_name)
            for record_name in split_df["filename_hr"].tolist()
        ])  # shape: (N, 12, 2500)

        if split_name == "train":
            print("🔬 Normalizing training data...")
            signals, scalers = normalize_signals(signals)
            joblib.dump(scalers, "data/scalers.pkl")
        else:
            scalers = joblib.load("data/scalers.pkl")
            signals = apply_scalers(signals, scalers)

        print(f"💾 Saving {split_name}_signals.npy...")
        np.save(f"data/{split_name}_signals.npy", signals)

        print(f"🔖 Binarizing and saving {split_name} labels...")
        mlb = MultiLabelBinarizer(classes=class_labels)
        multilabels = mlb.fit_transform(split_df["filtered_labels"])
        np.save(f"data/{split_name}_labels.npy", multilabels)

    print("\\n✅ Preprocessing complete!\\n")

if __name__ == "__main__":
    main()
