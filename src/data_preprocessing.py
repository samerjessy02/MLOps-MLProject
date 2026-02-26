import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import logging

RANDOM_SEED = 42


def load_and_preprocess_data(file_path: Path):
    logging.info("Loading dataset...")
    dataset = pd.read_csv(file_path)

    X = dataset.drop(columns=["label"])
    y = dataset["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    # === Normalize landmarks relative to wrist (landmark 1 = x1, y1, z1) ===
    logging.info("Normalizing landmarks...")
    for split in [X_train, X_test]:
        wrist_x = split["x1"].values.copy()
        wrist_y = split["y1"].values.copy()

        x_cols = [f"x{i}" for i in range(1, 22)]
        y_cols = [f"y{i}" for i in range(1, 22)]

        split[x_cols] = split[x_cols].subtract(wrist_x, axis=0)
        split[y_cols] = split[y_cols].subtract(wrist_y, axis=0)

        # Scale by distance from wrist to middle fingertip (landmark 13)
        scale = np.sqrt(split["x13"] ** 2 + split["y13"] ** 2).replace(0, 1)
        split[x_cols] = split[x_cols].divide(scale, axis=0)
        split[y_cols] = split[y_cols].divide(scale, axis=0)

    # === Encode target labels ===
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    return X_train, X_test, y_train, y_test
