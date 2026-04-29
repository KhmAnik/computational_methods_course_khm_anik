from minicamels import MiniCamels
import numpy as np
from sklearn.preprocessing import StandardScaler


def create_sequences(data, seq_len=30):
    X_vars = ["prcp", "tmax", "tmin", "srad", "vp"]
    y_var = "qobs"

    X = np.stack([data[var].values for var in X_vars], axis=1)
    y = data[y_var].values

    X_seq, y_seq = [], []

    for i in range(len(X) - seq_len):
        X_seq.append(X[i:i+seq_len])
        y_seq.append(y[i+seq_len])

    return np.array(X_seq), np.array(y_seq)


def normalize_data(X, y):
    n_samples, seq_len, n_features = X.shape

    X_reshaped = X.reshape(-1, n_features)

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_reshaped)

    X_scaled = X_scaled.reshape(n_samples, seq_len, n_features)

    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    return X_scaled, y_scaled, scaler_X, scaler_y


def train_test_split(X, y, split_ratio=0.8):
    split = int(len(X) * split_ratio)

    X_train = X[:split]
    y_train = y[:split]

    X_test = X[split:]
    y_test = y[split:]

    return X_train, X_test, y_train, y_test