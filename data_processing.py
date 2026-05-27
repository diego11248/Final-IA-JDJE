
import numpy as np
import pandas as pd
import yfinance as yf
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

def prepare_data(ticker, start_date, end_date, n_steps=30, batch_size=32):
    """
    Descarga datos de Yahoo Finance, genera etiquetas binarias, crea ventanas 
    temporales y retorna DataLoaders de PyTorch listos para el entrenamiento.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        raise ValueError(f"No se encontraron datos para el ticker {ticker}")
        
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].copy()
    
    # Target: 1 si el precio sube mañana, 0 si baja
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data = data.dropna()
    
    # Normalización
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(data[features])
    
    X, y = [], []
    for i in range(len(scaled_features) - n_steps):
        X.append(scaled_features[i : i + n_steps])
        y.append(data['Target'].iloc[i + n_steps])
        
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1) # Formato (Batch, 1) para BCELoss
    
    # División cronológica (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Conversión a Tensores de PyTorch
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test)
    
    # Creación de DataLoaders para iterar en batches
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # Falso para mantener orden temporal
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, n_steps, len(features), y_test
