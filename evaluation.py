
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def train_and_evaluate(model, name_model, train_loader, test_loader, y_true_test, epochs=30, lr=0.001):
    print(f"\n Entrenando: {name_model}")
    
    # Configuración de dispositivo (Usa GPU si está disponible)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.BCELoss() # Función de pérdida binaria
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Bucle de Entrenamiento
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Resetear gradientes
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass y optimización
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * X_batch.size(0)
            
        total_loss = epoch_loss / len(train_loader.dataset)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Época [{epoch+1}/{epochs}] - Pérdida: {total_loss:.4f}")
            
    # Bucle de Evaluación
    model.eval()
    all_preds = []
    
    with torch.no_grad(): # Desactivar cálculo de gradientes para ahorrar memoria
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            # Si la probabilidad es mayor a 0.5 clasifica como 1 (Sube)
            preds = (outputs > 0.5).int().cpu().numpy()
            all_preds.append(preds)
            
    y_pred = np.vstack(all_preds)
    
    # Despliegue de métricas
    print("\n" + "="*50)
    print(f"      REPORTE DE EVALUACIÓN: {name_model.upper()}      ")
    print("="*50)
    print(classification_report(y_true_test, y_pred, target_names=['Baja', 'Subida'], zero_division=0))
    
    print("MATRIZ DE CONFUSIÓN:")
    print(confusion_matrix(y_true_test, y_pred))
    print("="*50 + "\n")
