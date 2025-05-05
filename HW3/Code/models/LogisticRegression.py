import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np

def to_tensor(x, dtype=torch.float32):
    if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
        x = x.values
    return torch.tensor(np.array(x), dtype=dtype)


class LogisticRegression(nn.Module):
    def __init__(self, num_features=784, num_classes=10):
        super(LogisticRegression, self).__init__()
        self.W = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.W(x)  # logits (no softmax here)

    def l2_regularization(self):
        loss = None
        # ========== STUDENT CODE HERE ==========
        
        # ========== STUDENT CODE HERE ==========
        return loss
    def train(self, X, y, X_val, y_val, learning_rate=0.1, epochs=1000, 
                     early_stopping=True, lambda_reg= 0.01, patience=5 ):

        # Convert to tensors
        X_train = to_tensor(X, dtype=torch.float32)
        y_train = to_tensor(y, dtype=torch.long)
        X_val = to_tensor(X_val, dtype=torch.float32)
        y_val = to_tensor(y_val, dtype=torch.long)
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        no_improve = 0
        best_weights = None
        train_losses, val_losses = [], []
        lambda_reg  = float(lambda_reg)
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.forward(X_train)
            loss = criterion(outputs, y_train)
            if lambda_reg>0:
                l2_norm = self.l2_regularization()
                loss += lambda_reg * l2_norm

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # Validation
            with torch.no_grad():
                val_outputs = self.forward(X_val)
                val_loss = criterion(val_outputs, y_val)
                if lambda_reg>0:
                    val_loss += lambda_reg * self.l2_regularization()
                val_losses.append(val_loss.item())

            if early_stopping:
                # ========== STUDENT CODE HERE ==========
                #use val_loss, best_val_loss, best_weights, no_imporve
                pass
                # ========== STUDENT CODE HERE ==========


        if early_stopping and best_weights is not None:
            self.load_state_dict(best_weights)

        return {
            "train_loss": train_losses[-1],
            "val_loss": val_losses[-1]
        }

    def predict(self, X):
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        with torch.no_grad():
            logits = self.forward(X_tensor)
            return torch.argmax(F.softmax(logits, dim=1), dim=1).numpy()
