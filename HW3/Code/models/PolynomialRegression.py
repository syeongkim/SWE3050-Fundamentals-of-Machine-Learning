# models/PolynomialRegression.py
import torch
import numpy as np
from torch.optim import SGD

class PolynomialRegression:
    def __init__(self, degree, reg_method=None, reg_lambda=0.0):
        self.degree = degree
        self.W = torch.randn((self.degree + 1, 1), dtype=torch.float32, requires_grad=True)
        self.reg_method = reg_method
        self.reg_lambda = reg_lambda

    def PolynomialFeatures(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.view(-1, 1)
        # ========== STUDENT CODE HERE ==========
        features = [x ** i for i in reversed(range(1, self.degree + 1))]
        features.append(torch.ones_like(x))
        # ========== STUDENT CODE HERE ==========
        return torch.cat(features, dim=1)

    def forward(self, x):
        return torch.matmul(x, self.W)

    def l2_reg(self):
        # ========== STUDENT CODE HERE ==========
        return torch.sum(self.W ** 2)
        # ========== STUDENT CODE HERE ==========

    def train(self, x, y, batch_size, epochs, lr, val_x=None, val_y=None, early_stopping=False, patience=5):
        self.optimizer = SGD([self.W], lr=lr)
        x = self.PolynomialFeatures(x)
        y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

        best_val_rmse = float('inf')
        best_weights = None
        no_improve = 0

        for epoch in range(1, epochs + 1):
            permutation = torch.randperm(x.size(0))
            epoch_loss = 0.0

            for i in range(0, x.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_x, batch_y = x[indices], y[indices]
                
                # ========== STUDENT CODE HERE ==========

                # Forward pass (compute prediction)
                pred = self.forward(batch_x)
                # Compute MSE loss
                loss = torch.mean((pred-batch_y) ** 2)
                

                if self.reg_method == 'l2':
                # ========== STUDENT CODE HERE ==========
                    reg = self.reg_lambda * self.l2_reg()
                # ========== STUDENT CODE HERE ==========
                else:
                    reg = torch.tensor(0.0)      
                # Add regularization term to the loss if reg is used.
                loss += reg

                # Backward + update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Accumulate epoch_loss
                epoch_loss += loss.item() * batch_x.size(0)
                # ========== STUDENT CODE HERE ==========
                

            if epoch % 1000 == 0 or epoch == epochs:
                print(f"[Epoch {epoch}] Training Loss: {epoch_loss:.4f}")

            if early_stopping and val_x is not None and val_y is not None:
                val_rmse = self.eval(val_x, val_y, batch_size)
                # ========== STUDENT CODE HERE ==========
                #use val_rmse, best_val_rmse, best_weights, no_imporve
                if val_rmse < best_val_rmse:
                    best_val_rmse = val_rmse
                    best_weights = self.W.detach().clone()
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break
                # ========== STUDENT CODE HERE ==========

        if early_stopping and best_weights is not None:
            self.W = best_weights

    def eval(self, x, y, batch_size):
        x = self.PolynomialFeatures(x)
        y = y.reshape(-1, 1)
        pred = []

        with torch.no_grad():
            for i in range(0, x.size(0), batch_size):
                batch_x = x[i:i + batch_size]
                batch_pred = self.forward(batch_x)
                pred.append(batch_pred)

        pred = torch.cat(pred, dim=0).numpy()
        return np.sqrt(np.mean((y - pred) ** 2))
