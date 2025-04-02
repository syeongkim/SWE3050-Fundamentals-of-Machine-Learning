"""
DO NOT EDIT ANY PARTS OTHER THAN "EDIT HERE" !!!

"""

import numpy as np

class SGD:
    def __init__(self, threshold):
        self.threshold = threshold # for gradient_clipping

    def update(self, w, grad, lr):
        grad = gradient_clipping(grad, self.threshold)
        # ==== EDIT HERE ====

        updated_weight = w - lr * grad
        
        # ===================
        return updated_weight

class Momentum:
    def __init__(self, gamma, threshold):
        self.threshold = threshold # for gradient_clipping
        self.prev_grad = 0
        self.gamma = gamma

    def update(self, w, grad, lr):
        grad = gradient_clipping(grad, self.threshold)
        # ==== EDIT HERE ====
        
        self.prev_grad = self.gamma * self.prev_grad + lr * grad
        updated_weight = w - self.prev_grad
        
        # ===================
        return updated_weight
    
class Adam: # 5번 문제(Mini Kaggle challenge)를 위해 추가함
    def __init__(self, threshold):
        self.m = None
        self.v = None
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.threshold = threshold
        self.step = 0

    def update(self, w, grad, lr, **kwargs):
        self.step += 1
        if self.m is None:
            self.m = np.zeros_like(grad)
            self.v = np.zeros_like(grad)

        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad ** 2)

        m_hat = self.m / (1 - self.beta1 ** self.step)
        v_hat = self.v / (1 - self.beta2 ** self.step)

        return w - lr * m_hat / (np.sqrt(v_hat) + self.eps)

class Adagrad: # 5번 문제(Mini Kaggle challenge)를 위해 추가함
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon
        self.G = None  # 누적된 제곱 gradients

    def update(self, W, grad, lr):
        if self.G is None:
            self.G = np.zeros_like(W)

        self.G += grad ** 2
        adjusted_grad = lr * grad / (np.sqrt(self.G) + self.epsilon)

        return W - adjusted_grad
    
class RMSProp: # 5번 문제(Mini Kaggle challenge)를 위해 추가함
    def __init__(self, beta=0.9, epsilon=1e-8):
        self.beta = beta
        self.epsilon = epsilon
        self.Eg = None  # 지수이동평균된 squared gradients

    def update(self, W, grad, lr):
        if self.Eg is None:
            self.Eg = np.zeros_like(W)

        self.Eg = self.beta * self.Eg + (1 - self.beta) * (grad ** 2)
        adjusted_grad = lr * grad / (np.sqrt(self.Eg) + self.epsilon)

        return W - adjusted_grad

def gradient_clipping(grad, threshold):
    """
    Gradient clipping stabilizes learning by clipping the gradient when it exceeds a certain threshold.
    This is usually done by dividing by the L2 norm of the gradient.
    """
    norm_squared = sum(g**2 for g in grad)  
    L2_norm = norm_squared ** 0.5  # calculate L2_norm 
    # ==== EDIT HERE ====
    
    if L2_norm > threshold:
        grad = (grad / L2_norm) * threshold
        
    # ===================
    return grad