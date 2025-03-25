"""
DO NOT EDIT ANY PARTS OTHER THAN "EDIT HERE" !!!

"""

class SGD:
    def __init__(self, threshold):
        self.threshold = threshold # for gradient_clipping

    def update(self, w, grad, lr):
        grad = gradient_clipping(grad, self.threshold)
        # ==== EDIT HERE ====

        updated_weight = None
        
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
        
        updated_weight = None
        
        # ===================
        return updated_weight
    

def gradient_clipping(grad, threshold):
    """
    Gradient clipping stabilizes learning by clipping the gradient when it exceeds a certain threshold.
    This is usually done by dividing by the L2 norm of the gradient.
    """
    norm_squared = sum(g**2 for g in grad)  
    L2_norm = norm_squared ** 0.5  # calculate L2_norm 
    # ==== EDIT HERE ====
    
    
    # ===================
    return grad