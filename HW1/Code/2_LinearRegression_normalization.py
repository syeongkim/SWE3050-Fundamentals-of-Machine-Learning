import numpy as np
import matplotlib.pyplot as plt
from models.LinearRegression import LinearRegression

from utils import RMSE, load_data, optimizer, load_data_with_feature_select


np.random.seed(2023)

import itertools

# =========== You can EDIT the hyperparameters below. =========== 
_normalize = None           # Write one of ZScore, MinMax, or None(Default).
# =============================================================== 

_selected_feature = ["crim","zn","indus","chas","nox","rm","age","dis","rad","tax","ptratio","black","lstat"]
_target_feature = ["medv"]

_dataset_name = 'Boston' # Write one of Boston, Wine or Traffic

_epoch=1000
_batch_size=64
_lr = 0.001
_optim = 'SGD'              # Write one of SGD, or Momentum
_gamma = 0.1

"""
Change GD to:
    True for numerical solution
    False for analytic solution
"""

GD = True

# Data generation
print("="*20)
print(f"selected_feature: {_selected_feature}")
train_data, _ = load_data_with_feature_select(_dataset_name, _normalize, _selected_feature+_target_feature)
x_data, y_data = train_data[0], train_data[1]

# Build model
model = LinearRegression(num_features=x_data.shape[1])
optim = optimizer(_optim, _gamma)
print('Initial weight: \n', model.W.reshape(-1))

# Solve
if GD:
    model.numerical_solution(x=x_data, y=y_data, epochs=_epoch, batch_size=_batch_size, lr=_lr, optim=optim, batch_gradient=False)
    # pass
else:
    model.analytic_solution(x=x_data, y=y_data)

print('Trained weight: \n', model.W.reshape(-1))

# Inference
inference = model.eval(x_data)
# Assess model
error = RMSE(inference, y_data)
print('RMSE on Train Data : %.4f' % error)
