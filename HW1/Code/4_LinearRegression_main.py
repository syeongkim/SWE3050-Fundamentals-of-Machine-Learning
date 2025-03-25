import numpy as np
import matplotlib.pyplot as plt
from models.LinearRegression import LinearRegression

from utils import RMSE, load_data, optimizer, load_data_with_feature_select


np.random.seed(2023)



# You can EDIT the hyperparameters below.
_epoch=1000
_batch_size=64
_lr = 0.001
_optim = 'SGD'              # Write one of SGD, or Momentum
_gamma = 0.1
_normalize = None           # Write one of ZScore, MinMax, or None(Default).

_dataset_name = 'Wine'    # Write one of Boston, Wine or Traffic


"""
Change GD to:
    True for numerical solution
    False for analytic solution
"""


GD = True

# Data generation
train_data, _ = load_data_with_feature_select(_dataset_name, _normalize)
x_data, y_data = train_data[0], train_data[1]

# Build model
model = LinearRegression(num_features=x_data.shape[1])
optim = optimizer(_optim, _gamma)
print('Initial weight: \n', model.W.reshape(-1))

# Solve
if GD:
    model.numerical_solution(x=x_data, y=y_data, epochs=_epoch, batch_size=_batch_size, lr=_lr, optim=optim, batch_gradient=False)
else:
    model.analytic_solution(x=x_data, y=y_data)
    pass

print('Trained weight: \n', model.W.reshape(-1))

# Inference
inference = model.eval(x_data)
# Assess model
error = RMSE(inference, y_data)
print('RMSE on Train Data : %.4f' % error)
