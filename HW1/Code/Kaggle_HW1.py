import numpy as np
import matplotlib.pyplot as plt
from models.LinearRegression import LinearRegression

from utils import RMSE, load_data, optimizer, load_data_with_feature_select, Inference2SolutionCsv


np.random.seed(2023)

# You can EDIT the hyperparameters below.
_epoch=1000
_batch_size=256
_lr = 0.01
_optim = 'SGD'              # Write one of SGD, or Momentum
_gamma = 0.1
_normalize = None       # Write one of ZScore, MinMax, or None(Default).
_threshold = 1.5
selected_feature = ["holiday","hour"]



_dataset_name = 'Traffic'
target_feature = ["traffic_volume"]

# Data generation
train_data, test_data = load_data_with_feature_select(_dataset_name, _normalize, selected_feature+target_feature)
x_data, y_data = train_data[0], train_data[1]
test_x_data, test_y_data = test_data[0], test_data[1]

# Build model
model = LinearRegression(num_features=x_data.shape[1])
optim = optimizer(_optim, _gamma, threshold=_threshold)
print('Initial weight: \n', model.W.reshape(-1))

# Solve
model.numerical_solution(x=x_data, y=y_data, epochs=_epoch, batch_size=_batch_size, lr=_lr, optim=optim, batch_gradient=False)

print('Trained weight: \n', model.W.reshape(-1))

# Inference in Train dataset
inference = model.eval(x_data)
error = RMSE(inference, y_data)
print('RMSE on Train Data : %.4f' % error)


# Inference in Test dataset & save inference csv
inference = model.eval(test_x_data)
output_file_name = "samlpe_submission.csv"
Inference2SolutionCsv(inference, output_file_name)

