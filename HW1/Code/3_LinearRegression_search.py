import numpy as np
from models.LinearRegression import LinearRegression
import matplotlib.pyplot as plt
from utils import RMSE, load_data, optimizer


np.random.seed(2023)

"""
Choose param to search. (batch_size or lr or gamma for Momentum)
Specify values of the parameter to search,
and fix the other.
e.g.)
search_param = 'lr'
_batch_size = 32
_lr = [0.1, 0.01, 0.05]
"""

# You can EDIT the hyperparameters below.
_epoch=200
search_param = 'clipping'
_batch_size = 64                
_lr = 0.01              
_optim = 'SGD'                        # Write one of SGD, or Momentum
_gamma = 0.2        
_clipping = [0.1, 2, 10]              # Write clipping threshold
_normalize = None                     # Write one of ZScore, MinMax, or None(Default).

_dataset_name = 'Wine'                # Write one of Boston, Wine or Traffic




# Data generation
train_data, test_data = load_data(_dataset_name, _normalize)
x_train_data, y_train_data = train_data[0], train_data[1]
x_test_data, y_test_data = test_data[0], test_data[1]

train_results = []
test_results = []
if search_param == 'lr':
    search_space = _lr
elif search_param == 'batch_size':
    search_space = _batch_size
elif search_param == 'gamma':
    search_space = _gamma
elif search_param == 'clipping':
    search_space = _clipping
else:
    pass


total_errors = []
# 서브플롯 생성
fig, ax = plt.subplots(1, 3, figsize=(10, 4))

for i, space in enumerate(search_space):
    # Build model
    model = LinearRegression(num_features=x_train_data.shape[1], weight_tracking=True)
    optim = optimizer(_optim, _gamma)

    # Train model with gradient descent
    if search_param == 'lr':
        model.numerical_solution(x=x_train_data, y=y_train_data, epochs=_epoch, batch_size=_batch_size, lr=space, optim=optim)
    elif search_param == 'batch_size':
        model.numerical_solution(x=x_train_data, y=y_train_data, epochs=_epoch, batch_size=space, lr=_lr, optim=optim)
    elif search_param == 'gamma':
        optim = optimizer(_optim, space)
        model.numerical_solution(x=x_train_data, y=y_train_data, epochs=_epoch, batch_size=_batch_size, lr=_lr, optim=optim)
    elif search_param == 'clipping':
        optim = optimizer(_optim, _gamma, space)
        model.numerical_solution(x=x_train_data, y=y_train_data, epochs=_epoch, batch_size=_batch_size, lr=_lr, optim=optim)
    
    ################### Evaluate on train data
    # Inference
    inference = model.eval(x_train_data)

    # Assess model
    error = RMSE(inference, y_train_data)
    print('[Search %d] RMSE on Train Data : %.4f' % (i+1, error))

    ################### Evaluate on test data
    # Inference
    inference = model.eval(x_test_data)

    # Assess model
    error = RMSE(inference, y_test_data)
    print('[Search %d] RMSE on Test Data : %.4f' % (i+1, error))

    test_results.append(error)
    
    # Calculate train & test loss
    train_loss_list = []
    for W in model.W_list:
        inference = np.dot(x_train_data, W)
        train_loss = RMSE(inference, y_train_data)
        train_loss_list.append(train_loss)
        
    test_loss_list = []
    for W in model.W_list:
        inference = np.dot(x_test_data, W)
        test_loss = RMSE(inference, y_test_data)
        test_loss_list.append(test_loss)

    ax[i].plot(train_loss_list, color="r", label="train_loss")
    ax[i].plot(test_loss_list, color="b", label="test_loss")
    ax[i].legend()
    ax[i].set_title(f"{search_param}:{space} Plot")
    ax[i].set_xlabel("epoch")
    ax[i].set_ylabel("RMSE")
plt.tight_layout()
plt.savefig(f"subplots_plot_{search_param}.png", dpi=300)