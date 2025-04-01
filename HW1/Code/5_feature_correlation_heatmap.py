import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from models.LinearRegression import LinearRegression

from utils import RMSE, load_data, optimizer, load_data_with_feature_select


np.random.seed(2023)



# You can EDIT the hyperparameters below.

_normalize = None           # Write one of ZScore, MinMax, or None(Default).

_selected_feature = ["holiday", "temp", "rain_1h", "snow_1h", "clouds_all", "year", "month", "day", "hour"]
_target_feature = ["traffic_volume"]
print(f"num of feature in X: {len(_selected_feature)-1}")

_dataset_name = 'Traffic' # Write one of Boston, Wine or Traffic



"""
Change GD to:
    True for numerical solution
    False for analytic solution
"""

# Data generation
train_data, _ = load_data_with_feature_select(_dataset_name, _normalize, _selected_feature+_target_feature)
x_data, y_data = train_data[0], train_data[1]

x_reshaped = x_data[:,:-1] # slice bias column
y_reshaped = y_data.reshape(-1,1)
combined_data = np.hstack((x_reshaped, y_data.reshape(-1,1)))

# 2. translate to DataFrame
df = pd.DataFrame(combined_data, columns=_selected_feature+_target_feature)

# 3. caculate correlations
correlation_matrix = df.corr()

matplotlib.use('Agg')  # Non-GUI 백엔드 사용


## Correlation plot
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.savefig("Correlation.png")
plt.close() 



## Scatter plot
fig, axes = plt.subplots(3, 5, figsize=(15, 8))  
fig.suptitle(f"Scatter Plots of {len(_selected_feature)} Features Against {_target_feature[0]}")

for i, ax in enumerate(axes.flat):
    if i == len(_selected_feature):
        break
    ax.scatter(x_data[:, i], y_data, alpha=0.5, s=0.1)
    ax.set_xlabel(f'Feature {_selected_feature[i]}')
    ax.set_ylabel(f'Feature {_target_feature[0]}')
    ax.set_title(f'Feature {_selected_feature[i]} vs. {_target_feature[0]}')
    

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("Scatter_Plots.png")
plt.close()