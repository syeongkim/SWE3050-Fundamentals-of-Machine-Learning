import numpy as np
import pandas as pd
from models.LinearRegression import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from utils import RMSE, load_data, optimizer, load_data_with_feature_select, Inference2SolutionCsv

np.random.seed(2023)

# You can EDIT the hyperparameters below.
_epoch=5000
_batch_size=32
_lr = 0.001
_optim = 'Adam'              # Write one of SGD, or Momentum
_gamma = 0.2
_normalize = None            # Write one of ZScore, MinMax, or None(Default).
_threshold = 10
selected_feature = ["holiday", "temp", "clouds_all", "year", "month", "day", "hour", "snow_1h", "rain_1h"]

_dataset_name = 'Traffic'
target_feature = ["traffic_volume"]


# Data generation
train_data, test_data = load_data_with_feature_select(_dataset_name, _normalize, selected_feature+target_feature)
x_data, y_data = train_data[0], train_data[1]
test_x_data, test_y_data = test_data[0], test_data[1]


# Feature Engineering
temp_idx = selected_feature.index("temp")
hour_idx = selected_feature.index("hour")
year_idx = selected_feature.index("year")
month_idx = selected_feature.index("month")
day_idx = selected_feature.index("day")
snow_idx = selected_feature.index("snow_1h")
rain_idx = selected_feature.index("rain_1h")
cloud_idx = selected_feature.index("clouds_all")

# train data preprocessing
temp = x_data[:, temp_idx]
hours = x_data[:, hour_idx]
years = x_data[:, year_idx].astype(int)
months = x_data[:, month_idx].astype(int)
days = x_data[:, day_idx].astype(int)
snow_1h = x_data[:, snow_idx]
rain_1h = x_data[:, rain_idx]
clouds = x_data[:, cloud_idx]

temp_conditions = [
    (temp <= 272),
    ((272 < temp) & (temp <= 282)),
    ((282 < temp) & (temp <= 291)),
    (291 < temp)
]
temp_choices = [0, 1, 2, 3]
temp_default = -1

peak_hour_conditions = [
    ((1 <= hours) & (hours <= 4)),                 
    ((6 <= hours) & (hours <= 8)) | ((15 <= hours) & (hours <= 17))  
]
peak_hour_choices = [0, 2]
peak_hour_default = 1

temp_category = np.select(temp_conditions, temp_choices, default = temp_default).reshape(-1, 1)
is_peak_hour = np.select(peak_hour_conditions, peak_hour_choices, default=peak_hour_default).reshape(-1, 1)
dates = pd.to_datetime({'year': years, 'month': months, 'day': days})
is_weekday = (dates.dt.weekday).astype(bool).values.reshape(-1, 1)
is_weekend = (dates.dt.weekday >= 5).astype(bool).values.reshape(-1, 1)
hour_sin = np.sin(2 * np.pi * hours / 24).reshape(-1, 1)
hour_cos = np.cos(2 * np.pi * hours / 24).reshape(-1, 1)
is_snow = (snow_1h > 0).astype(bool).reshape(-1, 1)
is_rain = (rain_1h > 0.4).astype(bool).reshape(-1, 1)
is_bad_weather = (clouds > 49).astype(bool).reshape(-1, 1)

x_data_preprocessing = np.concatenate([x_data[:, :-1], temp_category, is_peak_hour, is_weekday, is_weekend, hour_sin, hour_cos, is_snow, is_rain, is_bad_weather], axis=1)
selected_feature += ["temp_category", "is_peak_hour", "is_weekday", "is_weekend", "hour_sin", "hour_cos", "is_snow", "is_rain", "is_bad_weather"]

delete_features = ["year", "snow_1h", "rain_1h", "month", "day", "clouds_all"]
cols_to_delete = [selected_feature.index(col) for col in delete_features]
x_data_preprocessing = np.delete(x_data_preprocessing, cols_to_delete, axis=1)

# test data preprocessing
test_temp = test_x_data[:, temp_idx]
test_hours = test_x_data[:, hour_idx]
test_years = test_x_data[:, year_idx].astype(int)
test_months = test_x_data[:, month_idx].astype(int)
test_days = test_x_data[:, day_idx].astype(int)
test_snow_1h = test_x_data[:, snow_idx]
test_rain_1h = test_x_data[:, rain_idx]
test_clouds = test_x_data[:, cloud_idx]

test_temp_conditions = [
    (test_temp <= 272),
    ((272 < test_temp) & (test_temp <= 281)),
    ((281 < test_temp) & (test_temp <= 291)),
    (291 < test_temp)
]
test_temp_choices = [0, 1, 2, 3]
test_temp_default = -1

test_peak_hour_conditions = [
    ((1 <= test_hours) & (test_hours <= 4)),                   # 조건 1: 0
    ((6 <= test_hours) & (test_hours <= 8)) | ((15 <= test_hours) & (test_hours <= 17))  # 조건 2: 2
]
test_peak_hour_choices = [0, 2]
test_peak_hour_default = 1

test_is_peak_hour = np.select(test_peak_hour_conditions, test_peak_hour_choices, default=test_peak_hour_default).reshape(-1, 1)
test_temp_category = np.select(test_temp_conditions, test_temp_choices, default = test_temp_default).reshape(-1, 1)
test_dates = pd.to_datetime({'year': test_years, 'month': test_months, 'day': test_days})
test_is_weekday = (test_dates.dt.weekday).astype(bool).values.reshape(-1, 1)
test_is_weekend = (test_dates.dt.weekday >= 5).astype(bool).values.reshape(-1, 1)
test_hour_sin = np.sin(2 * np.pi * test_hours / 24).reshape(-1, 1)
test_hour_cos = np.cos(2 * np.pi * test_hours / 24).reshape(-1, 1)
test_is_snow = (test_snow_1h > 0).astype(bool).reshape(-1, 1)
test_is_rain = (test_rain_1h > 0.4).astype(bool).reshape(-1, 1)
test_is_bad_weather = (test_clouds > 0).astype(bool).reshape(-1, 1)

test_x_data_preprocessing = np.concatenate([test_x_data[:, :-1], test_temp_category, test_is_peak_hour, test_is_weekday, test_is_weekend, test_hour_sin, test_hour_cos, test_is_rain, test_is_snow, test_is_bad_weather], axis=1)
test_x_data_preprocessing = np.delete(test_x_data_preprocessing, cols_to_delete, axis=1)

for col in delete_features:
   selected_feature.remove(col)


# 다중공선성 분석
feature_names = selected_feature 
X_df = pd.DataFrame(x_data_preprocessing, columns=feature_names)

vif_data = pd.DataFrame()
vif_data["Feature"] = X_df.columns
vif_data["VIF"] = [variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])]

print(vif_data)


# Build model
model = LinearRegression(num_features=x_data_preprocessing.shape[1])
optim = optimizer(_optim, _gamma, threshold=_threshold)
print('Initial weight: \n', model.W.reshape(-1))


# Solve
model.numerical_solution(x=x_data_preprocessing, y=y_data, epochs=_epoch, batch_size=_batch_size, lr=_lr, optim=optim, batch_gradient=False)

print('Trained weight: \n', model.W.reshape(-1))


# Inference in Train dataset
inference = model.eval(x_data_preprocessing)
error = RMSE(inference, y_data)
print('RMSE on Train Data : %.4f' % error)


# Inference in Test dataset & save inference csv
inference = model.eval(test_x_data_preprocessing)
output_file_name = "sample_submission.csv"
Inference2SolutionCsv(inference, output_file_name)