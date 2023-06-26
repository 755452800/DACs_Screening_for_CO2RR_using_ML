import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score

matplotlib.rcParams["font.sans-serif"] = "Times New Roman"


def load_data_training(file_path):
    data = pd.read_csv(file_path)
    x_data = data[list(data.columns)].drop(["name", "G"], axis=1)
    y_data = data.pop("G")
    return x_data, y_data


def load_data_predicting(file_path, scaler):
    data = pd.read_csv(file_path)
    x_data = data[list(data.columns)].drop(["name"], axis=1)
    x_name = pd.DataFrame(data["name"])
    x_data_pre = scaler.transform(x_data)  # use the same MinMax scaler from training set
    return x_data_pre, x_name


def build_dataset(file_path):
    x_data, y_data = load_data_training(file_path)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=10)
    features = x_train.columns
    mm_scaler = MinMaxScaler()
    x_train = mm_scaler.fit_transform(x_train)
    x_test = mm_scaler.transform(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return x_train, x_test, y_train, y_test, mm_scaler, features


def plot_scatter(y_train, y_train_pre, y_test, y_test_pre):
    plt.scatter(y_train, y_train_pre, c="#ff7f0e")
    plt.scatter(y_test, y_test_pre, c="#1f77b4")
    plt.legend(["Training set", "Test set"])
    plt.plot(plt.xlim(), plt.ylim(), ls="--", color="black")
    plt.show()


def save_data_for_plotting(y_train, y_train_pre, y_test, y_test_pre, model):
    y_train, y_train_pre = y_train.tolist(), y_train_pre.tolist()
    y_test, y_test_pre = y_test.tolist(), y_test_pre.tolist()
    extend_len = len(y_train) - len(y_test)
    y_test.extend([np.nan] * extend_len)
    y_test_pre.extend([np.nan] * extend_len)
    data = {"y_train": y_train, "y_train_pre": y_train_pre, "y_test": y_test, "y_test_pre": y_test_pre}
    df = pd.DataFrame(data)
    df.to_csv(f"../data/{model}-data.csv", index=False)


def metrics(y_train, y_train_pre, y_test, y_test_pre):
    rmse = np.sqrt(mse(y_train, y_train_pre))
    r2 = r2_score(y_train, y_train_pre)
    rmse_t = np.sqrt(mse(y_test, y_test_pre))
    r2_t = r2_score(y_test, y_test_pre)
    return rmse, r2, rmse_t, r2_t


def plot_feature_importance(importance, names, model_type):
    n_largest = 6
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    data = {"feature_names": feature_names, "feature_importance": feature_importance}
    fi_df = pd.DataFrame(data)
    fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)
    fi_df = fi_df.iloc[:n_largest, :]
    plt.figure(figsize=(10, 8), dpi=600)
    sns.barplot(x=fi_df["feature_importance"], y=fi_df["feature_names"])
    plt.title(f"{model_type} feature importance")
    plt.xlabel("Feature Importance")
    plt.show()
