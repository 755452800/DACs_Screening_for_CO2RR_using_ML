import torch
import utils
from torch import nn
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt


def build_nn_dataset(dataset_path):
    x_train, x_test, y_train, y_test, scaler, features = utils.build_dataset(dataset_path)
    x_train = x_train.astype(float)
    y_train = y_train.astype(float)
    x_test = x_test.astype(float)
    y_test = y_test.astype(float)
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    y_train = torch.unsqueeze(y_train, 1)
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()
    y_test = torch.unsqueeze(y_test, 1)
    return x_train, x_test, y_train, y_test, scaler, features


class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


def plot_training_rst(loss_train, loss_test):
    plt.figure(figsize=(10, 9))
    plt.plot(loss_train, "-o", linewidth=3, markersize=1)
    plt.plot(loss_test, "-o", linewidth=3, markersize=1)
    plt.ylim([0, 10])
    plt.legend(["Train loss", "Test loss"], fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("Epoch", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.grid(which="both", linestyle="--", linewidth=2, axis="y")
    plt.show()


def train_func(net, train_data, lr, epochs):
    cal_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train_loss_epoch_record, test_loss_epoch_record = [], []
    test_loss_flag = 1
    model_sav = net
    for epoch_num in range(epochs):
        # ============== Training process ==============
        running_loss, batch_num = 0, 0
        for batch_id, (x, y) in enumerate(train_data):
            batch_num = batch_id
            output = net(x)
            loss = cal_loss(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # ============== Testing process ==============
        testing_loss = 0
        with torch.no_grad():
            for x, y in test_data:
                output = net(x)
                loss = cal_loss(output, y)
                testing_loss += loss.item()

        running_loss_batch_mean = running_loss / (batch_num + 1)
        testing_loss_mean = testing_loss / (len(test_data))

        train_loss_epoch_record.append(running_loss_batch_mean)
        test_loss_epoch_record.append(testing_loss_mean)
        if testing_loss_mean <= test_loss_flag:
            test_loss_flag = testing_loss_mean
            model_sav = model
            print(f"Epoch: {epoch_num}, now the best models's performace:\n"
                  f"training loss: {running_loss_batch_mean:.3f}, testing loss: {testing_loss_mean:.3f}")
    plot_training_rst(train_loss_epoch_record, test_loss_epoch_record)
    return model_sav


if __name__ == "__main__":
    # ============== Set random seed and model settings ==============
    torch.manual_seed(0)
    epochs, init_lr, batch_size = 2000, 1e-3, 32  # batch_size 32 will generate very smooth loss curve

    # ============== Load data ==============
    x_train, x_test, y_train, y_test, scaler, features = build_nn_dataset("../data/training_set_100.csv")
    train_data = Data.TensorDataset(x_train, y_train)
    train_data = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_data = Data.TensorDataset(x_test, y_test)
    test_data = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

    # ============== Instantiate model ==============
    n_feature = x_train.shape[1]
    n_output = y_train.shape[1]
    n_hidden = 44
    model = Net(n_feature, n_hidden, n_output)

    # ============== Training and Testing ==============
    model = train_func(model, train_data, init_lr, epochs)

    # ============== Model Performace ==============
    y_train = y_train.numpy()
    y_train_pre = model(x_train)
    y_train_pre = y_train_pre.detach().numpy()
    y_test = y_test.numpy()
    y_test_pre = model(x_test)
    y_test_pre = y_test_pre.detach().numpy()
    utils.plot_scatter(y_train, y_train_pre, y_test, y_test_pre)
    rmse, r2, rmse_t, r2_t = utils.metrics(y_train, y_train_pre, y_test, y_test_pre)
    print(f"RMSE is {rmse:.3f}, R2 is {r2:.3f}, RMSE on test is {rmse_t:.3f}, R2 on test is {r2_t:.3f}")

    # ============== Save results ==============
    OPTIMAL_G = 0.0812
    x_data, samples = utils.load_data_predicting("../data/sample_properties_all_406.csv", scaler)
    x_data = x_data.astype(float)
    x_data = torch.from_numpy(x_data).float()
    y_pre = model(x_data)
    y_pre = y_pre.detach().numpy()
    samples["Predicted value"] = y_pre
    samples["Difference from optimal G"] = abs(samples["Predicted value"] - OPTIMAL_G)
    potential_candidates = samples.sort_values(by=["Difference from optimal G"], ascending=True, inplace=False)
    print(potential_candidates.head(20))
    samples.to_csv("../data/predicted_result_FNN.csv", index=False)
