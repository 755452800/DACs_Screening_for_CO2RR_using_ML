import utils
from sklearn.svm import SVR


def model():
    SVR_model = SVR(kernel="rbf", gamma=0.6, C=1.8, epsilon=0.01)
    SVR_model.fit(x_train, y_train)
    y_train_pre = SVR_model.predict(x_train)
    y_test_pre = SVR_model.predict(x_test)
    utils.plot_scatter(y_train, y_train_pre, y_test, y_test_pre)
    rmse, r2, rmse_t, r2_t = utils.metrics(y_train, y_train_pre, y_test, y_test_pre)
    print(f"rmse is {rmse:.3f}, r2 is {r2:.3f}, rmset is {rmse_t:.3f}, r2t is {r2_t:.3f}")
    print("Predicted results: {}".format(list(SVR_model.predict(x_test))))
    print("DFT results: {}".format(list(y_test)))
    return SVR_model


if __name__ == "__main__":
    OPTIMAL_G = 0.0812
    x_train, x_test, y_train, y_test, scaler, features = utils.build_dataset("../data/training_set_100.csv")
    model = model()
    x_data, samples = utils.load_data_predicting("../data/sample_properties_all_406.csv", scaler)
    y_pre = model.predict(x_data)
    samples["Predicted value"] = y_pre
    samples["Difference from optimal G"] = abs(samples["Predicted value"] - OPTIMAL_G)
    potential_candidates = samples.sort_values(by=["Difference from optimal G"], ascending=True, inplace=False)
    print(potential_candidates.head(20))
    samples.to_csv("../data/predicted_result_SVR.csv", index=False)
