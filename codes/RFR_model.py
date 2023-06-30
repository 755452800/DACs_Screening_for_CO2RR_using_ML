import utils
from sklearn.ensemble import RandomForestRegressor as RFR


def model(features):
    RFR_model = RFR(n_estimators=60, oob_score=True, max_depth=6, random_state=0)
    RFR_model.fit(x_train, y_train)
    y_train_pre = RFR_model.predict(x_train)
    y_test_pre = RFR_model.predict(x_test)
    utils.plot_scatter(y_train, y_train_pre, y_test, y_test_pre)
    rmse, r2, rmse_t, r2_t = utils.metrics(y_train, y_train_pre, y_test, y_test_pre)
    print(f"RMSE is {rmse:.3f}, R2 is {r2:.3f}, RMSE on test is {rmse_t:.3f}, R2 on test is {r2_t:.3f}")
    utils.plot_feature_importance(RFR_model.feature_importances_, features, "RFR")
    print("Predicted results: {}".format(list(RFR_model.predict(x_test))))
    print("DFT results: {}".format(list(y_test)))
    return RFR_model


if __name__ == "__main__":
    OPTIMAL_G = 0.0812
    x_train, x_test, y_train, y_test, scaler, features = utils.build_dataset("../data/training_set_100.csv")
    model = model(features)
    x_data, samples = utils.load_data_predicting("../data/sample_properties_all_406.csv", scaler)
    y_pre = model.predict(x_data)
    samples["Predicted value"] = y_pre
    samples["Difference from optimal G"] = abs(samples["Predicted value"] - OPTIMAL_G)
    potential_candidates = samples.sort_values(by=["Difference from optimal G"], ascending=True, inplace=False)
    print(potential_candidates.head(20))
    samples.to_csv("../data/predicted_result_RFR.csv", index=False)
