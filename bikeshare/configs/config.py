CFGLog = {
    "data": {
        "path": "./data/raw/SeoulBikeData_cleaned_cols.csv",
        "X": [
            'hour', 'temp',
            'humidity', 'wind_speed', 
            'visibility', 'solar_rad',
            'rainfall', 'snowfall', 'seasons',
            'holiday', 'day', 'month'
        ],
        "y": "rented_bike_count",
        "test_size": 0.2,
        "random_state": 42
    },
    "decision_tree": {
        "criterion": "squared_error",
        "max_depth": 20,
        "max_features": None,
        "max_leaf_nodes": None,
        "min_samples_leaf": 4,
        "min_samples_split": 10,
        "splitter": "best"
    },
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 20,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "min_samples_split": 2
    },
    "gradient_boosting": {
        "n_estimators": 300,
        "max_depth": 7,
        "subsample": 0.8,
        "learning_rate": 0.1
    },
    "output": {
        "output_path": "./artifacts/",
        "dt_path": "dt_model/",
        "dt_model": "DecisionTree_2024-09-25_07-10-59.pkl",
        "rf_path": "rf_model/",
        "rf_model": "RandomForest_2024-09-25_07-11-02.pkl",
        "xgb_path": "xgb_model/",
        "xgb_model": "XGBoost_2024-09-25_07-11-04.pkl",
    }
}

