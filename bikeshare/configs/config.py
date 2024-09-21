CFGLog = {
    "data": {
        "path": "./data/SeoulBikeData.csv",
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
        "crierion": "squared_error",
        "max_depth": 20,
        "max_features": None,
        "max_leaf_nodes": None,
        "min_samples_leaf": 2,
        "min_samples_split": 10,
        "splitter": "best"
    },
    "random_forest": {
        "n_estimators": 300,
        "max_depth": 30,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "min_samples_split": 2
    },
    "gradient_boosting": {
        "n_estimators": 300,
        "max_depth": None,
        "max_features": "auto",
        "min_samples_leaf": 1,
        "min_samples_split": 2
    },
    "output": {
        "output_path": "./artifacts/",
        "dt_model": "dt_model.pkl",
        "rf_model": "rf_model.pkl",
        "xgb_model": "xgb_model.pkl",
    }
}

