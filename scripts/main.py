# Main executing script for the program

from bikeshare.configs.config import CFGLog
from bikeshare.model.bikeshare_model import BikeshareDecisionTree, BikeshareRandomForest, BikeshareGradientBoosting
from bikeshare.executor.inferrer import Inferrer


def run():
    # Load the configuration file
    config = CFGLog
    print("Configuration file loaded")
    
    # Decision Tree model
    dt_model = BikeshareDecisionTree(config)
    dt_model.load_data()
    dt_model.build()
    dt_model.train()
    dt_model.evaluate()
    dt_model.export_model()
    
    # Random Forest model
    rf_model = BikeshareRandomForest(config)
    rf_model.load_data()
    rf_model.build()
    rf_model.train()
    rf_model.evaluate()
    rf_model.export_model()
    
    # Gradient Boosting model
    xgb_model = BikeshareGradientBoosting(config)
    xgb_model.load_data()
    xgb_model.build()
    xgb_model.train()
    xgb_model.evaluate()
    xgb_model.export_model()
    

if __name__ == "__main__":
    run()
