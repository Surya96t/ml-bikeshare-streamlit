from bikeshare.utils.config import Config
from bikeshare.configs.config import CFGLog
import os 
import pickle

class Inferrer:
    def __init__(self):
        self.config = Config.from_json(CFGLog)
        self.dt_saved_path = os.path.join(self.config.output.output_path, self.config.output.dt_model)
        with open(self.dt_saved_path, "rb") as f:
            self.dt_col_transformer, self.dt_model = pickle.load(f)
        
        self.rf_saved_path = os.path.join(self.config.output.output_path, self.config.output.rf_model)
        with open(self.rf_saved_path, "rb") as f:
            self.rf_col_transformer, self.rf_model = pickle.load(f)
        
        self.xgb_saved_path = os.path.join(self.config.output.output_path, self.config.output.xgb_model)
        with open(self.xgb_saved_path, "rb") as f:
            self.xgb_col_transformer, self.xgb_model = pickle.load(f)
    
    def dt_preprocess(self, new_data, col_transformer):
        return self.dt_col_transformer.transform(new_data)
    
    def rf_preprocess(self, new_data, col_transformer):
        return self.rf_col_transformer.transform(new_data)
    
    def xgb_preprocess(self, new_data, col_transformer):
        return self.xgb_col_transformer.transform(new_data)
    
    def dt_infer(self, new_data):
        """ Infer data using decision tree model """
        transformed_data = self.dt_col_transformer.transform(new_data)
        dt_prediction = self.dt_model.predict(transformed_data)
        print(f'Model in use: {self.dt_saved_path}')
        return dt_prediction
    
    def rf_infer(self, new_data):
        """ Infer data using random forest model """
        transformed_data = self.rf_col_transformer.transform(new_data)
        rf_prediction = self.rf_model.predict(transformed_data)
        print(f'Model in use: {self.rf_saved_path}')
        return rf_prediction
    
    def xgb_infer(self, new_data):
        """ Infer data using xgboost model """
        transformed_data = self.xgb_col_transformer.transform(new_data)
        xgb_prediction = self.xgb_model.predict(transformed_data)
        print(f'Model in use: {self.xgb_saved_path}')
        return xgb_prediction
    
    #### Possible alternative solution ####
    # def __init__(self):
    #     self.config = Config.from_json(CFGLog)
    #     self.dt_model = None
    #     self.rf_model = None
    #     self.xgb_model = None
    #     self.col_transformer = None
        
    # def load_models(self):
    #     """ Load models """
    #     dt_model_path = os.path.join(self.config.output_path, self.config.dt_model)
    #     rf_model_path = os.path.join(self.config.output_path, self.config.rf_model)
    #     xgb_model_path = os.path.join(self.config.output_path, self.config.xgb_model)
        
    #     self.dt_model = pickle.load(open(dt_model_path, "rb"))
    #     self.rf_model = pickle.load(open(rf_model_path, "rb"))
    #     self.xgb_model = pickle.load(open(xgb_model_path, "rb"))
        
    # def load_col_transformer(self):
    #     """ Load column transformer """
    #     col_transformer_path = os.path.join(self.config.output_path, "col_transformer.pkl")
    #     self.col_transformer = pickle.load(open(col_transformer_path, "rb"))
        
    # def infer(self, X):
    #     """ Infer """
    #     X = self.col_transformer.transform(X)
        
    #     dt_preds = self.dt_model.predict(X)
    #     rf_preds = self.rf_model.predict(X)
    #     xgb_preds = self.xgb_model.predict(X)
        
    #     return dt_preds, rf_preds, xgb_preds