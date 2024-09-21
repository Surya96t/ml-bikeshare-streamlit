import datetime
import os
import pickle 

class ModelSaving(object):
    
    @staticmethod
    def get_current_timestamp():
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d_%H-%M-%S")
    
    @staticmethod 
    def save_model_with_timestamp(col_transformer, model, model_name, output_config):
        filename = model_name + "_" + ModelSaving.get_current_timestamp() + ".pkl"
        filepath = os.path.join(output_config.output_path, filename)
        with open(filepath, 'wb') as outputfile:
            pickle.dump((col_transformer, model), outputfile)
        
        return print("Saved column transformer and model to: ", filepath)