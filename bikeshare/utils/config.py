import json

class Config:
    
    
    def __init__(self, data, decision_tree, random_forest, gradient_boosting, output):
        self.data = data
        self.decision_tree = decision_tree
        self.random_forest = random_forest
        self.gradient_boosting = gradient_boosting
        self.output = output
        
    @classmethod # class method to load the configuration from a JSON file
    def from_json(cls, cfg):
        """ Creates config from json file """
        params = json.loads(json.dumps(cfg), object_hook=HelperDict)
        
        return cls(params.data, params.decision_tree, params.random_forest, params.gradient_boosting, params.output)
    
    
class HelperDict(object):
    """ Helper class to convert dictionary into Python object """
    def __init__(self, dict_):
        self.__dict__.update(dict_)