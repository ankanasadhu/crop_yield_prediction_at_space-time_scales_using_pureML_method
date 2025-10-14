from MSTAC import MSTAC
from Test import Test
import pandas as pd
import os
import argparse
from sklearn.preprocessing import StandardScaler
import pickle
import yaml


class CodeRun:
    def __init__(self, args):
        self.args = args
        self.data = pd.read_csv(self.args.train_file_path) #reading the data containing features
        self.yield_data = pd.read_csv(self.args.train_yield_path) # reading the yield values
        self.scaler = StandardScaler() # initializing a standard scaler to zero center the data
        self.ridge_model_path = self.args.load_models_path
        self.test=pd.read_csv(self.args.test_data_path) 
        self.test_path=self.args.test_pred_path
        self.category=self.args.category
      
        if self.args.pred_time=='15_days':
            self.ftr_name = "../data/ftr_names_groundnut_15d.txt"
            self.J_value = 3 #Setting the iteration value for M-STAC (model hyperparameter)
            self.aplha= 5
        elif  self.args.pred_time=='1_month':
            self.ftr_name = "../data/ftr_names_groundnut_1m.txt"
            self.J_value = 3 #Setting the iteration value for M-STAC (model hyperparameter)
            self.aplha= 1
        elif self.args.pred_time=='2_months':
            self.ftr_name = "../data/ftr_names_groundnut_2m.txt"
            self.J_value = 3 #Setting the iteration value for M-STAC (model hyperparameter)
            self.aplha= 1

    
    def run_algo(self):
        if self.args.scale_data:# if the user wants to scale the data, then scale the training data
            self.data.iloc[:,4:]=pd.DataFrame(self.scale(self.data.reset_index(drop=True).iloc[:,4:]),columns=self.data.columns[4:])# load the trained LSTM model
        my_instance=MSTAC(self.data,
                          self.yield_data,
                          self.ridge_model_path,
                          self.J_value,
                          self.aplha,
                          self.ftr_name,self.category)
        
        my_instance.model()

    def test_fn(self):
        
        # read the test data
        # if the user wants to scale the data, same scaler used for training is also used to scale the test data
        if self.args.scale_data:
            self.test.iloc[:, 4:] = pd.DataFrame(self.scale(self.test.reset_index(drop=True).iloc[:, 4:]),columns=self.test.columns[4:])
        
        # instantiates the test object
        my_instance=Test( self.test,self.J_value,self.ftr_name,self.test_path,self.ridge_model_path,self.category)
        # runs the test loop and saves the test results
        my_instance.model_test()
    
    def scale(self, data):
        scaler_name = self.args.pred_time
        scaler_path = self.ridge_model_path+f"/scaler_{scaler_name}.pkl" # initialize a name for saving the scaler model based on the prediction time-horizon
        if os.path.exists(scaler_path): # if the scaler model already exists, use that scaler
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler.fit(data) # otherwise fit the scaler to the data
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f) # save the scaler object for the specified prediction time
        scaled_data = self.scaler.transform(data) # use the scaler to transform the data
        return scaled_data # return the scaled data

def print_args(args):
    for arg, val in args._get_kwargs():
        print(f"{arg} = {val}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_config',
                        dest='config_file',
                        default="params.yaml",
                        help='The yaml configuration file') # creates a parser object that takes config file name
    args, unprocessed_args = parser.parse_known_args() 
    if args.config_file:
        with open(args.config_file, 'r') as f:
            parser.set_defaults(**yaml.safe_load(f)) # opens a yaml file with paramters and loads them into the parser object
    args = parser.parse_args(unprocessed_args) # the parser object parses all the values for the fields and stores them
    return args # return the parser object with all fields and their values

## based on the argument input the corresponding function is executed
if __name__ == '__main__':
    args = parse_args()
    print_args(args)
    code_runner = CodeRun(args)    
    if args.run_mstac:
        code_runner.run_algo()
    if args.test:
        code_runner.test_fn() 