from NeuralNet_model import NeuralNetworkTimeSeries
from MSTAC import MSTAC
from Test_code import Test
import pandas as pd
import tensorflow as tf
import os
import argparse
from sklearn.preprocessing import StandardScaler
import pickle
import yaml
import tkinter as tk
from tkinter import ttk
import threading

class CodeRun:
    def __init__(self, args):
        self.args = args
        self.data = pd.read_csv(self.args.train_file_path).iloc[:,2:].reset_index(drop=True) #reading the data containing features
        self.yield_data = pd.read_csv(self.args.train_yield_path).iloc[:,2:3].reset_index(drop=True)# reading the yield values
        self.districts = pd.read_csv(self.args.train_yield_path).iloc[:,0].reset_index(drop=True)
        # splitting the data into training and validation sets
        self.train_start=0 # setting indexes for training start 
        self.train_end= self.args.train_years * self.args.num_districts# setting indexes for training end 
        self.val_start=self.train_end # setting indexes for validation start (1 year)
        self.val_end=self.train_end + self.args.num_districts # setting indexes for validation end (1 year)
        self.train_years=self.args.train_years # setting the user defined number of training years
        self.num_districts=self.args.num_districts# setting the user defined number of districts
        self.scaler = StandardScaler()# initializing a standard scaler to zero center the data
        self.test_yield_path = self.args.test_yield_data_path
        if self.args.pred_time == "1_month":
            self.ftr_name = "ftr_names_MP_1_month.txt"# reading the feature names used for training the LSTM model 
            self.ridge_models_path = self.args.load_models_path + 'Ridge_model_pureML_1month'# filepath for storing the ridge models
            self.J_value = 6#Setting the iteration value for M-STAC (model hyperparameter)
        elif self.args.pred_time == "2_months":
            self.ftr_name = "ftr_names_MP_2_months.txt"
            self.ridge_models_path = self.args.load_models_path + 'Ridge_model_pureML_2months'
            self.J_value = 11
        else:
            self.ftr_name = "ftr_names_MP_15_days.txt"
            self.ridge_models_path = self.args.load_models_path + 'Ridge_model_pureML_15days'
            self.J_value = 5
        
        self.data_path = self.args.train_file_path
        self.train_val_years =  pd.read_csv(self.args.train_yield_path).iloc[:,1].reset_index(drop=True)

    def train_neural_net(self):
        # creating a neural network (LSTM) model with all the initialized parameters and data
        my_instance=NeuralNetworkTimeSeries(self.districts,self.train_val_years,self.train_start,
                                            self.train_end,
                                            self.val_start,
                                            self.val_end,
                                            self.train_years,
                                            self.num_districts,
                                            self.yield_data,
                                            self.data,self.data_path,
                                            self.args.load_models_path,
                                            self.args.lstm_model_name)
        final_model = my_instance.model() # fits the training data in the model by running the training loop and returns the trained model
        return final_model

    def run_algo(self):

        def run_mstac_with_progress(mstac_instance):
            def run_training():
                # Run your MSTAC model training
                mstac_instance.model(progress_bar=progress_bar, status_label=status_label)

                # When done, update label and enable the button
                status_label.config(text="Training Completed", foreground='green')
                close_button.config(state=tk.NORMAL)

            def close_window():
                root.destroy()

            # Create the Tkinter GUI
            root = tk.Tk()
            root.title("Training Progress")
            root.geometry("400x150")

            tk.Label(root, text="MSTAC training...", font=("Arial", 12)).pack(pady=10)

            progress_bar = ttk.Progressbar(root, orient='horizontal', length=300, mode='determinate')
            progress_bar.pack(pady=10)
            progress_bar['maximum'] = mstac_instance.J_value

            status_label = tk.Label(root, text="Running...", font=("Arial", 10))
            status_label.pack(pady=5)

            close_button = tk.Button(root, text="Press to continue", command=close_window, state=tk.DISABLED)
            close_button.pack(pady=5)

            # Start training in a separate thread to keep GUI responsive
            threading.Thread(target=run_training, daemon=True).start()

            root.mainloop()

        df_train=pd.read_csv(self.args.district_data_path)# read the training data for mstac
        if self.args.scale_data:# if the user wants to scale the data, then scale the training data
            district_data=pd.DataFrame(self.scale(df_train.reset_index(drop=True).iloc[:,2:]),columns=df_train.columns[2:])# load the trained LSTM model
        else:
            district_data = df_train
        modellstm = tf.keras.models.load_model(self.args.load_models_path + self.args.lstm_model_name)
        alpha_= 1# Regularization value
        # run the mstac algorithm
        my_instance=MSTAC(modellstm,self.train_val_years,
                          self.train_start,
                          self.train_end,
                          self.val_start,
                          self.val_end,
                          self.train_years,
                          self.num_districts,
                          self.yield_data,
                          self.data,
                          district_data,self.data_path,
                          self.ridge_models_path,
                          alpha_,
                          self.J_value,
                          self.ftr_name,self.districts)
        run_mstac_with_progress(my_instance)
        # print("Train SMAPE NMSE Val SMAPE NMSE")
        # print(my_instance.model())

    def test(self):
        filepath_lstm= self.args.load_models_path + self.args.lstm_model_name
        df_test=pd.read_csv(self.args.test_data_path)# read the test data
        # if the user wants to scale the data, same scaler used for training is also used to scale the test data
        if self.args.scale_data:
            test_data = pd.DataFrame(self.scale(df_test.reset_index(drop=True).iloc[:, 2:]),columns=df_test.columns[2:])
        else:
            test_data = df_test
        # instantiates the test object
        my_instance=Test(filepath_lstm,
                         self.ridge_models_path,
                         self.args.prev_data_path,
                         self.num_districts,
                         self.args.test_pred_path,
                         test_data,
                         self.J_value,
                         self.ftr_name,
                         self.test_yield_path)
        my_instance.model_test()
        # runs the test loop and saves the test results
    def scale(self, data):
        scaler_name = self.args.pred_time
        scaler_path = f"../soybean_models/scaler_{scaler_name}.pkl"# initialize a name for saving the scaler model based on the prediction time-horizon
        if os.path.exists(scaler_path):# if the scaler model already exists, use that scaler
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler.fit(data)# otherwise fit the scaler to the data
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)# otherwise fit the scaler to the data
        scaled_data = self.scaler.transform(data)# use the scaler to transform the data
        return scaled_data# return the scaled data

def print_args(args):
    for arg, val in args._get_kwargs():
        print(f"{arg} = {val}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_config',
                        dest='config_file',
                        default="params_soybean.yaml",
                        help='The yaml configuration file')# creates a parser object that takes config file name
    args, unprocessed_args = parser.parse_known_args()
    if args.config_file:
        with open(args.config_file, 'r') as f:
            parser.set_defaults(**yaml.safe_load(f))# opens a yaml file with paramters and loads them into the parser object
    args = parser.parse_args(unprocessed_args)# opens a yaml file with paramters and loads them into the parser object
    return args# return the parser object with all fields and their values

## based on the argument input the corresponding function is executed
if __name__ == '__main__':
    args = parse_args()
    print_args(args)
    code_runner = CodeRun(args)    
    if args.train:
        code_runner.train_neural_net()
    if args.run_mstac:
        code_runner.run_algo()
    if args.test:
        code_runner.test() 
