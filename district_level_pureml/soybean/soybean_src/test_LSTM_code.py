import tensorflow as tf
import pandas as pd
import joblib
from sklearn.preprocessing import PolynomialFeatures
import os
import sys
import warnings
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import ast
import numpy as np
warnings.filterwarnings("ignore")

class Test:
     ### Constructor definition
    def __init__(self, filepath_lstm,folder_mstac,filepath_data_previous,num_districts,path, test_data, J_value, ftr_name, test_yield_path):
        self.filepath_lstm=filepath_lstm###saved lstm model
        self.folder_mstac=folder_mstac###saved mstac model
        self.data_previous=pd.read_csv(filepath_data_previous).iloc[:,2:] ###Previous year's yield of the district under consideration along with 2 neighbouring districts (input data to lstm model) 
        self.num_districts=num_districts###Total number of districts available in dataset
        self.path_csv=path ### save  prediction in this path
        self.data = test_data### Exogeneous variables (weather, vegetation and soil data)
        self.J_value = J_value ### Number of iterations (set in main file)
        self.ftr_name = ftr_name ### Selected features saved in txt file in the same filepath as code (set in main file)
        self.districts = pd.read_csv(filepath_data_previous).iloc[:num_districts,0]#store district list
        self.test_yield_path = test_yield_path


    def model_test(self):

        modellstm = tf.keras.models.load_model(self.filepath_lstm) ###Loading lstm model from filepath
        #converting previous year's yield data into tensorflow arrays
        test_lstm=tf.convert_to_tensor(self.data_previous.values.reshape(self.num_districts,1,self.data_previous.shape[1]), dtype=tf.float32)
        test_pred=modellstm.predict(test_lstm)###prediction using lstm model
        
        districts=self.districts
        folder = os.path.dirname(self.path_csv)

        

        def smape(y_true, y_pred):
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            numerator = np.abs(y_true - y_pred)
            denominator = (np.abs(y_true) + np.abs(y_pred))
            smape_score = np.mean(numerator / denominator) * 100
            return smape_score

        def nmse(y_true, y_pred):
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            numerator = np.sum((y_true - y_pred) ** 2)
            denominator = np.sum(y_true ** 2)
            nmse_score = numerator / denominator * 100
            return nmse_score
        

        if not self.test_yield_path:

            ###Saving final test prediction as a csv file in self.path_csv
            prediction_csv = pd.DataFrame(test_pred.reshape(-1,1),columns=['Predicted Yield'])
            prediction_csv = pd.concat([districts,prediction_csv],axis=1)
            prediction_csv.to_csv(self.path_csv, index=False)


            districts = np.array(districts)
            test_pred = np.array(test_pred)

            # Plotting
            plt.figure(figsize=(10, 8))
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.3, top=0.9, left=0.1, right=0.9)
            plt.scatter(districts, test_pred, color='r', linewidth=2)
            plt.plot(districts, test_pred, linestyle='--', color='r', linewidth=2)
            plt.xticks(districts, districts, rotation=90, fontsize=8)
            plt.ylabel('Soybean Yield (ton/ha)')
            plt.xlabel('Districts')

            # Set title with SMAPE and NMSE values
            plt.title(f'LSTM Prediction on Test Data', fontsize=12)

            # Save the figure
            plt.savefig(folder + '/' + 'Test_plot.jpg', format='jpg')

            with open("test_LSTM_output.txt", "w") as ftxt:
                print(f" ", file=ftxt)
                # print(my_instance.model(), file=f)
                ftxt.close()


        else:
        
            test_yield = pd.read_csv(self.test_yield_path)
            # y_test = pd.read_csv(args.test_yield_data_path).iloc[:,2]
            test_yield = test_yield.iloc[:,2]

            ###Saving final test prediction as a csv file in self.path_csv
            prediction_csv = pd.DataFrame(zip(test_yield ,test_pred.reshape(-1,1)),
                                          columns=['True Yield','Predicted Yield'])
            prediction_csv = pd.concat([districts, prediction_csv],axis=1)
            prediction_csv.to_csv(self.path_csv, index=False)

            # Compute SMAPE and NMSE
            smape_score = smape(test_yield.values.reshape(self.num_districts,1),test_pred.reshape(self.num_districts,1))
            nmse_score = nmse(test_yield.values.reshape(self.num_districts,1),test_pred.reshape(self.num_districts,1))

            with open("test_LSTM_output.txt", "w") as ftxt:
                print(f"*Testing*\nsMAPE: {np.round(smape_score,2)}% NMSE: {np.round(nmse_score,2)}%\n", file=ftxt)
                # print(my_instance.model(), file=f)
                ftxt.close()

            districts = np.array(districts)
            test_pred = np.array(test_pred)
            test_yield = np.array(test_yield)

           # Plotting
            plt.figure(figsize=(10, 8))
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.3, top=0.9, left=0.1, right=0.9)

            # Scatter plots with labels for legend
            plt.scatter(districts, test_pred, color='r', linewidth=2, label='Predicted Yield')
            plt.scatter(districts, test_yield, color='b', linewidth=2, label='Actual Yield')

            # Line plots without labels (not added to legend)
            plt.plot(districts, test_pred, linestyle='--', color='r', linewidth=2)
            plt.plot(districts, test_yield, linestyle='--', color='b', linewidth=2)

            # Axis formatting
            plt.xticks(districts, districts, rotation=90, fontsize=8)
            plt.ylabel('Soybean Yield (ton/ha)')
            plt.xlabel('Districts')
            plt.legend()
            # plt.show()


            # Set title with SMAPE and NMSE values
            plt.title(f'LSTM Test Prediction sMAPE: {smape_score:.2f}%  |  NMSE: {nmse_score:.2f}%', fontsize=12)

            # Save the figure
            plt.savefig(folder + '/' + 'Test_plot.jpg', format='jpg')

        return folder        