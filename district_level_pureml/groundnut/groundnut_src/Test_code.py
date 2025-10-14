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
        self.districts=pd.read_csv(filepath_data_previous).iloc[:num_districts,0]#store district list
        self.test_yield_path = test_yield_path


    def model_test(self):
        modellstm = tf.keras.models.load_model(self.filepath_lstm) ###Loading lstm model from filepath
        #converting previous year's yield data into tensorflow arrays
        test_lstm=tf.convert_to_tensor(self.data_previous.values.reshape(self.num_districts,1,self.data_previous.shape[1]), dtype=tf.float32)
        test_pred=modellstm.predict(test_lstm)###prediction using lstm model
        ###Adding prediction to data variable as a new feature
        lstm_pred=pd.DataFrame(test_pred).reset_index(drop=True)
        lstm_pred.columns = ["Prediction"]
        self.data=pd.concat([self.data.reset_index(drop=True),lstm_pred],axis=1)
        ###Choosing clustering variables that are nitrogen, pH and prediction in last step.
        cluster_test_input=self.data[['nitrogen', 'phh2o',"Prediction"]]

        cols = ['nitrogen', 'phh2o', 'Prediction']
        unique_cols = list(dict.fromkeys(cols))  # Preserves order and removes duplicates
        cluster_test_input = self.data[unique_cols]
        
        J=self.J_value
        ###Feature file (txt) should be in the groundnut_data folder
        ftr_path = "../groundnut_data/" + self.ftr_name
        names=ast.literal_eval(open(ftr_path, "r", encoding="utf-8").read())## reading feature names
        indx=names
        ### selecting choice of features from data and splitting into training and validation
        test_data=self.data[indx]
        
        ### M-STAC model
        for j in range (J):
            ###Loading k-means model for j-th iteration
            model_kmeans=joblib.load(self.folder_mstac+'/'+str(j+1)+'_iteration_kmeans_model.pkl')
            print(cluster_test_input.head())
            print(model_kmeans)
            print(self.folder_mstac+'/'+str(j+1)+'_iteration_kmeans_model.pkl')
            cluster_test=model_kmeans.predict(cluster_test_input)### clustering model prediction of labels 
            
           
            polys=[]
            models_=[]
            for d in range (J-j):### Loading the models for d-th cluster
                polys.append(joblib.load(self.folder_mstac+'/'+str(j+1)+'_iteration_'+str(d+1)+'poly_transformer.pkl'))
                
                models_.append(joblib.load(self.folder_mstac+'/'+str(j+1)+'_iteration_'+str(d+1)+'cluster_ridge_model.pkl'))
            for n in range(cluster_test.shape[0]):###Loop runs for entire test data set. cluster_test.shape[0] gets the total number data points
                x_poly=polys[cluster_test[n]].transform(test_data.iloc[n:n+1,:])### polys[cluster_test[n]] chooses the model corresponding to the cluster label for the data point
                temp=models_[cluster_test[n]].predict(x_poly)### Stores the predicted error from the corresponding model
                
                test_pred[n]=float(temp[0])+test_pred[n]###updates prediction as= predicted error + previous stage prediction
                
           
            cluster_test_input["Prediction"]=test_pred ###Updating  prediction

            print(cluster_test_input.head())
        
        districts=self.districts
        
        folder = os.path.dirname(self.path_csv)

        # y_test = pd.read_csv(args.test_yield_data_path).iloc[:,2]
        # test_yield = self.test_yield.iloc[:,2]

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
            prediction_csv=pd.DataFrame(test_pred.reshape(-1,1),columns=['Predicted Yield'])
            prediction_csv=pd.concat([districts,prediction_csv],axis=1)
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
            plt.ylabel('Groundnut Yield (ton/ha)')
            plt.xlabel('Districts')

            # Set title with SMAPE and NMSE values
            plt.title(f'LSTM-MSTAC Prediction on Test Data', fontsize=12)

            # Save the figure
            plt.savefig(folder + '/' + 'Test_plot.jpg', format='jpg')

            with open("test_output.txt", "w") as ftxt:
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
            prediction_csv = pd.concat([districts,prediction_csv,],axis=1)
            prediction_csv.to_csv(self.path_csv, index=False)

            # Compute SMAPE and NMSE
            smape_score = smape(test_yield.values.reshape(self.num_districts,1),test_pred.reshape(self.num_districts,1))
            nmse_score = nmse(test_yield.values.reshape(self.num_districts,1),test_pred.reshape(self.num_districts,1))

            with open("test_output.txt", "w") as ftxt:
                print(f"*Testing*\nsMAPE: {np.round(smape_score,2)}% NMSE: {np.round(nmse_score,2)}%", file=ftxt)
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
            plt.ylabel('Groundnut Yield (ton/ha)')
            plt.xlabel('Districts')
            plt.legend()

            # Set title with SMAPE and NMSE values
            plt.title(f'LSTM-MSTAC Test Prediction sMAPE: {smape_score:.2f}%  |  NMSE: {nmse_score:.2f}%', fontsize=12)

            # Save the figure
            plt.savefig(folder + '/' + 'Test_plot.jpg', format='jpg')

        return folder        




