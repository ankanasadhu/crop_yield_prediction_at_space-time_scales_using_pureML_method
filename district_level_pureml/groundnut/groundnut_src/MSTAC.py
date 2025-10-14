from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import tensorflow as tf
import joblib
import os
import sys
import warnings
import ast
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

class MSTAC :
    ### Constructor definition
    def __init__(self,modellstm,train_val_years,train_start,train_end,val_start,val_end,train_years,num_districts,yield_data,data_previous,data,data_path,file_path,alpha_, J_value, ftr_name,districts):
        self.modellstm=modellstm ###saved lstm model
        self.train_start=train_start ###starting index of training
        self.train_end=train_end ###Ending index of training
        self.val_start=val_start###Starting index of validation
        self.val_end=val_end###Ending index of validation
        self.train_years=train_years###Total number of years used for training
        self.num_districts=num_districts###Total number of districts available in dataset
        self.yield_data=yield_data###Yield data (target value of lstm model)
        self.data_previous=data_previous###Previous year's yield of the district under consideration along with 2 neighbouring districts (input data to lstm model)
        self.data=data ### Exogeneous variables (weather, vegetation and soil data)
        self.filepath=file_path ### Filepath to models
        self.alpha_=alpha_ ### regularization parameter (set in main file)
        self.J_value = J_value ### Number of iterations (set in main file)
        self.ftr_name = ftr_name ### Selected features saved in txt file in the same filepath as code (set in main file)
        self.districts = districts
        self.data_path = data_path
        self.train_val_years = train_val_years


    #Symmetric Mean Absolute Percentage Error calculation function
    def smape(self,y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred))
        smape_score = np.mean(numerator / denominator) * 100
        return smape_score
    
    def nmse(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        numerator = np.sum((y_true - y_pred) ** 2)
        denominator = np.sum(y_true ** 2)
        nmse_score = numerator / denominator *100
        return nmse_score
    
    ##Prediction using ridge regression model
    def prediction(self,a,degree,res,train_data,y_pred,data_fit):
        polynomial_features = PolynomialFeatures(degree=degree,include_bias=True) ### defining polynomial features transformation model
        
        x_poly =polynomial_features.fit_transform(train_data) ### fitting polynomial model on training data
        
        ridge = Ridge(alpha=a,fit_intercept=False) ### defining ridge regression model
        ridge.fit(x_poly, res) ### fitting ridge model on training data and residuals (error in previous stage prediction)
        ### data_fit has been used as a set to check the choice of degree of polynomial. Default: data_fit=training_data. Can also be set as: data_fit=validation_data
        x_poly =polynomial_features.transform(data_fit) ### transforming data_fit into polynomial features
        
        res_predict=ridge.predict(x_poly) ## predicting residuals for data_fit as input
        
        res_predict= res_predict.reshape(-1)
        
        temp=res_predict+y_pred ### corrected yield= predicted error(res_predict) + previous prediction(y_pred)
        
        return [temp,ridge,polynomial_features]
    ###Choosing the most suitable model
    def choose(self,a,f,res,train_data,y_val_pred,y_val,val_data):
        error=np.inf 
        degree_f=-1
        
        for degree in range (0,f,1):
            pred,ridge,poly=self.prediction(a,degree,res,train_data,y_val_pred,val_data) ##prediction function
            
            temp_error=mean_absolute_error(y_val,pred) ## mean absolute error calculation
            
            if (temp_error<error): ### retain the model that has less MAE
                degree_f=degree ## degree_f can be passed and the trend can be observed to analyse the model training process
                error=temp_error ## error update
                ridgef=ridge ## ridge model update
                polyf=poly ##polynomial feature model update
        return [ridgef,polyf]


    ###Model training M-STAC
    def model(self, progress_bar=None, status_label=None):

       
        ## Seed value set
        np.random.seed(42)
        tf.random.set_seed(42)

        ###Feature file (txt) should be in the groundnut_data folder
        ftr_path = "../groundnut_data/" + self.ftr_name
        names=ast.literal_eval(open(ftr_path, "r", encoding="utf-8").read()) ## reading feature names
        indx=names 
        ### selecting choice of features from data and splitting into training and validation
        train_data=self.data[indx].iloc[self.train_start:self.train_end,:] ##exogeneous variables
        val_data=self.data[indx].iloc[self.val_start:self.val_end,:] ##exogeneous variables
        y_train=self.yield_data.iloc[self.train_start:self.train_end]
        y_val=self.yield_data.iloc[self.val_start:self.val_end]
      
        #converting previous year's yield data into tensorflow arrays
        train_data_=tf.convert_to_tensor(self.data_previous.iloc[self.train_start:self.train_end,:].values.reshape(self.train_years*self.num_districts,1,self.data_previous.shape[1]), dtype=tf.float32)
        val_data_=tf.convert_to_tensor(self.data_previous.iloc[self.val_start:self.val_end,:].values.reshape(self.num_districts,1,self.data_previous.shape[1]), dtype=tf.float32)
       
        #combined validation and training set
        input_lstm=tf.convert_to_tensor(self.data_previous.iloc[self.train_start:self.val_end,:].values.reshape((self.train_years+1)*self.num_districts,1,self.data_previous.shape[1]), dtype=tf.float32)
        y_pred=self.modellstm(input_lstm)#prediction using saved lstm model

        ###Adding prediction to data variable as a new feature
        lstm_pred=pd.DataFrame(y_pred).reset_index(drop=True)
        lstm_pred.columns = ["Prediction"]
        self.data=pd.concat([self.data,lstm_pred],axis=1)

        ###Choosing clustering variables as nitrogen, pH and prediction in last step.
        cluster_train_input=self.data[['nitrogen', 'phh2o',"Prediction"]].iloc[self.train_start:self.train_end,:]
        cluster_val_input=self.data[['nitrogen',  'phh2o', "Prediction"]].iloc[self.val_start:self.val_end,:]
       
        #calculating error in prediction for entire dataset
        res=np.array(self.yield_data).reshape(self.val_end,1)-np.array(y_pred).reshape(self.val_end,1)
        #storing only training error separately for training purpose 
        res_train=res[self.train_start:self.train_end]
        ##Total number of iterations
        J=self.J_value
        f=2 ##choice of model order=f-1 (chosen models are linear but f can be changed to try higher order models)
        a=self.alpha_ ### Regularization value
        ### prediction from lstm model (base model)
        val_pred = self.modellstm.predict(val_data_,verbose=False).reshape(-1)
        train_pred= self.modellstm.predict(train_data_,verbose=False).reshape(-1)

        ### M-STAC model training and validation
        for j in range(J):

            if progress_bar is not None:
                progress_bar['value'] = j + 1
                progress_bar.update_idletasks()
            if status_label is not None:
                status_label.config(text=f"Running iteration {j+1} of {J}")

            ### Clustering the dataset on nitrogen, pH and last stage prediction.
            kmeans_ = KMeans(n_clusters=(J-j),init='k-means++',n_init='auto',random_state= 42)  

            ### clustering model training and storing the labels
            cluster_train= kmeans_.fit(cluster_train_input).labels_
            ### clustering model prediction of labels on validation set
            cluster_val=kmeans_.predict(cluster_val_input)
            ### storing clustering model
            joblib.dump(kmeans_, self.filepath+'/'+str(j+1)+'_iteration_kmeans_model.pkl')
            print(self.filepath+'/'+str(j+1)+'_iteration_kmeans_model.pkl')
            models_=[]
            polys=[]

            ### Creating an ensemble models to predict error in previous stage predictions. 
            ### Training the same number of models as clusters. 
            for d in range (J-j):
                ### Separating the d-th cluster label data points
                training_data=train_data[cluster_train==d]
                
                res_data=res_train[cluster_train==d] ## res_data stores the residuals or error in training data for d-th label
                yield_train=y_train[cluster_train==d]
                yield_train_pred=train_pred[cluster_train==d] ## stores the prediction for d-th label in training data in the previous stage
                
                ### training best choice model
                model,poly=self.choose(a,f,res_data,training_data,yield_train_pred,yield_train,training_data)
                ### appending models for each cluster
                models_.append(model)
                polys.append(poly)
                ### saving the models to filepath
                joblib.dump(model, self.filepath+'/'+str(j+1)+'_iteration_'+str(d+1)+'cluster_ridge_model.pkl')
                joblib.dump(poly, self.filepath+'/'+str(j+1)+'_iteration_'+str(d+1)+'poly_transformer.pkl')
                
            ###Validation prediction
            for n in range(cluster_val.shape[0]):###Loop runs for entire validation set. cluster_val.shape[0] gets the total number val data points
                x_poly=polys[cluster_val[n]].transform(val_data.iloc[n:n+1,:]) ### polys[cluster_val[n]] chooses the model corresponding to the cluster label for the data point
                temp=models_[cluster_val[n]].predict(x_poly) ### Stores the predicted error from the corresponding model
                
                val_pred[n]=float(temp[0])+val_pred[n] ###updates prediction as= predicted error + previous stage prediction
            
            smape_val=(self.smape(y_val.values.reshape(self.num_districts,1),val_pred.reshape(self.num_districts,1))) ###Calculating sMAPE for validation set
            nmse_val=(self.nmse(y_val.values.reshape(self.num_districts,1),val_pred.reshape(self.num_districts,1)))

            districts = self.districts.iloc[self.val_start:self.val_end].values.reshape(self.num_districts,1)
            districts = np.array(districts)
            districts = [d[0] for d in districts]

            folder = os.path.dirname(self.data_path)

            plt.figure(figsize=(10, 8))
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.3, top=0.9, left=0.1, right=0.9)

            # Scatter plots with labels for legend
            plt.scatter(districts, val_pred.reshape(self.num_districts,1), color='r', linewidth=2, label='Predicted Yield')
            plt.scatter(districts, y_val.values.reshape(self.num_districts,1), color='b', linewidth=2, label='Actual Yield')

            # Line plots without labels (not added to legend)
            plt.plot(districts, val_pred.reshape(self.num_districts,1), linestyle='--', color='r', linewidth=2)
            plt.plot(districts, y_val.values.reshape(self.num_districts,1), linestyle='--', color='b', linewidth=2)

           

            # Axis formatting
            plt.xticks(districts, districts, rotation=90, fontsize=8)
            plt.ylabel('Groundnut Yield (ton/ha)')
            plt.xlabel('Districts')
            plt.legend()
            # plt.show()

             # Set title with SMAPE and NMSE values
            plt.title(f'LSTM-MSTAC Validation sMAPE: {smape_val:.2f}%  |  NMSE: {nmse_val:.2f}%', fontsize=12)

            plt.savefig(folder + '/' + 'Train_plot.jpg', format='jpg')




            
            #### Traininig prediction
            for n in range(cluster_train.shape[0]): ###Loop runs for entire training set. cluster_train.shape[0] gets the total number training data points
                x_poly=polys[cluster_train[n]].transform(train_data.iloc[n:n+1,:])### polys[cluster_train[n]] chooses the model corresponding to the cluster label for the data point
                temp=models_[cluster_train[n]].predict(x_poly)### Stores the predicted error from the corresponding model
            
                train_pred[n]=float(temp[0])+train_pred[n] ###updates prediction as= predicted error + previous stage prediction
            
            smape_train=(self.smape(y_train.values.reshape(self.train_years*self.num_districts),train_pred.reshape(self.train_years*self.num_districts)))###Calculating sMAPE for training set
            nmse_train=(self.nmse(y_train.values.reshape(self.train_years*self.num_districts),train_pred.reshape(self.train_years*self.num_districts)))
          
            res_train=np.array(y_train).reshape(self.train_end,1)-np.array(train_pred).reshape(self.train_end,1) ### Updating residuals (error) in the new training prediction
            ### Please note the true validation yield is not used here for training
            cluster_train_input["Prediction"]=train_pred ###Updating  prediction
            cluster_val_input["Prediction"]=val_pred###Updating  prediction

            with open("mstac_output.txt", "w") as ftxt:
                print(f"*MSTAC*\nTraining sMAPE: {np.round(smape_train,2)}% NMSE: {np.round(nmse_train,2)}% \nValidation sMAPE: {np.round(smape_val,2)}% NMSE: {np.round(nmse_val,2)}%", file=ftxt)
                # print(my_instance.model(), file=f)
                ftxt.close()

            # Create the district list for training and validation
            districts_train = districts * self.train_years  # repeats the list for training years
            districts_val = districts                       # one repetition for validation
            districts_full = districts_train + districts_val  # total list

            # Sanity check: match length
            assert len(districts_full) == len(y_train) + len(y_val), "District length mismatch"

            # Reshape the arrays
            y_true = np.concatenate([y_train.values.reshape(self.train_years*self.num_districts,1),y_val.values.reshape(self.num_districts,1)], axis=0)
            y_pred = np.concatenate([train_pred.reshape(self.train_years*self.num_districts,1),val_pred.reshape(self.num_districts,1)], axis=0)

            # Combine all data
            df = pd.DataFrame({
                "District": districts_full,
                "Years": self.train_val_years,
                "Actual": y_true.flatten(),
                "Predicted": y_pred.flatten()
            })

            # Save to CSV
            df.to_csv("train_valid_prediction.csv", index=False)

        

        return [smape_train,nmse_train, smape_val, nmse_val]### Returning validation and training sMAPE 
           
       
