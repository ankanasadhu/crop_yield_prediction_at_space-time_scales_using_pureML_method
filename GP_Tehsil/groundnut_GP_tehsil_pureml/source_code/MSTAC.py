from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import warnings
# import openpyxl as op
import joblib
import ast
import matplotlib.pyplot as plt

class MSTAC:
    def __init__(self, train,yield_train,filepath,J,alpha,ftr_names,category):
        self.train=train
        self.yield_train=yield_train
        self.filepath=filepath
        self.J=J
        self.a=alpha
        self.ftr_names=ftr_names
        self.category=category
    def smape(self,y_true, y_pred):

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred))

        smape_score = np.mean(numerator / denominator) * 100

        return smape_score
    
    def prediction(self, a,degree,res,train_data,y_pred,data):
   
        polynomial_features = PolynomialFeatures(degree=degree,include_bias=True)
        x_poly =polynomial_features.fit_transform(train_data)
        ridge = Ridge(alpha=a,fit_intercept=False)
        ridge.fit(x_poly, res)
    
        x_poly =polynomial_features.transform(data)
        res_predict=ridge.predict(x_poly)
        res_predict= res_predict.reshape(-1)
        temp=res_predict+y_pred
        
        return [temp,ridge,polynomial_features]
    
    def choose(self,a,f,res,train_data,y_val_pred,y_val,val_data):
    
        mape=np.inf 
        degree_f=-1
        for degree in range (0,f,1):
            pred,ridge,poly= self.prediction(a,degree,res,train_data,y_val_pred,val_data)
            
            temp_mae=mean_absolute_error(y_val,pred)
            
            if (temp_mae<mape):
                mape=temp_mae
                degree_f=degree
                ridgef=ridge
                polyf=poly
        return [degree_f,ridgef,polyf]
    
    def model(self, progress_bar=None, status_label=None):
        warnings.filterwarnings("ignore")
        np.random.seed(42)

   
        indx=ast.literal_eval(open(self.ftr_names, "r", encoding="utf-8").read())


        data=self.train
        yield_data=self.yield_train['yield (ton/ha)']
        train_start=0
        train_end=self.train.shape[0]
        ridge = Ridge(alpha=2,fit_intercept=False)
        ridge.fit(self.train.iloc[:,4:],yield_data)
        y_pred=ridge.predict(data.iloc[:,4:])
        joblib.dump(ridge , self.filepath+'/'+'Base linear model')

        lr_pred=pd.DataFrame(y_pred).reset_index(drop=True)
        lr_pred.columns = ["Prediction"]
        data=pd.concat([data,lr_pred],axis=1)


        cluster_train_input= data[['nitrogen', 'phh2o',"Prediction"]].iloc[ train_start: train_end,:]

        res=np.array( yield_data).reshape( train_end,1)-np.array(y_pred).reshape( train_end,1)
        res_train=res[ train_start: train_end]


        J=self.J
        f=2
        a=self.a


        train_pred= ridge.predict( data.iloc[ train_start: train_end,4:-1]).reshape(-1)
        train=self.train[indx]
        
       

        for j in range(J):
            seed_value = 42
            np.random.seed(seed_value)

            if progress_bar is not None:
                progress_bar['value'] = j + 1
                progress_bar.update_idletasks()
            if status_label is not None:
                status_label.config(text=f"Running iteration {j+1} of {J}")



            kmeans_ = KMeans(n_clusters=(J-j),init='k-means++',n_init='auto',random_state= 42)  
            cluster_train= kmeans_.fit(cluster_train_input).labels_
            joblib.dump(kmeans_,  self.filepath+'/'+str(j+1)+'_iteration_kmeans_model.pkl')
            models_=[]
            polys=[]

            for d in range (J-j):
                
                training_data=train[cluster_train==d]
                
                res_data=res_train[cluster_train==d]
                training_yield=yield_data[cluster_train==d]
                yield_train_pred= train_pred[cluster_train==d]
            
            
                degree,model,poly= self.choose(a,f,res_data,training_data,yield_train_pred,training_yield,training_data)
                models_.append(model)
                polys.append(poly)
                
                joblib.dump(model,  self.filepath+'/'+str(j+1)+'_iteration_'+str(d+1)+'cluster_ridge_model.pkl')
                joblib.dump(poly,  self.filepath+'/'+str(j+1)+'_iteration_'+str(d+1)+'poly_transformer.pkl')
                
            
            
            for n in range(cluster_train.shape[0]):
                x_poly=polys[cluster_train[n]].transform(train.iloc[n:n+1,:])
                temp=models_[cluster_train[n]].predict(x_poly)
            
                train_pred[n]=float(temp[0])+train_pred[n]
                
            

    
        
            res_train=np.array(yield_data).reshape( train_end,1)-np.array(train_pred).reshape( train_end,1)
            
            cluster_train_input["Prediction"]=np.abs(train_pred)

        final_train=pd.concat([train.reset_index(drop=True),cluster_train_input["Prediction"],self.yield_train],axis=1)
        final_train=final_train[[self.category,'Prediction','yield (ton/ha)']]
        tr=final_train.groupby(self.category).mean().reset_index()
        nmse=mean_squared_error(tr["yield (ton/ha)"],tr["Prediction"])/np.mean(tr["yield (ton/ha)"])
        sMAPE=self.smape(tr["yield (ton/ha)"],tr["Prediction"])
        print("NMSE:",nmse,"sMAPE:",sMAPE)

        tr.to_csv("train_valid_prediction.csv", index=False)
        tr_plot = tr.reset_index()
        category_vector = np.array(tr_plot[self.category])

        with open("mstac_output.txt", "w") as ftxt:
                print(f"*MSTAC*\nTraining sMAPE: {np.round(sMAPE,2)}% NMSE: {np.round(nmse,2)}%", file=ftxt)
                # print(my_instance.model(), file=f)
                ftxt.close()

        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(category_vector, np.array(tr_plot["yield (ton/ha)"]), marker='o', linestyle='-', label='True Yield', color='green')
        plt.plot(category_vector,  np.array(tr_plot["Prediction"]), marker='x', linestyle='--', label='Predicted Yield', color='blue')
        plt.xlabel(self.category)
        plt.ylabel("Groundnut Yield (ton/ha)")
        plt.title("True vs Predicted Yield (Training)")
        plt.xticks(rotation=45)
        plt.grid(False)
        plt.legend()
        plt.tight_layout()
        # plt.show()

        plt.savefig('Train_plot.jpg', format='jpg')
    
        return [sMAPE, nmse]
        

