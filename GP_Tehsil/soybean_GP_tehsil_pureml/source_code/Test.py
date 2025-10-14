import numpy as np
import joblib
import pandas as pd
import ast
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures



class Test:
    def __init__(self,test,J,ftr_name,test_path,model_path,category):
        self.test=test
        self.J=J
        self.ftr_name=ftr_name
        self.test_path=test_path
        self.filepath=model_path
        self.category=category
        
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
    
    def model_test(self):
        modellr = joblib.load(self.filepath+'/'+'Base linear model')

        test_pred=modellr.predict(self.test.iloc[:,4:])

        pred=pd.DataFrame(test_pred).reset_index(drop=True)
        pred.columns = ["Prediction"]
        data_test=pd.concat([self.test.reset_index(drop=True),pred],axis=1)

        cluster_test_input=data_test[['nitrogen', 'phh2o',"Prediction"]]

        J=self.J
        indx=ast.literal_eval(open(self.ftr_name, "r", encoding="utf-8").read())
        test_data=data_test[indx]


        for j in range (J):
            
            model_kmeans=joblib.load( self.filepath+'/'+str(j+1)+'_iteration_kmeans_model.pkl')
            cluster_test=model_kmeans.predict(cluster_test_input)
            
            
            polys=[]
            models_=[]
            for d in range (J-j):
                polys.append(joblib.load(self.filepath+'/'+str(j+1)+'_iteration_'+str(d+1)+'poly_transformer.pkl'))
                models_.append(joblib.load(self.filepath+'/'+str(j+1)+'_iteration_'+str(d+1)+'cluster_ridge_model.pkl'))
            for n in range(cluster_test.shape[0]):
                x_poly=polys[cluster_test[n]].transform(test_data.iloc[n:n+1,:])
                temp=models_[cluster_test[n]].predict(x_poly)
            
                test_pred[n]=float(temp[0])+test_pred[n]
                
            cluster_test_input["Prediction"]=np.abs(test_pred)
        final_test=pd.concat([self.test.reset_index(drop=True),cluster_test_input["Prediction"]],axis=1)
        final_test=final_test[[self.category,'Prediction']]
        tr=final_test.groupby(self.category).mean().reset_index()

        tr = tr[tr[self.category] != "Sanwer_noname"]

        tr.to_csv(self.test_path, index=True)
