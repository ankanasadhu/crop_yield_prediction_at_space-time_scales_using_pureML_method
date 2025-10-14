import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping
import tensorflow as tf
# from tensorflow import keras###Uncomment all the commented code lines to reinitialize the initial weights
# from tensorflow.keras import layers###Uncomment all the commented code lines to reinitialize the initial weights
import matplotlib.pyplot as plt

class NeuralNetworkTimeSeries:
    # Constructor initialization 
    def __init__(self,train_start,train_end,val_start,val_end,train_years,num_districts,yield_data,data,file_path, lstm_model_name):
        self.train_start=train_start###Starting index of training
        self.train_end=train_end###Ending index of training
        self.val_start=val_start###Starting index of validation
        self.val_end=val_end###Ending index of validation
        self.train_years=train_years###Total number of years used for training
        self.num_districts=num_districts###Total number of districts available in dataset
        self.yield_data=yield_data###Yield data (target value)
        self.data=data###Previous year's yield of the district under consideration along with 2 neighbouring districts (input data)
        self.filepath=file_path### Filepath to models
        self.lstm_model_name = lstm_model_name###The user assigned model name 
    
    #Symmetric Mean Absolute Percentage Error calculation function
    def smape(self,y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        numerator = np.abs(y_true - y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred))
        smape_score = np.mean(numerator / denominator) * 100
        return smape_score
    
    #Model training 
    def model(self):
        #Select seeding value to prevent random initialization
        seed_value = 42
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
        initializer = tf.keras.initializers.GlorotUniform(seed=seed_value)
        ###Converting input and output data to tensorflow arrays
        train_data_=tf.convert_to_tensor(self.data.iloc[self.train_start:self.train_end,:].values.reshape(self.train_years*self.num_districts,1,self.data.shape[1]), dtype=tf.float32)
        val_data_=tf.convert_to_tensor(self.data.iloc[self.val_start:self.val_end,:].values.reshape(self.num_districts,1,self.data.shape[1]), dtype=tf.float32)
        y_train_=tf.convert_to_tensor(self.yield_data.iloc[self.train_start:self.train_end].values.reshape(self.train_years*self.num_districts,1), dtype=tf.float32)
        y_val_=tf.convert_to_tensor(self.yield_data.iloc[self.val_start:self.val_end].values.reshape(self.num_districts,1), dtype=tf.float32)
        
         ###To stop training when validation loss keeps going up even after 60 epochs
        early_stopping = EarlyStopping(monitor='val_loss', mode='min',patience=60, restore_best_weights=True)
        
        ###Uncomment all the commented code lines to reinitialize the initial weights 

        # model_config = tf.keras.Sequential([
        #         tf.keras.layers.LSTM(16, kernel_initializer=initializer,activation='tanh', input_shape=(1,self.data.shape[1],), return_sequences=True),
        #         tf.keras.layers.LSTM(8, kernel_initializer=initializer,activation='tanh'),
        #         tf.keras.layers.Dense(1, activation='linear')  
        #     ])
        # optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        # model_config.compile(optimizer=optimizer, loss='mean_squared_error')
        # model_config.save(self.filepath+"initial_wts.h5")
        # model=model_config
        model = tf.keras.models.load_model(self.filepath + "initial_wts.h5")#Loading initial saved weights (to get the same point of convergence).

       ## model training
        model.fit(train_data_, y_train_, epochs=1000, batch_size=self.num_districts, shuffle=True,validation_data=(val_data_,y_val_), callbacks=[early_stopping])
         ## validation set prediction
        val_op=model.predict(val_data_)
        ## training set prediction
        tr_op=model.predict(train_data_)
        ## calculation of training and validation sMAPE
        tr_smape=self.smape(np.array(y_train_).reshape(self.train_end,1),tr_op.reshape(self.train_end,1))
        val_smape=self.smape(y_val_,val_op.reshape(self.num_districts,1))
        print([tr_smape, val_smape])
      
        model.save(self.filepath+self.lstm_model_name)
        return model

