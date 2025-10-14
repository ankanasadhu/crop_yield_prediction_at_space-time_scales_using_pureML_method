import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping
import tensorflow as tf
# from tensorflow import keras ###Uncomment all the commented code lines to reinitialize the initial weights
# from tensorflow.keras import layers ###Uncomment all the commented code lines to reinitialize the initial weights
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import threading
from keras.callbacks import EarlyStopping
import os


class NeuralNetworkTimeSeries:
    # Constructor initialization 
    def __init__(self, districts,train_val_years, train_start,train_end,val_start,val_end,train_years,num_districts,yield_data,data, data_path,file_path, lstm_model_name):
        self.train_start=train_start ###Starting index of training
        self.train_end=train_end ###Ending index of training
        self.val_start=val_start ###Starting index of validation
        self.val_end=val_end ###Ending index of validation
        self.train_years=train_years ###Total number of years used for training
        self.num_districts=num_districts ###Total number of districts available in dataset
        self.yield_data=yield_data ###Yield data (target value)
        self.data=data ###Previous year's yield of the district under consideration along with 2 neighbouring districts (input data)
        self.filepath=file_path ### Filepath to models
        self.lstm_model_name = lstm_model_name ###The user assigned model name 
        self.districts = districts
        self.data_path = data_path
        self.train_val_years = train_val_years

    
    #Symmetric Mean Absolute Percentage Error calculation function
    def smape(self, y_true, y_pred):
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
        nmse_score = numerator / denominator * 100
        return nmse_score
    
    #Model training 
    def model(self):
        # Create a Tkinter window for the progress bar
        root = tk.Tk()
        root.title("Training Progress")
        root.geometry("400x150")

        model_label = tk.Label(root, text="LSTM training...",font=("bold"))
        model_label.pack(pady=5)

        def close_window():
            # root.quit()
            root.destroy()

        # Create a progress bar widget
        progress = ttk.Progressbar(root, length=300, mode='determinate')
        progress.pack(pady=5)

        # Create a label to show percentage
        progress_label = tk.Label(root, text="0%", font=("Helvetica", 12))
        progress_label.pack(pady=5)

        close_button = tk.Button(root, text="Press to continue", command=close_window, state=tk.DISABLED)
        close_button.pack(pady=5)

        # model_label = tk.Label(root, text="Note : Please Close the Training Progress bar after training to continue with Model Execution",
        #                        wraplength=300,  # Wraps the text at 380px to ensure it fits within the 400px window
        #                        justify='center',  # Justify the text in the center
        #                        font=("Helvetica", 10,"bold"))
        # model_label.pack(pady=5)

        # Function to update progress bar and label
        def update_progress_bar(epoch, logs=None):
            if logs is not None:
                progress_value = (epoch / 1000) * 100  # Update progress (1000 epochs max)
                progress['value'] = progress_value  # Update the progress bar
                progress_label.config(text=f"{progress_value:.1f}%")  # Update the label
                root.update_idletasks()

        #Select seeding value to prevent random initialization
        seed_value = 42
        np.random.seed(seed_value)
        tf.random.set_seed(seed_value)
        # initializer = tf.keras.initializers.GlorotUniform(seed=seed_value)  ###Uncomment all the commented code lines to reinitialize the initial weights
        ###Converting input and output data to tensorflow arrays
        train_data_=tf.convert_to_tensor(self.data.iloc[self.train_start:self.train_end,:].values.reshape(self.train_years*self.num_districts,1,self.data.shape[1]), dtype=tf.float32)
        val_data_=tf.convert_to_tensor(self.data.iloc[self.val_start:self.val_end,:].values.reshape(self.num_districts,1,self.data.shape[1]), dtype=tf.float32)
        y_train_=tf.convert_to_tensor(self.yield_data.iloc[self.train_start:self.train_end].values.reshape(self.train_years*self.num_districts,1), dtype=tf.float32)
        y_val_=tf.convert_to_tensor(self.yield_data.iloc[self.val_start:self.val_end].values.reshape(self.num_districts,1), dtype=tf.float32) 
        
        ###To stop training when validation loss keeps going up even after 60 epochs
        early_stopping = EarlyStopping(monitor='val_loss', mode='min',patience=60, restore_best_weights=True)

        districts = self.districts.iloc[self.val_start:self.val_end].values.reshape(self.num_districts,1)
        districts = np.array(districts)
        districts = [d[0] for d in districts]

        folder = os.path.dirname(self.data_path)
       
        ###Uncomment all the commented code lines to reinitialize the initial weights 

        # model_config = tf.keras.Sequential([
        #         tf.keras.layers.LSTM(16, kernel_initializer=initializer,activation='tanh', input_shape=(1,self.data.shape[1],), return_sequences=True),
        #         tf.keras.layers.LSTM(8, kernel_initializer=initializer,activation='tanh'),
        #         tf.keras.layers.Dense(1, activation='linear')  
        #     ])           
        # optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        # model_config.compile(optimizer=optimizer, loss='mean_squared_error')
        # model_config.save(self.filepath+"initial_wts.h5")
        # model=model_config



        model = tf.keras.models.load_model(self.filepath + "initial_wts.h5") #Loading initial saved weights (to get the same point of convergence)

        # Custom callback for updating the progress bar
        class ProgressBarCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if self.model.stop_training:  # Check if training has stopped (early stopping)
                    # If training stops early, set progress to 100%
                    root.after(0, update_progress_bar, 1000, logs)  # Set progress to 100%
                    root.after(0, lambda: close_button.config(state=tk.NORMAL))
                elif epoch == 999:  # Ensure the progress bar hits 100% on the 1000th epoch
                    root.after(0, update_progress_bar, epoch + 1, logs)
                    root.after(0, lambda: close_button.config(state=tk.NORMAL))  # Enable button
                else:
                    root.after(0, update_progress_bar, epoch, logs)  # Safely update the GUI
    

        def train_model() :
            ## model training
            model.fit(
            train_data_, y_train_, epochs=1000, batch_size=self.num_districts, shuffle=True, validation_data=(val_data_, y_val_),
            callbacks=[early_stopping, ProgressBarCallback()]
            )
            ## validation set prediction
            val_op=model.predict(val_data_)
            ## training set prediction
            tr_op=model.predict(train_data_)
            ## calculation of training and validation sMAPE
            tr_smape=self.smape(np.array(y_train_).reshape(self.train_end,1),tr_op.reshape(self.train_end,1))
            val_smape=self.smape(y_val_,val_op.reshape(self.num_districts,1))

            tr_nmse = self.nmse(np.array(y_train_).reshape(self.train_end,1),tr_op.reshape(self.train_end,1))
            val_nmse = self.nmse(y_val_,val_op.reshape(self.num_districts,1))

            print([tr_smape, val_smape])

            with open("lstm_output.txt", "w") as f:
                print(f"*LSTM*\nTraining sMAPE: {np.round(tr_smape,2)}% NMSE: {np.round(tr_nmse,2)}% \nValidation sMAPE: {np.round(val_smape,2)}% NMSE: {np.round(val_nmse,2)}%", file=f)
                # print(my_instance.model(), file=f)
                f.close()

            # Create the district list for training and validation
            districts_train = districts * self.train_years  # repeats the list for training years
            districts_val = districts                       # one repetition for validation
            districts_full = districts_train + districts_val  # total list

            # Sanity check: match length
            assert len(districts_full) == len(y_train_) + len(y_val_), "District length mismatch"

            # Reshape the arrays
            y_true = np.concatenate([np.array(y_train_).reshape(-1, 1), np.array(y_val_).reshape(-1, 1)], axis=0)
            y_pred = np.concatenate([tr_op.reshape(-1, 1), val_op.reshape(-1, 1)], axis=0)

            # Combine all data
            df = pd.DataFrame({
                "District": districts_full,
                "Years": self.train_val_years,
                "Actual": y_true.flatten(),
                "Predicted": y_pred.flatten()
            })

            # Save to CSV
            df.to_csv("train_valid_prediction.csv", index=False)
            
            # Plotting
            plt.figure(figsize=(10, 8))
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.3, top=0.9, left=0.1, right=0.9)

            # Scatter plots with labels for legend
            plt.scatter(districts, val_op, color='r', linewidth=2, label='Predicted Yield')
            plt.scatter(districts, y_val_, color='b', linewidth=2, label='Actual Yield')

            # Line plots without labels (not added to legend)
            plt.plot(districts, val_op, linestyle='--', color='r', linewidth=2)
            plt.plot(districts, y_val_, linestyle='--', color='b', linewidth=2)

           

            # Axis formatting
            plt.xticks(districts, districts, rotation=90, fontsize=8)
            plt.ylabel('Groundnut Yield (ton/ha)')
            plt.xlabel('Districts')
            plt.legend()
            # plt.show()

             # Set title with SMAPE and NMSE values
            plt.title(f'LSTM Validation sMAPE: {val_smape:.2f}%  |  NMSE: {val_nmse:.2f}%', fontsize=12)

            plt.savefig(folder + '/' + 'Train_plot.jpg', format='jpg')
            plt.close()
  
            model.save(self.filepath+self.lstm_model_name) ###Saving the model with the user defined filepath and filename
            return model
    
        # Start training in a separate thread to avoid blocking the GUI
        training_thread = threading.Thread(target=train_model, daemon=True)
        training_thread.start()

        # Start Tkinter main loop in the main thread
        root.mainloop()

