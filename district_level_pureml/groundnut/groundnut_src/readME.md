# Prerequisites
## Libraries used:

Some of the major libraries used are mentioned below:

* numpy==1.24.3
* pandas==1.4.2
* tensorflow==2.13.0
* scikit-learn==1.3.0
* joblib==1.4.2

## Installations:

pip installment - <requirements.txt> was used for the purpose of installments.

# How to use?
1. Open the params_groundnut.yaml file to ensure the correct choices for each input argument.

2. Start execution from main.py. 

How to train the models (NeuralNet_Model and MSTAC) by updating params_groundnut.yaml:

1. Set train= True to train the neural network (LSTM). Ensure the input (train_file_path) and target variables (train_yield_path) required for the training of the neural network model is updated with the filenames as per requirement.

2. Set run_mstac= True to train the MSTAC model. Again ensure the input(district_data_path) and target variables(train_yield_path) required for the training of the MSTAC model is updated with the filenames as per requirement.

How to test the model by updating params_groundnut.yaml:

1. Set test= True to test the trained models. Ensure the test input required for the testing  is updated with the filenames as per requirement.

### Neural Network Model training procedure:

1. model() function of class NeuralNetworkTimeSeries is called from the function train_neural_net() in main.py.

2. In model() function initial model weights are saved for the LSTM network to reach to the same stable condition for repeated training of the model. The commented code lines can be uncommented to change the initial model weights.

3. The input data is split into training years and one validation year. The model is then fitted with the previous years' yield of the corresponding district and two neighbouring districts. Thd model is saved with the filename mentioned in params_groundnut.yaml.

4. The validation and training sMAPE error is displayed at the end.

### Multi-STAge Clustered (M-STAC) model training:

1. model() function of class MSTAC is called from the function run_algo() in main.py.

2. The default choice of features are used to select from the exogeneous variables the best choices.

3. The error for training set is calculated, which is set as the target variable in all the ridge regression models inside M-STAC iteration.

4. The iterations run for J (set in main.py) times.

5. In each iteration the data is clustered using K-means method based on nitrogen, pH and previous stage crop yield prediction. In the first iteration the number of Clusters= J. The fitted model is saved and also used to predict the clusters of the validation data points.

6. Within the same iteration J number of ridge models are trained, for each cluster. Each of these models are saved and are also used to predict error in previous stage prediction for validation data points.

7. The number of clusters in the next iteration is reduced to J-1. The loop continues J times.

### Model Testing Procedure:

1. model_test() function of class Test is called from function test() in main.py.

2. The features used in training are used.

3. The saved model files both LSTM model and M-STAC model are used to predict the test yield.

4. The predictions are saved as <user_defined_name>.csv in the mentioned filepath.



Program Flow -->execution starts from main.py file --> method train_neural_network() is executed if train= True is set in params_groundnut.yaml--> model() method of class NeuralNetworkTimeSeries which trains the LSTM network--> then method run_algo() of main.py is executed if train_mstac= True is set in params_groundnut.yaml--> then model() method of class MSTAC is executed --> if test= True set in params_groundnut.yaml then test() method is executed in main.py --> following that model_test() method of Test class is executed.


