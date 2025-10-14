This project aims at predicting soybean crop yield 2 years ahead of training data. The proposed algorithm utilizes a linear ridge regression model to provide a base prediction of yield. Following the linear model an algorithm named Multi STAge Clustered (M-STAC) Prediction has been proposed that is essentially an ensemble of linear models where a clustering technique enforces explainability in the model. The repetitive error modelling in M-STAC modelling enables a step-wise error reduction method, that compensates for the scarce dataset. The model has been trained on Madhya Pradesh district soybean yield data between the years 2001 to 2017, and validated and tested on 2018 and 2019 respectively.

# Prerequisites
## Libraries used:

Some of the major libraries used are mentioned below:

* python 3.11.3
* numpy==1.24.3
* pandas==1.4.2
* scikit-learn==1.3.0
* joblib==1.4.2

## Installations:

pip install - <requirements.txt> was used for the purpose of installing required packages.

# How to use?
1. Open the params_soybean.yaml file to ensure the correct choices for each input argument.

2. Start execution from main.py. 

How to train the models M-STAC model by updating params_soybean.yaml:


1. Set run_mstac = True to train the MSTAC model. Again ensure the input(train_file_path) and target variables(train_yield_path) required for the training of the MSTAC model is updated with the filenames as per requirement.

How to test the model by updating params_soybean.yaml:

1. Set test= True to test the trained models. Ensure the test input required for the testing  is updated with the filenames as per requirement.


### Multi-STAge Clustered (M-STAC) model training:

1. model() function of class MSTAC is called from the function run_algo() in main.py.

2. The default choice of features are used to select from the exogeneous variables the best choices.

3. The linear ridge regression model is used to predict the initial yield.

4. The error for training set is calculated, which is set as the target variable in all the ridge regression models inside M-STAC iteration.

5. The iterations run for J (set in main.py) times.

7. In each iteration the data is clustered using K-means method based on nitrogen, pH and previous stage crop yield prediction. In the first iteration the number of Clusters= J. The fitted model is saved and also used to predict the clusters of the validation data points.

8. Within the same iteration J number of ridge models are trained, for each cluster. Each of these models are saved and are also used to predict error in previous stage prediction for validation data points.

9. The number of clusters in the next iteration is reduced to J-1. The loop continues J times.

### Model Testing Procedure:

1. model_test() function of class Test is called from function test() in main.py.

2. The features used in training are used.

3. The saved model files both linear model and M-STAC model are used to predict the test yield.

4. The predictions are saved as <user_defined_name>.csv in the mentioned filepath.



Program Flow -->execution starts from main.py file --> method train_neural_network() is executed if train= True is set in params_soybean.yaml--> then method run_algo() of main.py is executed if train_mstac= True is set in params_soybean.yaml--> then model() method of class MSTAC is executed --> if test= True set in params_soybean.yaml then test() method is executed in main.py --> following that model_test() method of Test class is executed.


