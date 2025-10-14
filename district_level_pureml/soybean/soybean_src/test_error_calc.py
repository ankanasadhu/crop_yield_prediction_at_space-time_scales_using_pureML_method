
import pandas as pd
import numpy as np
import argparse
#Symmetric Mean Absolute Percentage Error calculation function
def test(args):
    y_test=pd.read_csv(args.test_yield_data_path).iloc[:,2]
    test_pred_2=pd.read_csv(args.test_pred_path).iloc[:,1]

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

    print("sMAPE :", smape(y_test.values.reshape(args.num_districts,1),test_pred_2.values.reshape(args.num_districts,1)))
    print("NMSE :", nmse(y_test.values.reshape(args.num_districts,1),test_pred_2.values.reshape(args.num_districts,1)))
#Calculate sMAPE of test data
if __name__ == '__main__':
    data_path = '../soybean_data/'
    models_path = '../soybean_models/'
    mstac_models_path = models_path + 'Ridge_model_pureML2'
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_pred_path', type=str, default=data_path + 'Test_prediction.csv')##Change filename as per user need
    parser.add_argument('--test_yield_data_path', type=str, default=data_path + 'Test_yield_MP.csv')
    parser.add_argument('--num_districts', type=int, default=40)
    
    args = parser.parse_args()
    for arg, val in args._get_kwargs():
        print(f"{arg} = {val}")
    print(" ")
    test(args)