
import pandas as pd
import numpy as np
import argparse
#Symmetric Mean Absolute Percentage Error calculation function
def test(args,category,data_path):
    
    y_test=pd.read_csv(args.test_yield_data_path)
    y_test=y_test[[category,'yield (ton/ha)']]
    y_test=y_test.groupby(category).mean()
   
    y_test=y_test.reset_index()
    test_pred_2=pd.read_csv(args.test_pred_path)
    arr=pd.merge(y_test, test_pred_2, how='inner', on=category)

    arr.to_csv(data_path+"true_pred_test"+category+".csv",index=False)
    

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

    print("Smape :", smape(arr["yield (ton/ha)"],arr["Prediction"]))
    print("NMSE :", nmse(arr["yield (ton/ha)"],arr["Prediction"]))
#Calculate sMAPE of test data
if __name__ == '__main__':
    data_path = '../data/1m/'
    category='Tehsil'
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_pred_path', type=str, default=data_path + 'test_1m_soybean_pred_'+category+'.csv')##Change filename as per user need
    parser.add_argument('--test_yield_data_path', type=str, default=data_path + 'test_yield_soybean.csv')
    parser.add_argument('--num_districts', type=int, default=40)
    
    args = parser.parse_args()
    for arg, val in args._get_kwargs():
        print(f"{arg} = {val}")
    print("\n")
    test(args,category,data_path)