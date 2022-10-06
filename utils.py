import os
import pandas as pd
import json

def save_instance_data(car_info, instance_info, instance, path):
    os.makedirs(path, exist_ok=True)
    car_info.to_csv(path + '/car_info.csv')
    instance.to_csv(path + '/instance.csv')
    with open(path + '/instance_info.csv', 'w') as f:
        f.write(json.dumps(instance_info))

def load_instance_data(path):
    car_info = pd.read_csv(path + '/car_info.csv', index_col=0)
    instance = pd.read_csv(path + '/instance.csv', index_col=0)

    with open(path + '/instance_info.csv', 'r') as f:
        instance_info = json.loads(f.read())
    car_info.rename(columns={str(x): int(x) for x in instance_info['available_car_speeds']}, inplace=True)
    return car_info, instance_info, instance

def save_true_quantum_result(res_df, logdir, instance_name, p, energy):
    path = logdir + '/' + instance_name
    os.makedirs(path, exist_ok=True)
    res_df.to_csv(path + '/p_{}_e_{}.csv'.format(p, energy), index=False)