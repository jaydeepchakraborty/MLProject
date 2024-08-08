from util.util_lib import *
import util.util_cnst as cnst

def get_conf():
    conf = EnvYAML(cnst.CONFIG_FL)
    conf = conf[conf['env_nm']]
    return conf

def read_data(conf):
    data_fl_path = conf['data']['data_fl_path'] + conf['data']['data_fl_nm'][0]
    print(f"data loaded STARTED:- {data_fl_path}")
    data_df = pd.read_json(data_fl_path)
    print(f"data loaded FINISHED:- {data_fl_path}")
    print(f"data_df shape:- {data_df.shape}")
    return data_df

def read_test_data(conf):
    data_fl_path = conf['data']['data_fl_path'] + conf['data']['data_fl_nm'][1]
    print(f"data loaded STARTED:- {data_fl_path}")
    data_df = pd.read_json(data_fl_path)
    print(f"data loaded FINISHED:- {data_fl_path}")
    print(f"data_df shape:- {data_df.shape}")
    return data_df

def split_data(data_df, conf):
    data_df_train, data_df_valid = train_test_split(data_df, test_size=0.2)
    print(f"data_df_train shape:- {data_df_train.shape}")
    print(f"data_df_valid shape:- {data_df_valid.shape}")
    return data_df_train, data_df_valid

def save_file(data_df, fl_path_nm, conf):
    print(f"data save STARTED:- {fl_path_nm}")
    data_df.to_csv(fl_path_nm, index=False)
    print(f"data save FINISHED:- {fl_path_nm}")
    return

def data_prep():
    # reading configuration
    conf = get_conf()

    # reading raw data
    data_df = read_data(conf)

    # spliting the data into train and validation data
    data_df_train, data_df_valid = split_data(data_df, conf)

    # saving train data
    train_fl_path = conf['data']['data_fl_path'] + conf['data']['train_fl_nm']
    save_file(data_df_train, train_fl_path, conf)

    # saving validation data
    valid_fl_path = conf['data']['data_fl_path'] + conf['data']['valid_fl_nm']
    save_file(data_df_valid, valid_fl_path, conf)

    # reading raw test data
    data_df_test = read_test_data(conf)

    # saving test data
    test_fl_path = conf['data']['data_fl_path'] + conf['data']['test_fl_nm']
    save_file(data_df_test, test_fl_path, conf)
    
    return

