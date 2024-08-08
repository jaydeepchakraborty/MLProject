from util.util_lib import *
import util.util_cnst as cnst


def get_conf():
    conf = EnvYAML(cnst.CONFIG_FL)
    conf = conf[conf['env_nm']]
    return conf

def read_data(data_fl_path, conf):
    print(f"data loaded STARTED:- {data_fl_path}")
    data_df = pd.read_csv(data_fl_path)
    print(f"data loaded FINISHED:- {data_fl_path}")
    print(f"data_df shape:- {data_df.shape}")
    return data_df

def data_preprocess_util(row):
    STOPWORDS = set(stopwords.words('english'))
    PUNCT_TO_REMOVE = string.punctuation
    if pd.notnull(row):
        # change to lower case
        row_list = ast.literal_eval(row)

        row_list = [word.lower() for word in row_list]
        row_list = [word.translate(str.maketrans('', '', PUNCT_TO_REMOVE)) for word in row_list]
        row_list = [word for word in row_list if word not in STOPWORDS]

        return row_list

    return None

def save_file(data_df, fl_path_nm, conf):
    print(f"data save STARTED:- {fl_path_nm}")
    data_df.to_csv(fl_path_nm, index=False)
    print(f"data save FINISHED:- {fl_path_nm}")
    return

def data_preprocess():
    # reading configuration
    conf = get_conf()

    train_fl_path = conf['data']['data_fl_path'] + conf['data']['train_fl_nm']
    train_df = read_data(train_fl_path, conf)
    train_df['ingredients_processed'] =  train_df['ingredients'].apply(lambda x: data_preprocess_util(x))
    train_fl_path = conf['data']['data_fl_path'] + conf['data']['train_preprocess_fl_nm']
    save_file(train_df, train_fl_path, conf)

    valid_fl_path = conf['data']['data_fl_path'] + conf['data']['valid_fl_nm']
    valid_df = read_data(valid_fl_path, conf)
    valid_df['ingredients_processed'] = valid_df['ingredients'].apply(lambda x: data_preprocess_util(x))
    valid_fl_path = conf['data']['data_fl_path'] + conf['data']['valid_preprocess_fl_nm']
    save_file(valid_df, valid_fl_path, conf)

    return

