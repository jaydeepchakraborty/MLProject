# import statements
from util.util_lib import *

class CookingDataset(Dataset):
    def __init__(self, ind, conf):
        if ind == "train":
            self.ind = "train"
            self.data = self.read_data(ind, conf)
            self.data = self.data[['ingredients_processed', 'cuisine']].copy()
        elif ind == "valid":
            self.ind = "valid"
            self.data = self.read_data(ind, conf)
            self.data = self.data[['ingredients_processed', 'cuisine']].copy()
        elif ind == "test":
            self.ind = "test"
            self.data = self.read_data(ind, conf)
            self.data = self.data[['ingredients_processed']].copy()
        else:
            print(f"incorrect indicator:- {ind}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[['ingredients_processed']].iloc[idx]
        item = item['ingredients_processed']
        item_arr = ast.literal_eval(item)
        
        if self.ind == "test":
            return item_arr
        
        label = self.data[['cuisine']].iloc[idx]
        label = label['cuisine']
        
        return item_arr, label
    
    def read_data(self, ind, conf):
        if ind == "train":
            data_fl_path = conf['data']['data_fl_path'] + conf['data']['train_preprocess_fl_nm']
        elif ind == "valid":
            data_fl_path = conf['data']['data_fl_path'] + conf['data']['valid_preprocess_fl_nm']
        elif ind == "test":
            data_fl_path = conf['data']['data_fl_path'] + conf['data']['test_preprocess_fl_nm']


        print(f"data loaded STARTED:- {data_fl_path}")
        data_df = pd.read_csv(data_fl_path)
        print(f"data loaded FINISHED:- {data_fl_path}")
        print(f"data_df shape:- {data_df.shape}")
        return data_df