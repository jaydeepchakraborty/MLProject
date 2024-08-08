import src.data_prep.data_prep as data_prep
import src.data_prep.data_preprocess as data_preprocess

def data_preparation():
    # data_prep.data_prep()
    data_preprocess.data_preprocess()


if __name__ == "__main__":
    # setting os environement
    import os
    os.environ["ENV_NM"] = "dev"
    data_preparation()
