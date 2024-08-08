1) DataPrep
--> read raw data and split it to train and validation ( raw_data/train.json )
--> save the train data ( train/train.csv )
--> save the validation data ( valid/valid.csv )
--> raed the test data ( raw_data/test.json )
--> save the test data ( test/test.csv )

2) DataPreprocess

3) DataEncode
--> encoding the TARGET ( label encoding )

4) Vocab
--> generate vocab object 

5) DataLoader
--> read the train data ( train/train.csv )
--> convert to CookingDataset ( pytorch dataset )
--> convert to torch dataloader ( train, valid, test)

6) ModelTrain
--> training model with train, valid dataloader

7) ModelTest
--> 