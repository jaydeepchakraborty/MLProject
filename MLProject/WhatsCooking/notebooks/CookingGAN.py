# import statements
from util.util_lib import *
from CookingClassifier import CookingClassifier

class CookingGAN:
    def __init__(self, conf):
        
        
        self.conf = conf
        
        self.N_EPOCHS = conf['model']['info']['train_epoch']
        self.VALIDATION_EPOCH = conf['model']['info']['valid_epoch']
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        
        #instantiate the model
        self.model = CookingClassifier(conf)
        self.model = self.model.to(device)
        
        #define the optimizer
        self.optimizer = optim.Adam(self.model.parameters())
        
        #define the loss
        self.criterion = MulticlassHingeLoss(num_classes=20) # Multiclass Hinge loss, Entropy loss
        self.criterion = self.criterion.to(device)
    
    #define metric
    def multicls_accuracy(self, preds, y):
        acc = multiclass_accuracy(preds, y)
        return acc

    def valid_model(self, valid_iterator):
        epoch_loss = 0
        epoch_acc = 0

        with torch.no_grad():
            for valid_batch in valid_iterator:

                #retrieve text
                _text, _lbl = valid_batch['text_to_vocab_list'], valid_batch['label_to_encode_list']

                #get prediction
                predictions = self.model(_text)
                preds = predictions.squeeze(-1) #convert to 1D tensor

                #compute the loss
                loss = self.criterion(preds, _lbl)

                #compute the binary accuracy
                acc = self.multicls_accuracy(preds, _lbl)

                # compute loss and accuracy
                epoch_loss += loss.item()
                epoch_acc += acc.item()

        valid_epoc_loss = epoch_loss / len(valid_iterator)
        valid_epoch_acc = epoch_acc / len(valid_iterator)

        return valid_epoc_loss, valid_epoch_acc
    
    def train(self, train_dl, valid_dl):
        
        #set the model in training phase
        self.model.train()
        
        for epoch in range(self.N_EPOCHS+1):
            
            #initialize every epoch 
            epoch_loss = 0
            epoch_acc = 0
            
            for train_batch in train_dl:
                #resets the gradients after every batch
                self.optimizer.zero_grad()
                
                _text, _lbl = train_batch['text_to_vocab_list'], train_batch['label_to_encode_list']
                
                _predictions = self.model(_text)
                _preds = _predictions.squeeze(-1) #convert to 1D tensor
                
                #compute the loss
                loss = self.criterion(_preds, _lbl)
                
                #compute the binary accuracy
                acc = self.multicls_accuracy(_preds, _lbl)
                
                #backpropage the loss and compute the gradients
                loss.backward()
                
                #update the weights
                self.optimizer.step()
                
                # compute loss and accuracy
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                
            if epoch%self.VALIDATION_EPOCH == 0:
                self.model.eval() # set the model in eval phase
                valid_epoc_loss, valid_epoch_acc = self.valid_model(valid_dl)
                self.model.train() # return back to training phase

                print(f"epoch:- ",epoch)
                print(f"training===> ","loss:- ", epoch_loss / len(train_dl), "  accuracy:- ", epoch_acc / len(train_dl))
                print(f"validation===> ","loss:- ", valid_epoc_loss, "  accuracy:- ", valid_epoch_acc)

            if epoch == self.N_EPOCHS:
                g_path = self.conf['data']['data_fl_path'] + self.conf['model']['model_path']
                torch.save(self.model.state_dict(), g_path)
                print(f"model saved:- {g_path}")
                
                
    def test(self, test_iterator):

        with torch.no_grad():
            for test_batch in test_iterator:

                #retrieve text
                _text = test_batch['text_to_vocab_list']

                #get prediction
                predictions = self.model(_text)
                preds = predictions.squeeze(-1) #convert to 1D tensor
                
                # receive output logits
                _, preds = torch.max(preds, 1)

        return preds
            