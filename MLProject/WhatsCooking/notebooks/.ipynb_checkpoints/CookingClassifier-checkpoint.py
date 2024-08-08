# import statements
from util.util_lib import *

class CookingClassifier(nn.Module):
    
    def __init__(self, conf, **kwargs):
        #Constructor
        super(CookingClassifier, self).__init__(**kwargs)
        
        self.conf = conf
        
        vocab_size = conf['model']['info']['vocab_size']
        self.embedding_dim = 32
        #embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                  embedding_dim=self.embedding_dim, 
                                     padding_idx = 0)
        # kaiming_uniform_, uniform_
        torch.nn.init.uniform_(self.embedding.weight, 0, 1) # make sure embedding weights are b/w 0 and 1
        self.embedding.weight.data[0] = 0 # reset the index of padding_idx to zeros
        
        
        #dense layer / linear layer
        self.output_dim = 20 # number of classes
        self.fc = nn.Linear(self.embedding_dim, self.output_dim)
        
        #activation function
        self.act = nn.Sigmoid()
            
    
    def forward(self, _txt):
        '''
            Step 0: input parameters
            # _txt ~ [batch_size, seq_len, word_len] 
        '''
        
        '''
            Step 1: pass through the embedding layer to convert text into vectors
            # embed_txt ~ [batch_size, seq_len, word_len, embedding_dim] 
        '''
        _embed_txt = self.embedding(_txt)
        
        '''
            Step 2: 
            # _updt_embed_txt_1 ~ [batch_size, seq_len, embedding_dim] 
            # _updt_embed_txt_2 ~ [batch_size, embedding_dim] 
        '''
        _updt_embed_txt_1 = torch.sum(_embed_txt, 2)
        _updt_embed_txt_2 = torch.sum(_updt_embed_txt_1, 1)
        
        '''
            Step 3: feeding the linear output to activation function 
            # fc_out ~ [batch_size, output_dim]
        '''
        fc_out = self.fc(_updt_embed_txt_2)
        
        
        '''
            Step 4: feeding the linear output to activation function 
            # out ~ [batch_size, output_dim]
        '''
        out = self.act(fc_out)
        
        return out