# import statements
from util.util_lib import *

class CookingBatchSampler(Sampler):
    
    def __init__(self, data_set, ind, conf, batch_size=8):
        self.data_set = data_set
        self.conf = conf
        
        indices = []
        if ind == "train":
            for _idx, (_text, _lebel) in enumerate(self.data_set):
                indices.append((_idx, len(_text)))
        elif ind == "valid":
            for _idx, (_text, _lebel) in enumerate(self.data_set):
                indices.append((_idx, len(_text)))
        elif ind == "test":
            for _idx, _text in enumerate(self.data_set):
                indices.append((_idx, len(_text)))

        random.shuffle(indices)
        
        self.batch_size = 8
        self.pooled_indices = []
        # create pool of indices with similar lengths 
        for i in range(0, len(indices), self.batch_size * 100):
            self.pooled_indices.extend(sorted(indices[i:i + self.batch_size * 100], 
                                         key=lambda x: x[1]))

        self.pooled_indices = [x[0] for x in self.pooled_indices]
    
    def __iter__(self):
        # yield indices for current batch
        for i in range(0, len(self.pooled_indices), self.batch_size):
            yield self.pooled_indices[i:i + self.batch_size]
            
    def __len__(self):
        return len(self.pooled_indices)