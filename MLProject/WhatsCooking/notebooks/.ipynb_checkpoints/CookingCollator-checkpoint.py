# import statements
from util.util_lib import *

class CookingCollator(object):
    
    def __init__(self, vocab, le, ind, conf):
        
        # indicator ~ train/ valid/ test
        self.ind = ind
        # label encoder
        self.le = le
        # vocabulary
        self.vocab = vocab
        
        # Load tokenizer
        self.tokenizer = get_tokenizer('basic_english')
        # self.text_transform = lambda x: [vocab[token] for token in tokenizer(x)]
        # label_transform = lambda x: 1 if x == 'pos' else 0
    
    def text_transform(self, txt):
        _txt_arr = [self.vocab[token] for token in self.tokenizer(txt)]
        return _txt_arr
    
    def collate_batch_train_valid(self, batch):
        max_rows, max_elm = 0, 0
        for (_text_arr, _label) in batch:
            max_rows = max(max_rows, len(_text_arr))
            for _text in _text_arr:
                processed_text = self.text_transform(_text)
                max_elm = max(max_elm, len(processed_text))

        label_list, label_to_encode_list, text_list, text_to_vocab_list = [], [], [], []
        for (_text_arr, _label) in batch:
            text_to_vocab = []
            for _text in _text_arr:
                processed_text = self.text_transform(_text)
                processed_text.extend([0] * (max_elm - len(processed_text)))
                text_to_vocab.append(processed_text.copy())

            text_to_vocab.extend([[0]*max_elm]*(max_rows - len(_text_arr)))

            # text data
            text_to_vocab_list.append(text_to_vocab.copy())
            text_list.append(_text_arr.copy())

            # label data
            label_list.append(_label)

        label_to_encode_list = self.le.transform(label_list)
        label_to_encode_list_tensor = torch.tensor(label_to_encode_list)

        text_list_tensor = torch.tensor(text_to_vocab_list)

        batch_info = {
            "text_list": text_list,
            "text_to_vocab_list": text_list_tensor, 
            "label_list": label_list, 
            "label_to_encode_list": label_to_encode_list_tensor
        }

        # return torch.tensor(label_list), pad_sequence(text_list, padding_value=3.0)
        return batch_info

    
    def collate_batch_test(self, batch):
    
        max_rows, max_elm = 0, 0
        for _text_arr in batch:
            max_rows = max(max_rows, len(_text_arr))
            for _text in _text_arr:
                processed_text = self.text_transform(_text)
                max_elm = max(max_elm, len(processed_text))

        text_list, text_to_vocab_list = [], []
        for _text_arr in batch:
            text_to_vocab = []
            for _text in _text_arr:
                processed_text = self.text_transform(_text)
                processed_text.extend([0] * (max_elm - len(processed_text)))
                text_to_vocab.append(processed_text.copy())

            text_to_vocab.extend([[0]*max_elm]*(max_rows - len(_text_arr)))

            # text data
            text_to_vocab_list.append(text_to_vocab.copy())
            text_list.append(_text_arr.copy())

        text_list_tensor = torch.tensor(text_to_vocab_list)

        batch_info = {
            "text_list": text_list,
            "text_to_vocab_list": text_list_tensor
        }

        # return torch.tensor(label_list), pad_sequence(text_list, padding_value=3.0)
        return batch_info
    
    
    def __call__(self, batch):
        
        if self.ind == "train":
            return self.collate_batch_train_valid(batch)
        elif self.ind == "valid":
            return self.collate_batch_train_valid(batch)
        elif self.ind == "test":
            return self.collate_batch_test(batch)