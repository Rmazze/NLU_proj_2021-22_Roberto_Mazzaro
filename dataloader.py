import json

import torch
from torch.utils.data import Dataset

from collections import Counter

from sklearn.model_selection import train_test_split

'''
Load the raw data from the given dataset
'''
def load_data(path):
    '''
        input: path/to/data
        output: json 
    '''
    raw_data = []
    with open(path) as f:
        raw_data = json.loads(f.read())
    return raw_data

'''
Create a development dataset since we have just a train and a test dataset 
'''
def create_dev_set(train_raw, test_raw):
    print('Initial TRAIN size:', len(train_raw))
    print('Initial TEST size:', len(test_raw))
    portion = round(((len(train_raw) + len(test_raw)) * 0.10)/(len(train_raw)),2)

    intents = [x['intent'] for x in train_raw] # We stratify on intents
    count_y = Counter(intents)

    Y = []
    X = []
    mini_Train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1: # Some intents have only one instance, we put them in training
            X.append(train_raw[id_y])
            Y.append(y)
        else:
            mini_Train.append(train_raw[id_y])
    # Random Stratify
    X_train, X_dev, y_train, y_dev = train_test_split(X, Y, test_size=portion, random_state=42, shuffle=True, stratify=Y)

    X_train.extend(mini_Train)
    train_raw = X_train
    dev_raw = X_dev

    y_test = [x['intent'] for x in test_raw]

    # Dataset size
    print('TRAIN size:', len(train_raw))
    print('DEV size:', len(dev_raw))
    print('TEST size:', len(test_raw))

    return train_raw, dev_raw, test_raw

'''
Main inputs of this class are raw data of the given dataset and the tokenizer that is a BertTokenizer that perform word piece tokenization,
that is a lossy tokenization that split rare words into meaningful subwords, in that way we obtain a vocabulary of resonable size but still
able to learn meaningful context-independent representations. 
In addition, subword tokenization enables the model to process words it has never seen before, by decomposing them into known subwords.

With this type of tokenization number of tokens is no more alligned with number of slots, so we have to add PAD_TOKEN in the case we 
split a word in subwords.
Moreover after each sequence of tokens we have to add a SEP-aretion tokens and before a CLS token

This class also convert tokens slots and intents into ids.
To convert slots and intents we just use the index of the corresponding label in the intent or slot labels lists,
while to convert tokens into ids we use the convert_tokens_to_ids function of the tokenizer

attention_mask and token_type_ids are also computed
'''
class Processed_data(object):                                                                                          

    def __init__(self, raw_data, intent_labels, slot_labels, unk_label, pad_label,
        max_seq_len, tokenizer, pad_token_label_id=0, cls_token_segment_id=0, pad_token_segment_id=0, sequence_a_segment_id=0):
        
        self.raw_data = raw_data

        self.intent_labels = intent_labels #add unk
        self.slot_labels = slot_labels #add pad and unk

        self.unk_label = unk_label
        self.pad_label = pad_label

        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.pad_token_label_id = pad_token_label_id
        self.cls_token_segment_id = cls_token_segment_id
        self.pad_token_segment_id = pad_token_segment_id
        self.sequence_a_segment_id = sequence_a_segment_id

        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.unk_token = self.tokenizer.unk_token
        self.pad_token_id = self.tokenizer.pad_token_id
        
        self.utterances = []
        self.intents = []
        self.slots = []
        
        self.features = []


        for x in self.raw_data:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        for utt, intent, slot in zip(self.utterances, self.intents, self.slots):
            # get words
            words = utt.split()
            # get intent ids
            intent_label = self.intent_labels.index(intent) if intent in self.intent_labels else self.intent_labels.index(self.unk)
            # 3. slot
            slot_labels = []
            for s in slot.split():
                slot_labels.append(self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index(self.unk))

            assert len(words) == len(slot_labels)
            
            # Tokenize word by word
            tokens = []
            slot_labels_ids = []
            for word, slot_label in zip(words, slot_labels):
                word_tokens = self.tokenizer.tokenize(word)
                if not word_tokens:
                    word_tokens = [self.unk_token]  # For handling the bad-encoded word
                tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                slot_labels_ids.extend([int(slot_label)] + [self.pad_token_label_id] * (len(word_tokens) - 1))
            
            # Account for [CLS] and [SEP]
            special_tokens_count = 2
            if len(tokens) > self.max_seq_len - special_tokens_count:
                tokens = tokens[:(self.max_seq_len - special_tokens_count)]
                slot_labels_ids = slot_labels_ids[:(self.max_seq_len - special_tokens_count)]

            # Add [SEP] token
            tokens += [self.sep_token]
            slot_labels_ids += [self.pad_token_label_id]
            token_type_ids = [self.sequence_a_segment_id] * len(tokens)

            # Add [CLS] token
            tokens = [self.cls_token] + tokens
            slot_labels_ids = [self.pad_token_label_id] + slot_labels_ids
            token_type_ids = [self.cls_token_segment_id] + token_type_ids

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            attention_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = self.max_seq_len - len(input_ids)
            input_ids = input_ids + ([self.pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([self.pad_token_segment_id] * padding_length)
            slot_labels_ids = slot_labels_ids + ([self.pad_token_label_id] * padding_length)

            assert len(input_ids) == self.max_seq_len, "Error with input length {} vs {}".format(len(input_ids), self.max_seq_len)
            assert len(attention_mask) == self.max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), self.max_seq_len)
            assert len(token_type_ids) == self.max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), self.max_seq_len)
            assert len(slot_labels_ids) == self.max_seq_len, "Error with slot labels length {} vs {}".format(len(slot_labels_ids), self.max_seq_len)

            intent_label_id = int(intent_label)
            self.features.append({'input_ids':input_ids,
                          'attention_mask':attention_mask,
                          'token_type_ids':token_type_ids,
                          'intent_label_id':intent_label_id,
                          'slot_labels_ids':slot_labels_ids,
                          })

'''
We define a dataset class that helps to handle the dataset and can be passed to a dataloader.
- input_ids: are the tokens converted in ids
- attention_mask: it's a mask to know how wich tokens to take into account in a sequence and which are just padding
- token_type_ids: it's a mask that indicates the first and second portions of the inputs
- intent_label_id: label of the reference intent
- slot_labels_ids: labels of the reference slots padded to the max sequence length 
''' 
class My_Dataset(Dataset):
    def __init__(self, datas):
        self.all_input_ids = torch.tensor([f['input_ids'] for f in datas.features], dtype=torch.long) 
        self.all_attention_mask = torch.tensor([f['attention_mask'] for f in datas.features], dtype=torch.bool)
        self.all_token_type_ids = torch.tensor([f['token_type_ids'] for f in datas.features], dtype=torch.long)
        self.all_intent_label_ids = torch.tensor([f['intent_label_id'] for f in datas.features], dtype=torch.long)
        self.all_slot_labels_ids = torch.tensor([f['slot_labels_ids'] for f in datas.features], dtype=torch.long)

    def __len__(self):
        return len(self.all_intent_label_ids)

    def __getitem__(self, idx):
        input_ids = self.all_input_ids[idx]
        attention_mask = self.all_attention_mask[idx]
        token_type_ids = self.all_token_type_ids[idx]
        intent_label_ids = self.all_intent_label_ids[idx]
        slot_labels_ids = self.all_slot_labels_ids[idx]
        sample = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
            'intent_label_id': intent_label_ids, 'slot_labels_ids': slot_labels_ids}
        return sample
