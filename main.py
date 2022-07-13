from transformers import BertConfig
from transformers import BertTokenizer

from torch.utils.data import DataLoader

import dataloader as dl
import trainer as tr

import baseline_dataloader as bdl
import baseline_model as bm
import baseline_trainer as btr

def main(task='atis', device='cpu', use_crf=False, dropout = True, runs=1):       
    
    if task == 'atis':                                                                       
        data_path = "./IntentSlotDatasets/ATIS"
    elif task == 'snips':
        data_path = "./IntentSlotDatasets/SNIPS"
    else:
        print("Invalid task")
        return 1

    PAD_TOKEN = 0 #Specifies a target value that is ignored and does not contribute to the input gradient
    pad_label='PAD'
    unk_label='UNK' 

    max_seq_len = 50 #The maximum total input sequence length after tokenization.
    train_batch_size=32 #Batch size for training
    eval_batch_size=64 #Batch size for evaluation

    epochs=10 #Total number of training epochs to perform                                                                 
    learning_rate=5e-5 #The initial learning rate for Adam
    adam_epsilon=1e-8 #Epsilon for Adam optimizer
    max_grad_norm=1.0 #Max gradient norm                                                                             
    dropout_rate=0.1 #Dropout for fully-connected layers

    #initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    #charge raw data
    train_raw = dl.load_data(data_path + "/train.json")
    test_raw = dl.load_data(data_path + "/test.json")
    print('Train samples len:', len(train_raw))
    print('Test samples len:', len(test_raw))

    #find intent and slot labels
    intent_labels = [unk_label]
    slot_labels = [pad_label, unk_label]
    corpus = train_raw + test_raw                                      
    data_intents = set([line['intent'] for line in corpus])
    data_slots = set(sum([line['slots'].split() for line in corpus],[]))
    intent_labels.extend(data_intents)
    slot_labels.extend(data_slots)
    print("len intent labels: ", len(intent_labels))
    print("len slot labels: ", len(slot_labels))

    #initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    #create dev raw data
    train_raw, dev_raw, test_raw = dl.create_dev_set(train_raw, test_raw)
    
    #create datasets
    train_datas = dl.Processed_data(train_raw, intent_labels, slot_labels, unk_label, pad_label, max_seq_len, tokenizer, PAD_TOKEN)  
    dev_datas = dl.Processed_data(dev_raw, intent_labels, slot_labels, unk_label, pad_label, max_seq_len, tokenizer, PAD_TOKEN)
    test_datas = dl.Processed_data(test_raw, intent_labels, slot_labels, unk_label, pad_label, max_seq_len, tokenizer, PAD_TOKEN)                                                                                       

    train_dataset = dl.My_Dataset(train_datas)
    dev_dataset = dl.My_Dataset(dev_datas)
    test_dataset = dl.My_Dataset(test_datas)

    #create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=eval_batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle = False)

    #charge pretrained Bert model configurations
    config = BertConfig.from_pretrained('bert-base-uncased', finetuning_task=task) 

    #do training and evaluation
    if runs <= 1:                                 
        tr.single_run(epochs, config, train_dataloader, dev_dataloader, test_dataloader, task,
                            dropout_rate, learning_rate, adam_epsilon, max_grad_norm, intent_labels, slot_labels, use_crf, dropout, PAD_TOKEN, tokenizer, device)
    else:
        tr.multiple_runs(runs, epochs, config, train_dataloader, dev_dataloader, test_dataloader, 
                            dropout_rate, learning_rate, adam_epsilon, max_grad_norm, intent_labels, slot_labels, use_crf, dropout, PAD_TOKEN, tokenizer, device)

def main_baseline(task='atis', device='cpu', use_crf=False, dropout=True, runs=1):       
    
    if task == 'atis':                                                                       
        data_path = "./IntentSlotDatasets/ATIS"
    elif task == 'snips':
        data_path = "./IntentSlotDatasets/SNIPS"
    else:
        print("Invalid task")
        return 1

    PAD_TOKEN = 0 #Specifies a target value that is ignored and does not contribute to the input gradient 

    train_batch_size=128 #Batch size for training
    eval_batch_size=64 #Batch size for evaluation

    hid_size=200 #size of the hidden layers
    n_layer=1 #number of hidden layers in the encoder
    emb_size=300 #size of the embedding
    epochs= 200 #Total number of training epochs to perform     
    patience = 3 #parameter for early stopping                                                        
    learning_rate=0.0001 #The initial learning rate for Adam
    max_grad_norm=5 #Max gradient norm 
    dropout_rate=0.1 #Dropout for fully-connected layers                                                                         

    #charge raw data
    train_raw = dl.load_data(data_path + "/train.json")
    test_raw = dl.load_data(data_path + "/test.json")
    print('Train samples len:', len(train_raw))
    print('Test samples len:', len(test_raw))

    #find intent and slot labels
    intent_labels = []
    slot_labels = []
    corpus = train_raw + test_raw                                      
    data_intents = set([line['intent'] for line in corpus])
    data_slots = set(sum([line['slots'].split() for line in corpus],[]))
    intent_labels.extend(data_intents)
    slot_labels.extend(data_slots)
    print("len intent labels: ", len(intent_labels))
    print("len slot labels: ", len(slot_labels))


    #create dev raw data
    train_raw, dev_raw, test_raw = dl.create_dev_set(train_raw, test_raw)

    #create datasets
    words = sum([x['utterance'].split() for x in train_raw], [])

    lang = bdl.Lang(words, intent_labels, slot_labels, PAD_TOKEN, cutoff=0)

    train_dataset = bdl.IntentsAndSlots(train_raw, lang)
    dev_dataset = bdl.IntentsAndSlots(dev_raw, lang)
    test_dataset = bdl.IntentsAndSlots(test_raw, lang)

    #create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=bdl.collate_fn,  shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=eval_batch_size, collate_fn=bdl.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=eval_batch_size, collate_fn=bdl.collate_fn)

    num_slot_labels = len(lang.slot2id)
    num_intent_labels = len(lang.intent2id)
    vocab_len = len(lang.word2id)
    print("vocab len: ", vocab_len)

    #create model
    model = bm.Base_Model(hid_size, num_slot_labels, num_intent_labels, emb_size, vocab_len, use_crf, dropout, dropout_rate, n_layer, pad_token=PAD_TOKEN)

    #do training and evaluation
    if runs <= 1:                                 
        btr.single_run(epochs, model, train_dataloader, dev_dataloader, test_dataloader, lang, task,
                        learning_rate, max_grad_norm, patience, intent_labels, slot_labels, use_crf, dropout, PAD_TOKEN, device)
    else:
        btr.multiple_runs(runs, epochs, model, train_dataloader, dev_dataloader, test_dataloader, lang,
                        learning_rate, max_grad_norm, patience, use_crf, PAD_TOKEN, device)

if __name__ == '__main__':
    #baseline = False #to select baseline model
    baseline = True #to select final model

    #task='atis' #to train and test on ATIS dataset
    task='snips' #to train and test on SNIPS dataset

    device='cpu'

    #use_crf = True #to add CRF layer in the slot classifier
    use_crf = False

    #dropout = True #to use dropout as normalization method
    dropout = False #to use layer normalization

    runs = 1 #for single run execution
    #runs = 3 #for multy run on final model
    #runs = 5 #for multy run on baseline model

    if baseline:
        main_baseline(task, device, use_crf, dropout, runs)
    else:
        main(task, device, use_crf, dropout, runs)
