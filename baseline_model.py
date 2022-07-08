import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF

'''
Base_Model bidirectional GRU as encoder shared with both intent and slot classifier
If use_crf=True we add a CRF for the slot filling task
'''
class Base_Model(nn.Module):

    def __init__(self, hid_size, num_slot_labels, num_intent_labels, emb_size, vocab_len, use_crf, dropout, dropout_rate, n_layer=1, pad_token=0):
        super(Base_Model, self).__init__()
        
        self.hid_size = hid_size
        self.num_intent_labels = num_intent_labels
        self.num_slot_labels = num_slot_labels
        self.emb_size = emb_size
        self.vocab_len = vocab_len
        self.use_crf = use_crf
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.n_layer = n_layer
        self.pad_token = pad_token
        
        self.embedding = nn.Embedding(self.vocab_len, self.emb_size, padding_idx=self.pad_token)
          
        self.utt_encoder = nn.GRU(emb_size, hid_size // 2, n_layer, bidirectional=True) 
        if self.dropout:
            self.intent_classifier = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hid_size // 2, self.num_intent_labels)
            )
            self.slot_classifier = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hid_size, self.num_slot_labels)
            )
        else:          
            self.intent_classifier = nn.Sequential(
                nn.LayerNorm(self.hid_size // 2),
                nn.Linear(self.hid_size // 2, self.num_intent_labels)
            )
            self.slot_classifier = nn.Sequential(
                nn.LayerNorm(self.hid_size),
                nn.Linear(self.hid_size, self.num_slot_labels)
            )
        if self.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=False)
        
    def forward(self, utterance, seq_lengths):
        utt_emb = self.embedding(utterance) # utt_emb.size() = batch_size X seq_len X emb_size
        utt_emb = utt_emb.permute(1,0,2) # we need seq len first -> seq_len X batch_size X emb_size
        
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy())
        packed_output, last_hidden = self.utt_encoder(packed_input) 
        # Unpack the sequence
        utt_encoded, input_sizes = pad_packed_sequence(packed_output)
        # Get the last hidden state
        last_hidden = last_hidden[-1,:,:]
        slots = self.slot_classifier(utt_encoded)
        intent = self.intent_classifier(last_hidden)
        
        # Slot size: seq_len, batch size, calsses 
        slots = slots.permute(1,2,0)
        # Slot size: batch_size, classes, seq_len
        return intent, slots
        
'''
Since our base model is not pretrained as the Bert model it's important to initialize the weights to make the training easier,
a good starting point it's better than a random one.
'''
def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)