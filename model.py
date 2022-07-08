import torch.nn as nn
from torchcrf import CRF
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel

'''
JointBERT model use Bert pretrained model as encoder shared with both intent and slot classifier
If use_crf=True we add a CRF for the slot filling task 
If dropout=True use dropout layer before linear layers otherwise Linear Normalization
'''
class JointBERT(BertPreTrainedModel):
    def __init__(self, config, dropout_rate, use_crf, dropout, intent_labels, slot_labels):
        super(JointBERT, self).__init__(config)

        self.num_intent_labels = len(intent_labels)
        self.num_slot_labels = len(slot_labels)
        
        self.dropout_rate = dropout_rate
        self.use_crf = use_crf
        self.dropout = dropout

        self.bert = BertModel(config=config)  # Load pretrained bert

        if self.dropout:
            self.intent_classifier = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(config.hidden_size, self.num_intent_labels)
            )
            self.slot_classifier = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(config.hidden_size, self.num_slot_labels)
            )
        else:          
            self.intent_classifier = nn.Sequential(
                nn.LayerNorm(config.hidden_size),
                nn.Linear(config.hidden_size, self.num_intent_labels)
            )
            self.slot_classifier = nn.Sequential(
                nn.LayerNorm(config.hidden_size),
                nn.Linear(config.hidden_size, self.num_slot_labels)
            )

        if self.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1] 

        intent = self.intent_classifier(pooled_output)
        slots = self.slot_classifier(sequence_output)

        return intent, slots
