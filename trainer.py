from tqdm import tqdm

import numpy as np

import pandas as pd

import matplotlib as mp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn

from torch.optim import AdamW

from conll import evaluate
from sklearn.metrics import classification_report

from model import JointBERT
import trainer as tr


'''
Use AdamW optimizer is strongly suggested for Transformers models,
it uses a different way to compute weight decay wrt Adam that is more efficient. 
'''
def get_optimizer(model, learning_rate, adam_epsilon):
    no_decay = ['bias', 'LayerNorm.weight']
    no_decay_param = []
    rest_of_the_net_weights = []
    for name, param in model.named_parameters():
        if name in no_decay:
            no_decay_param.append(param)
        else:
            rest_of_the_net_weights.append(param)
    optimizer = AdamW([
        {'params': rest_of_the_net_weights},
        {'params': no_decay_param, 'weight_decay': 0.0}
        ], lr=learning_rate, eps=adam_epsilon)
    return optimizer

'''
For both intent and slot use Cross Entropy Loss as standard loss function.
If use_crf = True for slot filling task use the negated result of the CRF layer forward that is a negative log likelihood. 
CRF forward compute the conditional log likelihood of a sequence of tags given emission scores.
'''
def get_losses(model, intent_label_id, slot_labels_ids, attention_mask, intent, slots, num_intent_labels,  num_slot_labels, use_crf, pad_token, device):
    intent_loss = 0
    slot_loss = 0
    if intent_label_id is not None:
        intent_loss_fct = nn.CrossEntropyLoss().to(device)
        intent_loss += intent_loss_fct(intent.view(-1, num_intent_labels), intent_label_id.view(-1))
    # 2. Slot Softmax
    if slot_labels_ids is not None:
        if use_crf:
            crf_slot_loss = model.crf(slots, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
            slot_loss += -1 * crf_slot_loss  # negative log-likelihood
        else:
            slot_loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token).to(device)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = slots.view(-1, num_slot_labels)[active_loss]
                active_labels = slot_labels_ids.view(-1)[active_loss]
                slot_loss += slot_loss_fct(active_logits, active_labels)
            else:
                slot_loss += slot_loss_fct(slots.view(-1, num_slot_labels), slot_labels_ids.view(-1))
    return intent_loss, slot_loss

'''
Iterate over train dataloader and make prodictions, than compute losses and update the model's wheights.
Clip the gradient to avoid explosioning gradients
'''
def train_loop(model, dataloader, optimizer, max_grad_norm, intent_labels, slot_labels, use_crf, pad_token, device):
    model.train()
    num_intent_labels = len(intent_labels)
    num_slot_labels = len(slot_labels)
    loss_array = []
    for sample in dataloader:
        optimizer.zero_grad() # Zeroing the gradient

        input_ids = sample['input_ids'].to(device)
        attention_mask = sample['attention_mask'].to(device)
        token_type_ids = sample['token_type_ids'].to(device)
        intent_label_id = sample['intent_label_id'].to(device)
        slot_labels_ids = sample['slot_labels_ids'].to(device)

        intent, slots = model(input_ids, attention_mask, token_type_ids)

        intent_loss, slot_loss = get_losses(model, intent_label_id, slot_labels_ids, attention_mask, intent, slots, num_intent_labels,  num_slot_labels, use_crf, pad_token, device)
        total_loss = intent_loss + slot_loss

        loss_array.append(total_loss.item())
        total_loss.backward() # Compute the gradient, deleting the computational graph
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step() # Update the weights
    return loss_array

'''
Iterate over evaluation dataloader and make prodictions, than compute losses and recover the original intents, slots from predicted and reference ids.
Unfortunately Bert use uses word-piece tokenization, that is a lossy tokenization, so is not guaranteed to get the original sentence after detokenization,
since recover the original tokens is not necessarelly needed but we just need them to build an input in a valid form for the conll script,
we replace them with a 'place_holder' that is not considered during the evaluation.
Evaluate the model performances, use the conll script to evaluate the slot filling task, while for intent classification use classification_report as requested.
'''
def eval_loop(model, dataloader, intent_labels, slot_labels, use_crf, pad_token, tokenizer, device):
    model.eval()

    loss_array = []
        
    intent_preds = None
    slot_preds = None
    out_intent_label_ids = None
    out_slot_labels_ids = None


    num_intent_labels = len(intent_labels)
    num_slot_labels = len(slot_labels)
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in dataloader:
            
            input_ids = sample['input_ids'].to(device)
            attention_mask = sample['attention_mask'].to(device)
            token_type_ids = sample['token_type_ids'].to(device)
            intent_label_id = sample['intent_label_id'].to(device)
            slot_labels_ids = sample['slot_labels_ids'].to(device)


            intent, slots = model(input_ids, attention_mask, token_type_ids)

            intent_loss, slot_loss = get_losses(model, intent_label_id, slot_labels_ids, attention_mask, intent, slots, num_intent_labels,  num_slot_labels, use_crf, pad_token, device)
            total_loss = intent_loss + slot_loss
            loss_array.append(total_loss.item())

            # Intent prediction
            if intent_preds is None:
                intent_preds = intent.detach().cpu().numpy()
                out_intent_label_ids = intent_label_id.detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent.detach().cpu().numpy(), axis=0)
                out_intent_label_ids = np.append(out_intent_label_ids, intent_label_id.detach().cpu().numpy(), axis=0)
            

            # Slot prediction
            if slot_preds is None:
                if use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(model.crf.decode(slots))
                else:
                    slot_preds = slots.detach().cpu().numpy()
                out_slot_labels_ids = slot_labels_ids.detach().cpu().numpy()
            else:
                if use_crf:
                    slot_preds = np.append(slot_preds, np.array(model.crf.decode(slots)), axis=0)
                else:
                    slot_preds = np.append(slot_preds, slots.detach().cpu().numpy(), axis=0)
                out_slot_labels_ids = np.append(out_slot_labels_ids, slot_labels_ids.detach().cpu().numpy(), axis=0)

    # Intent result
    intent_preds = np.argmax(intent_preds, axis=1)
    intent_label_map = {i: label for i, label in enumerate(intent_labels)}
    ref_intents = []
    hyp_intents = []
    for i in range(out_intent_label_ids.shape[0]):
        ref_intents.append(intent_label_map[out_intent_label_ids[i]])
        hyp_intents.append(intent_label_map[intent_preds[i]])  

    # Slot result
    if not use_crf:
        slot_preds = np.argmax(slot_preds, axis=2)
    slot_label_map = {i: label for i, label in enumerate(slot_labels)}
    ref_slots = [[] for _ in range(out_slot_labels_ids.shape[0])]
    hyp_slots = [[] for _ in range(out_slot_labels_ids.shape[0])]
    only_ref_slots = []
    only_hyp_slots = []
    for i in range(out_slot_labels_ids.shape[0]):
        for j in range(out_slot_labels_ids.shape[1]):
            if out_slot_labels_ids[i, j] != pad_token and out_slot_labels_ids[i, j] != tokenizer.unk_token: 
                ref_slot = slot_label_map[out_slot_labels_ids[i][j]]
                hyp_slot = slot_label_map[slot_preds[i][j]]
                if ref_slot in slot_labels:
                    ref_slots[i].append(('place_holder', ref_slot))
                    only_ref_slots.append(ref_slot)
                else:
                    ref_slots[i].append(('place_holder', 'O'))
                    only_ref_slots.append('O')

                if hyp_slot in slot_labels:
                    hyp_slots[i].append(('place_holder', hyp_slot))
                    only_hyp_slots.append(hyp_slot)
                else:
                    hyp_slots[i].append(('place_holder', 'O'))
                    only_hyp_slots.append('O')
    
    results = None
    try:            
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predics a class that is not in REF
        print(ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
      
    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    return results, report_intent, loss_array, only_ref_slots, only_hyp_slots, ref_intents, hyp_intents

'''
Since we have small dataset we train and test the model from scratch several times, in that way we can average results to obtain more reliable results
and compute the standard deviation to see how much they oscillate.
'''
def multiple_runs(runs, epochs, config, train_dataloader, dev_dataloader, test_dataloader, 
                        dropout_rate, learning_rate, adam_epsilon, max_grad_norm, intent_labels, slot_labels, use_crf, dropout, pad_token, tokenizer, device):

    slot_f1s, intent_acc = [], []
    for run in tqdm(range(0, runs)):
        model = JointBERT.from_pretrained('bert-base-uncased', config=config,
            dropout_rate=dropout_rate, use_crf=use_crf, dropout=dropout, intent_labels=intent_labels, slot_labels=slot_labels)
        optimizer = tr.get_optimizer(model, learning_rate, adam_epsilon)
        model.to(device)

        losses_train = []
        losses_dev = []
        sampled_epochs = []
        best_f1 = 0
        print("Run: ",  run)
        for epoch in range(1, epochs):
            loss = train_loop(model, train_dataloader, optimizer, max_grad_norm, intent_labels, slot_labels, use_crf, pad_token, device)

            sampled_epochs.append(epoch)
            avg_train_loss = np.asarray(loss).mean()
            losses_train.append(avg_train_loss)
            results_dev, intent_res, loss_dev, _, _, _, _ = eval_loop(model, dev_dataloader, intent_labels, slot_labels, use_crf, pad_token, tokenizer, device)
            avg_dev_loss = np.asarray(loss_dev).mean()
            losses_dev.append(avg_dev_loss)
            print("Epoch: ",  epoch)
            print("Average train loss: ", avg_train_loss)
            print("Average validation loss: ", avg_dev_loss)
            print('Intent Accuracy:', intent_res['accuracy'])
            if results_dev != None:
                f1 = results_dev['total']['f']
                if f1 > best_f1:
                    best_f1 = f1
                print('Slot F1: ', results_dev['total']['f'])
            else:
                print("Issues with f1 computation")

        print("Best f1 score during training: ", best_f1)

        results_test, intent_test, loss_test, _, _, _, _ = eval_loop(model, test_dataloader, intent_labels, slot_labels, use_crf, pad_token, tokenizer, device)
        avg_test_loss = np.asarray(loss_test).mean()
        intent_acc.append(intent_test['accuracy'])
        print("Test results of run: ",  run)
        print("Average test loss: ", avg_test_loss)
        print('Intent Accuracy:', intent_test['accuracy'])
        if results_test != None:
            slot_f1s.append(results_test['total']['f'])
            print('Slot F1: ', results_test['total']['f'])
        else:
            print("Issues with f1 computation")

    slot_f1s = np.asarray(slot_f1s)
    intent_acc = np.asarray(intent_acc)
    print("Average results over all runs:")
    print('Intent Acc', round(intent_acc.mean(), 3), '+-', round(slot_f1s.std(), 3))
    print('Slot F1', round(slot_f1s.mean(),3), '+-', round(slot_f1s.std(),3))

'''
Do a single run of training and testing, useful to debug.
Moreover we print losses graph, confusion matrixes and table results; we don't do that in multiple runs to not obtain too much data.
'''
def single_run(epochs, config, train_dataloader, dev_dataloader, test_dataloader, task,
                        dropout_rate, learning_rate, adam_epsilon, max_grad_norm, intent_labels, slot_labels, use_crf, dropout, pad_token, tokenizer, device):
    model = JointBERT.from_pretrained('bert-base-uncased', 
        config=config, dropout_rate=dropout_rate, use_crf=use_crf, dropout=dropout, intent_labels=intent_labels, slot_labels=slot_labels)
    optimizer = tr.get_optimizer(model, learning_rate, adam_epsilon)
    model.to(device)

    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    for epoch in tqdm(range(1, epochs)):
        loss = train_loop(model, train_dataloader, optimizer, max_grad_norm, intent_labels, slot_labels, use_crf, pad_token, device)

        sampled_epochs.append(epoch)
        avg_train_loss = np.asarray(loss).mean()
        losses_train.append(avg_train_loss)
        results_dev, intent_res, loss_dev, _, _, _, _ = eval_loop(model, dev_dataloader, intent_labels, slot_labels, use_crf, pad_token, tokenizer, device)
        avg_dev_loss = np.asarray(loss_dev).mean()
        losses_dev.append(avg_dev_loss)
        print("Epoch: ",  epoch)
        print("Average validation loss: ", avg_dev_loss)
        print("Average train loss: ", avg_train_loss)
        print('Intent Accuracy:', intent_res['accuracy'])
        if results_dev != None:
            f1 = results_dev['total']['f']
            if f1 > best_f1:
                best_f1 = f1
            print('Slot F1: ', results_dev['total']['f'])
        else:
            print("Issues with f1 computation")

    print("Best f1 score during training: ", best_f1)

    results_test, intent_test, loss_test, only_ref_slots, only_hyp_slots, ref_intents, hyp_intents = eval_loop(model, 
        test_dataloader, intent_labels, slot_labels, use_crf, pad_token, tokenizer, device)
    avg_test_loss = np.asarray(loss_test).mean()

    print("Final test results: ")
    print("Average test loss: ", avg_test_loss)

    print('Intent Accuracy:', intent_test['accuracy'])
    intent_test.pop('accuracy')
    intent_tbl = pd.DataFrame().from_dict(intent_test, orient='index')
    if use_crf:
        if dropout: 
            result_path_intent = "./results/jointbert_crf_dropout_intent_" + task + ".xlsx"
        else:
            result_path_intent = "./results/jointbert_crf_layernorm_intent_" + task + ".xlsx"
    else:
        if dropout:
            result_path_intent = "./results/jointbert_dropout_intent_" + task + ".xlsx"
        else:
            result_path_intent = "./results/jointbert_layernorm_intent_" + task + ".xlsx"
    intent_tbl.to_excel(result_path_intent, float_format="%.2f", encoding='utf-8')

    if results_test != None:
        print('Slot F1: ', results_test['total']['f'])
        slot_tbl = pd.DataFrame().from_dict(results_test, orient='index')
        if use_crf:
            if dropout:
                result_path_slot = "./results/jointbert_crf_dropout_slot_" + task + ".xlsx"
            else:
                result_path_slot = "./results/jointbert_crf_layernorm_slot_" + task + ".xlsx"
        else:
            if dropout:
                result_path_slot = "./results/jointbert_dropout_slot_" + task + ".xlsx"
            else:
                result_path_slot = "./results/jointbert_layernorm_slot_" + task + ".xlsx"
        slot_tbl.to_excel(result_path_slot, float_format="%.2f", encoding='utf-8')
    else:
        print("Issues with f1 computation")

    cm_intent = confusion_matrix(y_true=ref_intents, y_pred=hyp_intents, labels=intent_labels)
    cm_slot = confusion_matrix(y_true=only_ref_slots, y_pred=only_hyp_slots, labels=slot_labels)

    plt.figure(figsize = (10,10))
    plt.imshow(cm_slot, norm = mp.colors.Normalize(vmin = 0, vmax = 255))
    
    plt.figure(figsize = (10,10))
    plt.imshow(cm_intent, norm = mp.colors.Normalize(vmin = 0, vmax = 255))

    plt.figure(num = 3, figsize=(8, 5)).patch.set_facecolor('white')
    plt.title('Train and Dev Losses')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.plot(sampled_epochs, losses_train, label='Train loss')
    plt.plot(sampled_epochs, losses_dev, label='Dev loss')
    plt.legend()

    plt.show()
            
