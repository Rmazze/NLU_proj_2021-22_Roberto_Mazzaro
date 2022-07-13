from tqdm import tqdm

import numpy as np

import pandas as pd

import matplotlib as mp
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn

from torch.optim import Adam

from conll import evaluate
from sklearn.metrics import classification_report

import baseline_model as bm
import baseline_trainer as btr

'''
Use Adam optimizer as in laboratory 10
'''
def get_optimizer(model, learning_rate):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    return optimizer

'''
For both intent and slot use Cross Entropy Loss as standard loss function.
If use_crf = True for slot filling task use the negated result of the CRF layer forward that is a negative log likelihood. 
CRF forward compute the conditional log likelihood of a sequence of tags given emission scores.
'''
def get_losses(model, intent_label_id, slot_labels_ids, attention_masks, intent, slots, use_crf, pad_token, device):
    intent_loss = 0
    slot_loss = 0
    if intent_label_id is not None:
        intent_loss_fct = nn.CrossEntropyLoss().to(device)
        intent_loss += intent_loss_fct(intent, intent_label_id)

    if slot_labels_ids is not None:
        if use_crf:
            slots = slots.permute(2,0,1) 
            slot_labels_ids = slot_labels_ids.permute(1,0)
            attention_masks = attention_masks.permute(1,0)
            crf_slot_loss = model.crf(slots, slot_labels_ids, mask=attention_masks.byte(), reduction='mean')
            slot_loss += -1 * crf_slot_loss
        else:
            slot_loss_fct = nn.CrossEntropyLoss(ignore_index=pad_token).to(device)
            slot_loss += slot_loss_fct(slots, slot_labels_ids)
    return intent_loss, slot_loss

'''
Iterate over train dataloader and make prodictions, than compute losses and update the model's wheights.
Clip the gradient to avoid explosioning gradients
'''
def train_loop(model, dataloader, optimizer, max_grad_norm, use_crf, pad_token, device):
    model.train()

    loss_array = []
    for sample in dataloader:
        optimizer.zero_grad()

        input_ids = sample['utterances'].to(device)
        slots_len = sample['slots_len'].to(device)
        intent_label_id = sample['intents'].to(device)
        slot_labels_ids = sample['y_slots'].to(device)
        attention_masks = sample['attention_masks'].to(device)

        intent, slots = model(input_ids, slots_len)

        intent_loss, slot_loss = get_losses(model, intent_label_id, slot_labels_ids, attention_masks, intent, slots, use_crf, pad_token, device)
        total_loss = intent_loss + slot_loss

        loss_array.append(total_loss.item())
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
    return loss_array


'''
Iterate over evaluation dataloader and make prodictions, than compute losses and recover the original intents, slots and words from predicted and reference ids.
Evaluate the model performances, use the conll script to evaluate the slot filling task, while for intent classification use classification_report as requested.
'''
def eval_loop(model, dataloader, lang, use_crf, pad_token, device):
    model.eval()

    loss_array = []
        
    ref_intents = []
    hyp_intents = []
    
    ref_slots = []
    hyp_slots = []
    
    only_ref_slots = []
    only_hyp_slots = []

    with torch.no_grad():
        for sample in dataloader:
            
            input_ids = sample['utterances'].to(device)
            slots_len = sample['slots_len'].to(device)
            intent_label_id = sample['intents'].to(device)
            slot_labels_ids = sample['y_slots'].to(device)
            attention_masks = sample['attention_masks'].to(device)

            intent, slots = model(input_ids, slots_len)
            
            #losses are computed here before CRF step because we need emission scores and CRF layer directly output best tag sequence
            intent_loss, slot_loss = get_losses(model, intent_label_id, slot_labels_ids, attention_masks, intent, slots, use_crf, pad_token, device)
            total_loss = intent_loss + slot_loss
            loss_array.append(total_loss.item())

            out_intents = [lang.id2intent[x] for x in torch.argmax(intent, dim=1).tolist()] 
            gt_intents = [lang.id2intent[x] for x in intent_label_id.tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)
                       
            # Slot prediction
            if use_crf:
                slots = slots.permute(2,0,1) 
                attention_masks = attention_masks.permute(1,0)
                slots = model.crf.decode(slots, attention_masks)
                slot_preds = []
                for s in slots:
                    slot_preds.append(np.array(s))
            else:
                slots = torch.argmax(slots, dim=1)
                slot_preds = []
                for id_seq, seq in enumerate(slots):
                    length = slots_len.tolist()[id_seq]
                    to_decode = seq[:length].tolist()
                    slot_preds.append(to_decode)

            for id_seq, to_decode in enumerate(slot_preds):
                length = slots_len.tolist()[id_seq]
                utt_ids = input_ids[id_seq][:length].tolist()
                gt_ids = slot_labels_ids[id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                only_ref_slots.extend(gt_slots)
                utterance = [lang.id2word[elem] for elem in utt_ids]
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                    only_hyp_slots.append(lang.id2slot[elem])
                hyp_slots.append(tmp_seq)
    
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
def multiple_runs(runs, epochs, model, train_dataloader, dev_dataloader, test_dataloader, lang,
                        learning_rate, max_grad_norm, patience, use_crf, pad_token, device):

    slot_f1s, intent_acc = [], []
    for run in tqdm(range(0, runs)):
        model.apply(bm.init_weights)
        optimizer = btr.get_optimizer(model, learning_rate)
        model.to(device)

        best_f1 = 0
        pat = patience
        print("Run: ",  run)
        for epoch in range(1, epochs):
            loss = train_loop(model, train_dataloader, optimizer, max_grad_norm, use_crf, pad_token, device)
            if epoch %5 == 0:
                avg_train_loss = np.asarray(loss).mean()
                results_dev, intent_res, loss_dev, _, _, _, _ = eval_loop(model, dev_dataloader, lang, use_crf, pad_token, device)
                avg_dev_loss = np.asarray(loss_dev).mean()
                print("Epoch: ",  epoch)
                print("Average train loss: ", avg_train_loss)
                print("Average validation loss: ", avg_dev_loss)
                print('Intent Accuracy:', intent_res['accuracy'])
                if results_dev != None:
                    f1 = results_dev['total']['f']
                    if f1 > best_f1:
                        best_f1 = f1
                    else:
                        pat -= 1
                    print('Slot F1: ', results_dev['total']['f'])
                else:
                    print("Issues with f1 computation")

                if pat <= 0: # Early stoping with patient
                    print("no more improvement -> stop training of run: ", run)
                    break

        print("Best f1 score during training: ", best_f1)

        results_test, intent_test, loss_test, _, _, _, _ = eval_loop(model, test_dataloader, lang, use_crf, pad_token, device)
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
def single_run(epochs, model, train_dataloader, dev_dataloader, test_dataloader, lang, task,
                        learning_rate, max_grad_norm, patience, intent_labels, slot_labels, use_crf, dropout, pad_token, device):
    model.apply(bm.init_weights)
    optimizer = btr.get_optimizer(model, learning_rate)
    model.to(device)

    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    for epoch in tqdm(range(1, epochs)):
        loss = train_loop(model, train_dataloader, optimizer, max_grad_norm, use_crf, pad_token, device)

        if epoch %5 == 0:
            sampled_epochs.append(epoch)
            avg_train_loss = np.asarray(loss).mean()
            losses_train.append(avg_train_loss)
            results_dev, intent_res, loss_dev, _, _, _, _ = eval_loop(model, dev_dataloader, lang, use_crf, pad_token, device)
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
                else:
                    patience -= 1
                print('Slot F1: ', results_dev['total']['f'])
            else:
                print("Issues with f1 computation")

            if patience <= 0: # Early stoping with patient
                print("no more improvement -> stop training")
                break

    print("Best f1 score during training: ", best_f1)

    results_test, intent_test, loss_test, only_ref_slots, only_hyp_slots, ref_intents, hyp_intents = eval_loop(model, test_dataloader, lang, use_crf, pad_token, device)
    avg_test_loss = np.asarray(loss_test).mean()

    print("Final test results: ")
    print("Average test loss: ", avg_test_loss)
    print('Intent Accuracy:', intent_test['accuracy'])
    intent_test.pop('accuracy')
    intent_tbl = pd.DataFrame().from_dict(intent_test, orient='index')
    if use_crf:
        if dropout:
            result_path_intent = "./results/baseline_crf_dropout_intent_" + task + ".xlsx"
        else:
            result_path_intent = "./results/baseline_crf_layernorm_intent_" + task + ".xlsx"
    else:
        if dropout:
            result_path_intent = "./results/baseline_dropout_intent_" + task + ".xlsx"
        else:
            result_path_intent = "./results/baseline_layernorm_intent_" + task + ".xlsx"
    intent_tbl.to_excel(result_path_intent, float_format="%.2f", encoding='utf-8')
    if results_test != None:
        print('Slot F1: ', results_test['total']['f'])
        slot_tbl = pd.DataFrame().from_dict(results_test, orient='index')
        if use_crf:
            if dropout:
                result_path_slot = "./results/baseline_crf_dropout_slot_" + task + ".xlsx"
            else:
                result_path_slot = "./results/baseline_crf_layernorm_slot_" + task + ".xlsx"
        else:
            if dropout:
                result_path_slot = "./results/baseline_dropout_slot_" + task + ".xlsx"
            else:
                result_path_slot = "./results/baseline_layernorm_slot_" + task + ".xlsx"
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
            
