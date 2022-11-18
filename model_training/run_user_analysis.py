import os
import pandas as pd
import numpy as np
import pickle

import torch
from custom_models import MoralGroundModel, SimpleLossCompute, MoralGroundModel_singleAuthor
from custom_models import NoamOpt, MoralRoTSelector, MoralGroundTrainer

import datasets
from torch.utils.data import Dataset, DataLoader

from datetime import datetime
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt

def get_consistency_data(_df):
    return datasets.Dataset.from_pandas(_df, features=datasets.Features({
                                'situation': datasets.Value(id=None, dtype='string'), 
                                'rot': datasets.Value(id=None, dtype='string'), 
                                'author': datasets.Value(id=None, dtype='string'), 
                                'moral-ground': datasets.Value(id=None, dtype='string'), 
                                'moral-ground-categories': datasets.Value(id=None, dtype='string'), 
                                'label': datasets.ClassLabel(num_classes=2, names=[0,1], names_file=None, id=None)}))

def batch_f1(tgt, pred):
    return f1_score(tgt, pred, average='macro')


def generate_report(data, model, BATCH_SIZE):
    
    num_steps = int(data.num_rows / BATCH_SIZE)
    # num_steps = 1 if num_steps == 0 else num_steps

    all_preds, all_rot_attentions = [], []
    all_mg_norm_max_indices = []
    for i in range(num_steps+1):
        model.eval()
        curr_batch = data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        if len(curr_batch['situation']) == 0:
            break

        batch_pred, batch_tgt = model.forward(curr_batch)

        rots = [elem.split('_____') for elem in curr_batch['rot']]
        mg_rot_categories = [elem.split('_____') for elem in curr_batch['moral-ground']]

        model_value_attn = model.attn.cpu()
        mg_attn = model.MG_MODEL.attn.cpu()
        for item_idx, attn_elem in enumerate(model_value_attn):
            _norms = [norm_elem.squeeze(-1) for norm_elem in attn_elem[0]]
            max_index = _norms.index(max(_norms))
            print('Curr situation:',curr_batch['situation'][item_idx])
            print('Redditor true judgment:',batch_tgt[item_idx])
            print('Model prediction:',int(torch.argmax(batch_pred.cpu(), dim=-1)[item_idx]))

            _mg_norms = [norm_elem.squeeze(-1) for norm_elem in mg_attn[item_idx][0]]
            all_mg_norm_max_indices.append(_mg_norms.index(max(_mg_norms)))
            for _val, _norm in zip(rots[item_idx], _norms):
                print(_val,float(_norm))
            print()
            for _sg, _norm in zip(mg_rot_categories[item_idx], _mg_norms):
                print(_sg, float(_norm))
    
    assert len(all_mg_norm_max_indices)%4 == 0
    norm_steps = int(len(all_mg_norm_max_indices)/4)
    all_cnt, match_cnt, abst_cnt = 0, 0, 0
    for i in range(norm_steps):
        all_cnt += 3
        _slice = all_mg_norm_max_indices[i*4:(i+1)*4]
        _reference = _slice[0]
        for elem in _slice[1:]:
            if elem == _reference:
                match_cnt += 1
        if _slice[-1] == _reference:
            abst_cnt += 1
    return match_cnt, all_cnt, abst_cnt



def run_eval(data, model, BATCH_SIZE, criterion):
    num_data = data.num_rows
    num_steps = int(num_data/BATCH_SIZE)
    total_f1 = 0
    total_loss = 0

    model.eval()

    all_tgts, all_preds = [], []

    for i in range(num_steps+1):
        curr_batch = data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        if len(curr_batch['situation']) == 0:
            break
        pred, tgt = model.forward(curr_batch)
        tgt_onehot = torch.nn.functional.one_hot(torch.LongTensor(tgt), num_classes=2).type(torch.FloatTensor).to(device)

        loss_func = SimpleLossCompute(criterion)
        loss = loss_func(pred, tgt_onehot)
        total_loss += loss

        all_tgts += tgt
        all_preds += list(torch.argmax(pred.cpu(), dim=-1))

    return batch_f1(all_tgts, all_preds), total_loss, all_tgts, all_preds

if __name__ == "__main__":
    # Setting up the device for GPU usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Current device:",device)

    # df = pd.read_csv(os.getcwd()+'/data/CommsRoTs__(Sub_Sochem)__(Redditor_top)__(Judge_trunc)__MoralGrounds__v2.tsv', sep='\t')
    df = pd.read_csv(os.getcwd()+'/data/extended-situation.tsv', sep='\t')
    df = df[df.label != -1] # drop INFO judgment
    df = df[df['split']=='test']


    NUM_ATTN_HEAD = 12
    d_model = 768
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    d_ground = 6
    all_authors_list = sorted(list(set(df.author)), key=lambda x:x.lower())

    #mg_type = 'random'
    # mg_type = 'comment-based-cluster'
    mg_type = 'situation-based-cluster'
    mg_model_name = 'MGTraining_situAttn_noLayernorm_' + mg_type
    #mg_model_name = 'AITA_MGTraining_' + mg_type
    model_name = 'RoTSelector_noLayernorm_keepMG_noScale_veryslowLR_1_'+mg_model_name
    #model_name = 'RoTSelector_'+mg_model_name
    layernorm_mg = False
    layernorm_rot = False

    all_orig_tgts, all_orig_preds = [], []
    all_para_tgts, all_para_preds = [], []
    all_reph_tgts, all_reph_preds = [], []
    all_abst_tgts, all_abst_preds = [], []
    all_match_cnt, all_all_cnt, all_abs_cnt = 0, 0, 0
    for idx, _author in enumerate(all_authors_list):
        test_data = get_consistency_data(df[df.author==_author])
        if len(test_data['situation']) == 0:
            continue
        #######################################################################################
        #################################### load MG model ####################################
        mg_model_path =  os.getcwd()+'/outputs/'+mg_model_name+'/'+_author
        mg_best_epoch, epoch_cnt = 0, 0

        for file in os.listdir(mg_model_path):
            filename = os.fsdecode(file)
            if filename.endswith('.pt'):
                _epoch = filename.split('/')[-1].split('.pt')[0].split('_')[-1]
                epoch_cnt += 1
                mg_best_epoch = _epoch
        # assert epoch_cnt == 1
        for file in os.listdir(mg_model_path):
            filename = os.fsdecode(file)
            if filename.endswith('best_epoch_list.pkl'):
                with open(mg_model_path+'/best_epoch_list.pkl', 'rb') as f:
                    _epoch_list = pickle.load(f)
                mg_best_epoch = _epoch_list[0]

        MG_MODEL = MoralGroundTrainer(NUM_ATTN_HEAD, d_model, d_ground, BATCH_SIZE, mg_type, do_layernorm=layernorm_mg).to(device)
        MG_MODEL.load_state_dict(torch.load(mg_model_path+'/state_dict_'+str(mg_best_epoch)+'.pt')) #### mind the epoch number
        #######################################################################################

        #######################################################################################
        #################################### load Model #######################################
        model = MoralRoTSelector(NUM_ATTN_HEAD, d_model, BATCH_SIZE, MG_MODEL, do_layernorm=layernorm_rot).to(device)

        model_path = os.getcwd()+'/outputs/'+model_name+'/'+_author
        print(model_path)

        best_epoch, epoch_cnt = 0, 0
        all_model_epochs = []
        for file in os.listdir(model_path):
            filename = os.fsdecode(file)
            if filename.endswith('.pt'):
                _epoch = filename.split('/')[-1].split('.pt')[0].split('_')[-1]
                all_model_epochs.append(int(_epoch))
                epoch_cnt += 1
                best_epoch = _epoch
        if epoch_cnt != 1:
            best_epoch = max(all_model_epochs)
        model.load_state_dict(torch.load(model_path+'/state_dict_'+str(best_epoch)+'.pt'))
        #######################################################################################

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=4e-5, betas=(0.9, 0.98), eps=1e-9)

        _author_test_f1, _, _author_tgts, _author_preds, = run_eval(test_data, model, BATCH_SIZE, criterion)
        print("Author %s, %d instances, Test F1: %.2f"%(_author, len(_author_tgts),  (_author_test_f1*100)))

        # check slice of 4
        assert len(_author_tgts) %4 == 0
        slice_steps = int(len(_author_tgts)/4)

        orig_tgts, orig_preds = [], []
        para_tgts, para_preds = [], []
        reph_tgts, reph_preds = [], []
        abst_tgts, abst_preds = [], []
        for i in range(slice_steps):
            orig_tgts.append(_author_tgts[i*4])
            orig_preds.append(_author_preds[i*4].item())

            para_tgts.append(_author_tgts[i*4+1])
            para_preds.append(_author_preds[i*4+1].item())

            reph_tgts.append(_author_tgts[i*4+2])
            reph_preds.append(_author_preds[i*4+2].item())

            abst_tgts.append(_author_tgts[i*4+3])
            abst_preds.append(_author_preds[i*4+3].item())

        orig_f1 = batch_f1(orig_tgts, orig_preds)
        para_f1 = batch_f1(para_tgts, para_preds)
        reph_f1 = batch_f1(reph_tgts, reph_preds)
        abst_f1 = batch_f1(abst_tgts, abst_preds)
        print("orig test F1: %.2f, paraphrase: %.2f, rephrase: %.2f, abstract: %.2f "%((orig_f1*100), (para_f1*100), (reph_f1*100), (abst_f1*100)))
    
        all_orig_tgts += orig_tgts
        all_orig_preds += orig_preds
        all_para_tgts += para_tgts
        all_para_preds += para_preds
        all_reph_tgts += reph_tgts
        all_reph_preds += reph_preds
        all_abst_tgts += abst_tgts
        all_abst_preds += abst_preds

        match_cnt, all_cnt, abst_cnt = generate_report(test_data, model, BATCH_SIZE)
        print('Author match count: %d out of %d'%(match_cnt, all_cnt))
        all_match_cnt += match_cnt
        all_all_cnt += all_cnt
        all_abs_cnt += abst_cnt


    all_orig_f1 = batch_f1(all_orig_tgts, all_orig_preds)
    all_para_f1 = batch_f1(all_para_tgts, all_para_preds)
    all_reph_f1 = batch_f1(all_reph_tgts, all_reph_preds)
    all_abst_f1 = batch_f1(all_abst_tgts, all_abst_preds)
    print('Overall')
    print("orig test F1: %.2f, paraphrase: %.2f, rephrase: %.2f, abstract: %.2f "%((all_orig_f1*100), (all_para_f1*100), (all_reph_f1*100), (all_abst_f1*100)))
    print('Attention weight matches: %d out of %d, abstract matches: %d'%(all_match_cnt, all_all_cnt, all_abs_cnt))
