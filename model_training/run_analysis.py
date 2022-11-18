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
                                'rot-label': datasets.Value(id=None, dtype='string'), 
                                'label': datasets.ClassLabel(num_classes=2, names=[0,1], names_file=None, id=None)}))

def get_const_report_data(_df):
    return datasets.Dataset.from_pandas(_df, features=datasets.Features({
                                'situation': datasets.Value(id=None, dtype='string'), 
                                'rot': datasets.Value(id=None, dtype='string'), 
                                'author': datasets.Value(id=None, dtype='string'), 
                                'moral-ground': datasets.Value(id=None, dtype='string'), 
                                'label': datasets.ClassLabel(num_classes=2, names=[0,1], names_file=None, id=None)}))

def compute_consistency(data, model, BATCH_SIZE):
    
    num_steps = int(data.num_rows / BATCH_SIZE)
    # num_steps = 1 if num_steps == 0 else num_steps

    all_preds, all_rot_attentions = [], []
    for i in range(num_steps):
        model.eval()
        curr_batch = data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        batch_pred, batch_tgt = model.forward(curr_batch)

        rot_judgments = [elem.split('_____') for elem in curr_batch['rot-label']]
        rots = [elem.split('_____') for elem in curr_batch['rot']]
        max_rot_judgments = []

        mg_attn = model.MG_MODEL.attn.cpu()
        mg_rot_categories = [elem.split('_____') for elem in curr_batch['moral-ground']]


        model_value_attn = model.attn.cpu()
        for item_idx, attn_elem in enumerate(model_value_attn):
            _norms = [norm_elem.squeeze(-1) for norm_elem in attn_elem[0]]
            max_index = _norms.index(max(_norms))
            max_rot_judgments.append(int(rot_judgments[item_idx][max_index]))
            # print('Curr situation:',curr_batch['situation'][item_idx])
            # print('Redditor true judgment:',batch_tgt[item_idx])
            # print('Model prediction:',int(torch.argmax(batch_pred.cpu(), dim=-1)[item_idx]))

            # _mg_norms = [norm_elem.squeeze(-1) for norm_elem in mg_attn[item_idx][0]]
            # for _val, _norm in zip(rots[item_idx], _norms):
            #     print(_val,float(_norm))
            # for _sg, _norm in zip(mg_rot_categories[item_idx], _mg_norms):
            #     print(_sg, float(_norm))
        
        all_preds += [int(elem) for elem in torch.argmax(batch_pred.cpu(), dim=-1)]
        all_rot_attentions += max_rot_judgments
    
    matching_item_count = 0
    for elem1, elem2 in zip(all_preds, all_rot_attentions):
        if elem1 == elem2:
            matching_item_count += 1
    return matching_item_count, len(all_preds)

def generate_report(data, model, BATCH_SIZE):
    
    num_steps = int(data.num_rows / BATCH_SIZE)
    # num_steps = 1 if num_steps == 0 else num_steps

    all_preds, all_rot_attentions = [], []
    for i in range(num_steps):
        model.eval()
        curr_batch = data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
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
            for _val, _norm in zip(rots[item_idx], _norms):
                print(_val,float(_norm))
            print()
            for _sg, _norm in zip(mg_rot_categories[item_idx], _mg_norms):
                print(_sg, float(_norm))

def mf_correspondence(data, model, BATCH_SIZE):
    # largest attended moral ground's moral foundations score: sanctity:3,care:1_____authority:1,care:2__
    # largest attended RoT moral foundations: care-harm_____care-harm|fairness-cheating_
    # match -> 1, no match -> 0
    # highest mg mf matches -> high match (2)

    model.eval()
    num_steps = int(data.num_rows / BATCH_SIZE)

    all_mg_cat, all_rot_cat = [], []
    for i in range(num_steps):
        curr_batch = data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        batch_pred, _ = model.forward(curr_batch)

        mg_attn = model.MG_MODEL.attn.cpu()
        mg_rot_categories = [elem.split('_____') for elem in curr_batch['moral-ground-categories']]
        max_mg_rot_categories = []
        for item_idx, attn_elem in enumerate(mg_attn):
            _norms = [norm_elem.squeeze(-1) for norm_elem in attn_elem[0]]
            max_index = _norms.index(max(_norms))
            max_mg_rot_categories.append(mg_rot_categories[item_idx][max_index])


        model_value_attn = model.attn.cpu()
        rot_moral_foundations = [elem.split('_____') for elem in curr_batch['rot-moral-foundations']]
        max_rot_moral_foundations = []
        for item_idx, attn_elem in enumerate(model_value_attn):
            _norms = [norm_elem.squeeze(-1) for norm_elem in attn_elem[0]]
            max_index = _norms.index(max(_norms))
            max_rot_moral_foundations.append(rot_moral_foundations[item_idx][max_index])

        all_mg_cat += max_mg_rot_categories
        all_rot_cat += max_rot_moral_foundations
    
    match_items_cnt, high_match_items_cnt = 0, 0
    for elem1, elem2 in zip(all_mg_cat, all_rot_cat):
        _mgs = [item for mg_elem in elem1.split(',') for item in mg_elem.split(':')[0]]
        _mfs = [item for mg_elem in elem1.split('|') for item in mg_elem.split('-')[0]]

        _match = 1 if len(set(_mgs).intersection(set(_mfs)))>=1 else 0
        _high_match = 1 if _mgs[0] in _mfs else 0
        
        match_items_cnt += _match
        high_match_items_cnt += _high_match
    
    return high_match_items_cnt, match_items_cnt, len(all_mg_cat)

if __name__ == "__main__":
    # Setting up the device for GPU usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Current device:",device)

    df = pd.read_csv(os.getcwd()+'/data/CommsRoTs__(Sub_Sochem)__(Redditor_top)__(Judge_trunc)__MoralGrounds__v2.tsv', sep='\t')
    df = df[df.label != -1] # drop INFO judgment
    df = df[df['split']=='test']

    consistency_df = pd.read_csv(os.getcwd()+'/data/consistency-test.tsv', sep='\t')
    consistency_df = consistency_df[consistency_df.label != -1] # drop INFO judgment

    NUM_ATTN_HEAD = 12
    d_model = 768
    BATCH_SIZE = 16
    NUM_EPOCHS = 10
    d_ground = 6
    all_authors_list = list(set(df.author))

    # mg_type = 'random'
    # mg_type = 'comment-based-cluster'
    mg_type = 'situation-based-cluster'
    mg_model_name = 'MGTraining_situAttn_noLayernorm_' + mg_type

    model_name = 'RoTSelector_noLayernorm_keepMG_noScale_veryslowLR_1_'+mg_model_name
    
    mg_norm = False
    rot_norm = False

    _consistent_items, _consistent_totals, _high_match_items, _match_items, _match_totals = [], [], [], [], []
    for idx, _author in enumerate(all_authors_list):
        consistency_test = get_consistency_data(consistency_df[consistency_df['author']==_author])
        consistency_report = get_const_report_data(df[df['author']==_author])
        # correspondence_test = get_correspondencd_data(correspondence_df[correspondence_df['author']==_author])

        #######################################################################################
        #################################### load MG model ####################################
        mg_model_path =  os.getcwd()+'/outputs/'+mg_model_name+'/'+_author
        mg_best_eopch, epoch_cnt = 0, 0

        for file in os.listdir(mg_model_path):
            filename = os.fsdecode(file)
            if filename.endswith('.pt'):
                _epoch = filename.split('/')[-1].split('.pt')[0].split('_')[-1]
                epoch_cnt += 1
                mg_best_eopch = _epoch
        #assert epoch_cnt == 1
        for file in os.listdir(mg_model_path):
            filename = os.fsdecode(file)
            if filename.endswith('best_epoch_list.pkl'):
                with open(mg_model_path+'/best_epoch_list.pkl', 'rb') as f:
                    mg_best_epoch = pickle.load(f)[0]
        MG_MODEL = MoralGroundTrainer(NUM_ATTN_HEAD, d_model, d_ground, BATCH_SIZE, mg_type, do_layernorm=mg_norm).to(device)
        MG_MODEL.load_state_dict(torch.load(mg_model_path+'/state_dict_'+str(mg_best_eopch)+'.pt')) #### mind the epoch number
        #######################################################################################


        #######################################################################################
        #################################### load Model #######################################
        model = MoralRoTSelector(NUM_ATTN_HEAD, d_model, BATCH_SIZE, MG_MODEL, do_layernorm=rot_norm).to(device)

        model_path = os.getcwd()+'/outputs/'+model_name+'/'+_author
        print(model_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

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

        num_consistent_items, num_total_items = compute_consistency(consistency_test, model, BATCH_SIZE)
        generate_report(consistency_report, model, BATCH_SIZE )
        # num_high_match_items, num_match_items, num_total_match = mf_correspondence(correspondence_test, model, BATCH_SIZE)
        print('Author: %s, Consistency: %d consistent items out of %d'%(_author, num_consistent_items, num_total_items))
        # print('High correspondence: %d, Correspondence: %d, out of %d'%(num_high_match_items, num_match_items, num_total_match))
        print()

        _consistent_items += [num_consistent_items]
        _consistent_totals += [num_total_items]
        # _high_match_items += [num_high_match_items]
        # _match_items += [num_match_items]
        # _match_totals += [num_total_match]
    
    print('Overall numbers:')
    print('Consistency: %d consistent items out of %d'%(sum(_consistent_items), sum(_consistent_totals)))
    # print('High correspondence: %d, Correspondence: %d, out of %d'%(sum(_high_match_items), sum(_match_items), sum(_match_totals)))
