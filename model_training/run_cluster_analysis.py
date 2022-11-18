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

def convert_df_to_datasets(_df):
    test_data = datasets.Dataset.from_pandas(_df[_df['split']=='test'], features=datasets.Features({
                                'situation': datasets.Value(id=None, dtype='string'), 
                                'rot': datasets.Value(id=None, dtype='string'), 
                                'author': datasets.Value(id=None, dtype='string'), 
                                'moral-ground': datasets.Value(id=None, dtype='string'), 
                                'moral-ground-commCluster': datasets.Value(id=None, dtype='string'),                                 
                                'situation-cluster': datasets.Value(id=None, dtype='int32'), 
                                'label': datasets.ClassLabel(num_classes=2, names=[0,1], names_file=None, id=None)}))

    return test_data

def batch_f1(tgt, pred):
    return f1_score(tgt, pred, average='macro')

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

    return batch_f1(all_tgts, all_preds), total_loss/num_steps, all_tgts, all_preds

if __name__ == "__main__":
    # Setting up the device for GPU usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Current device:",device)

    df = pd.read_csv(os.getcwd()+'/data/CommsRoTs__(Sub_Sochem)__(Redditor_top)__(Judge_trunc)__MoralGrounds__v2.tsv', sep='\t')

    df = df[df.label != -1] # drop INFO judgment

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

    model_name = 'RoTSelector_noLayernorm_keepMG_noScale_veryslowLR_5_'+mg_model_name

    do_layernorm_rot = False
    do_layernorm_mg = False

    cluster_to_preds_tgts = {key:[ [],[] ] for key in range(20)}
    _preds, _tgts = [], []
    for idx, _author in enumerate(all_authors_list):
        minidf = df[df['author']==_author]
        test_data = convert_df_to_datasets(minidf)

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

        for file in os.listdir(mg_model_path):
            filename = os.fsdecode(file)
            if filename.endswith('best_epoch_list.pkl'):
                with open(mg_model_path+'/best_epoch_list.pkl', 'rb') as f:
                    _best_epoch_list = pickle.load(f)
                mg_best_epoch = _best_epoch_list[-1]

        MG_MODEL = MoralGroundTrainer(NUM_ATTN_HEAD, d_model, d_ground, BATCH_SIZE, mg_type, do_layernorm=do_layernorm_mg).to(device)
        MG_MODEL.load_state_dict(torch.load(mg_model_path+'/state_dict_'+str(mg_best_epoch)+'.pt')) #### mind the epoch number
        #######################################################################################


        #######################################################################################
        #################################### load Model #######################################
        model = MoralRoTSelector(NUM_ATTN_HEAD, d_model, BATCH_SIZE, MG_MODEL, do_layernorm=do_layernorm_rot).to(device)

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
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-6, betas=(0.9, 0.98), eps=1e-9)

        _author_test_f1, _, _author_tgts, _author_preds, = run_eval(test_data, model, BATCH_SIZE, criterion)
        print("Author %s with %d instances, Test F1: %.2f"%(_author, len(_author_tgts), (_author_test_f1*100)))

        _preds += _author_preds
        _tgts += _author_tgts

        for _inst_idx, _cluster_idx in enumerate(test_data['situation-cluster']):
            cluster_to_preds_tgts[_cluster_idx][0].append(_author_preds[_inst_idx].item())
            cluster_to_preds_tgts[_cluster_idx][1].append(_author_tgts[_inst_idx])
        author_cluster_preds = {key:[ [],[] ] for key in range(20)}
        for _inst_idx, _cluster_idx in enumerate(test_data['situation-cluster']):
            author_cluster_preds[_cluster_idx][0].append(_author_preds[_inst_idx].item())
            author_cluster_preds[_cluster_idx][1].append(_author_tgts[_inst_idx])
        for _cluster_idx in range(20):
            _f1 = batch_f1(author_cluster_preds[_cluster_idx][1], author_cluster_preds[_cluster_idx][0])
            print('Author %s, cluster %d, test f1: %.2f'%(_author, _cluster_idx, _f1*100))
        
    total_test_f1 = batch_f1(_tgts, _preds)
    print("Total Test F1: %.2f"%(total_test_f1*100))

    for _cluster_idx in cluster_to_preds_tgts:
        cluster_test_f1 =  batch_f1(cluster_to_preds_tgts[_cluster_idx][1], cluster_to_preds_tgts[_cluster_idx][0])
        print("Cluster %d with %d instances, F1: %.2f"%(_cluster_idx, len(cluster_to_preds_tgts[_cluster_idx][0]), (cluster_test_f1*100)))
