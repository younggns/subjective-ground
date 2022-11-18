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
    train_data = datasets.Dataset.from_pandas(_df[_df['split']=='train'], features=datasets.Features({
                            'situation': datasets.Value(id=None, dtype='string'), 
                            'rot': datasets.Value(id=None, dtype='string'), 
                            'author': datasets.Value(id=None, dtype='string'), 
                            'moral-ground': datasets.Value(id=None, dtype='string'), 
                            'moral-ground-commCluster': datasets.Value(id=None, dtype='string'), 
                            'situation-cluster': datasets.Value(id=None, dtype='int32'), 
                            'label': datasets.ClassLabel(num_classes=2, names=[0,1], names_file=None, id=None)}))
    eval_data = datasets.Dataset.from_pandas(_df[_df['split']=='dev'], features=datasets.Features({
                                'situation': datasets.Value(id=None, dtype='string'), 
                                'rot': datasets.Value(id=None, dtype='string'), 
                                'author': datasets.Value(id=None, dtype='string'), 
                                'moral-ground': datasets.Value(id=None, dtype='string'), 
                                'moral-ground-commCluster': datasets.Value(id=None, dtype='string'), 
                                'situation-cluster': datasets.Value(id=None, dtype='int32'), 
                                'label': datasets.ClassLabel(num_classes=2, names=[0,1], names_file=None, id=None)}))
    test_data = datasets.Dataset.from_pandas(_df[_df['split']=='test'], features=datasets.Features({
                                'situation': datasets.Value(id=None, dtype='string'), 
                                'rot': datasets.Value(id=None, dtype='string'), 
                                'author': datasets.Value(id=None, dtype='string'), 
                                'moral-ground': datasets.Value(id=None, dtype='string'), 
                                'moral-ground-commCluster': datasets.Value(id=None, dtype='string'),                                 
                                'situation-cluster': datasets.Value(id=None, dtype='int32'), 
                                'label': datasets.ClassLabel(num_classes=2, names=[0,1], names_file=None, id=None)}))

    return train_data, eval_data, test_data

def get_std_opt(model, d_model, factor, warmup_steps):
    return NoamOpt(d_model, factor, warmup_steps,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

def batch_f1(tgt, pred):
    return f1_score(tgt, pred, average='macro')

def run_epoch(data, model, BATCH_SIZE, criterion, optimizer, epoch, eval_steps, eval_data, model_path):
    start = datetime.now()
    num_data = data.num_rows
    num_steps = int(num_data/BATCH_SIZE)
    best_model_f1 = 0
    train_losses, train_accs = [], []
    valid_losses, valid_accs = [], []
    for i in range(num_steps):
        model.train()

        curr_batch = data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        pred, tgt = model.forward(curr_batch)
        tgt_onehot = torch.nn.functional.one_hot(torch.LongTensor(tgt), num_classes=2).type(torch.FloatTensor).to(device)
        loss_func = SimpleLossCompute(criterion, optimizer)
        loss = loss_func(pred, tgt_onehot)
        f1_score = batch_f1(tgt, list(torch.argmax(pred.cpu(), dim=-1)))

        train_losses.append(loss)
        train_accs.append(f1_score)

        if i % eval_steps == 1:
            eval_f1_score, eval_loss, _, _ = run_eval(eval_data, model, BATCH_SIZE, criterion)
            print("[Running eval at epoch %d step %d]  Train f1: %.5f, loss: %.5f  /  Eval f1: %.5f, loss: %.5f"%(int(epoch)+1, i, f1_score, loss, eval_f1_score, eval_loss))

            valid_accs.append(eval_f1_score)
            valid_losses.append(eval_loss)

            if eval_f1_score > best_model_f1:
                best_model_f1 = eval_f1_score
                torch.save(model.state_dict(), model_path+'/state_dict_'+str(epoch)+'.pt')
                
    end = datetime.now()
    print('Training time for epoch %d: %s'%(int(epoch), end-start))
    return train_losses, train_accs, valid_losses, valid_accs

def run_eval(data, model, BATCH_SIZE, criterion):
    num_data = data.num_rows
    num_steps = int(num_data/BATCH_SIZE)
    total_f1 = 0
    total_loss = 0

    model.eval()

    all_tgts, all_preds = [], []

    for i in range(num_steps):
        curr_batch = data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        pred, tgt = model.forward(curr_batch)
        tgt_onehot = torch.nn.functional.one_hot(torch.LongTensor(tgt), num_classes=2).type(torch.FloatTensor).to(device)

        loss_func = SimpleLossCompute(criterion)
        loss = loss_func(pred, tgt_onehot)
        total_loss += loss

        all_tgts += tgt
        all_preds += list(torch.argmax(pred.cpu(), dim=-1))

    return batch_f1(all_tgts, all_preds), total_loss/num_steps, all_tgts, all_preds

def create_figure(model_path, NUM_EPOCHS, all_train_losses, all_train_accs, all_valid_losses, all_valid_accs):

    pickle.dump(all_train_losses, open(model_path+'/all_train_losses.pkl', 'wb'))
    pickle.dump(all_train_accs, open(model_path+'/all_train_accs.pkl', 'wb'))
    pickle.dump(all_valid_losses, open(model_path+'/all_valid_losses.pkl', 'wb'))
    pickle.dump(all_valid_accs, open(model_path+'/all_valid_accs.pkl', 'wb'))  

    plt.plot(range(NUM_EPOCHS), [sum(elem)/len(elem) for elem in all_train_losses], label = "train loss")
    plt.plot(range(NUM_EPOCHS), [sum(elem)/len(elem) for elem in all_train_accs], label = "train acc")
    plt.plot(range(NUM_EPOCHS), [sum(elem)/len(elem) for elem in all_valid_losses], label = "valid loss")
    plt.plot(range(NUM_EPOCHS), [sum(elem)/len(elem) for elem in all_valid_accs], label = "valid acc")
    plt.legend()
    plt.savefig(model_path+'/loss_acc.png')
    plt.clf()
    plt.cla()
    plt.close()

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
    
    #mg_type = 'random'
    #mg_type = 'comment-based-cluster'
    mg_type = 'situation-based-cluster'
    mg_model_name = 'MGTraining_situAttn_noLayernorm_' + mg_type
    #mg_model_name = 'AITA_MGTraining_' + mg_type
    model_name = 'RoTSelector_noLayernorm_keepMG_noScale_woRoT_veryslowLR_5_'+mg_model_name
    do_layernorm_rot = False
    do_layernorm_mg = False
    _preds, _tgts = [], []
    for idx, _author in enumerate(all_authors_list):
        minidf = df[df['author']==_author]
        train_data, eval_data, test_data = convert_df_to_datasets(minidf)

        # load MG model
        mg_model_path =  os.getcwd()+'/outputs/'+mg_model_name+'/'+_author
        mg_best_eopch, epoch_cnt = 0, 0

        for file in os.listdir(mg_model_path):
            filename = os.fsdecode(file)
            if filename.endswith('.pt'):
                _epoch = filename.split('/')[-1].split('.pt')[0].split('_')[-1]
                epoch_cnt += 1
                mg_best_eopch = _epoch
        for file in os.listdir(mg_model_path):
            filename = os.fsdecode(file)
            if filename.endswith('best_epoch_list.pkl'):
                with open(mg_model_path+'/best_epoch_list.pkl', 'rb') as f:
                    _best_epoch_list = pickle.load(f)
                mg_best_epoch = _best_epoch_list[0]
        #assert epoch_cnt == 1
        #mg_best_epoch = 11
        MG_MODEL = MoralGroundTrainer(NUM_ATTN_HEAD, d_model, d_ground, BATCH_SIZE, mg_type, do_layernorm=do_layernorm_mg).to(device)
        MG_MODEL.load_state_dict(torch.load(mg_model_path+'/state_dict_'+str(mg_best_eopch)+'.pt')) #### mind the epoch number

        model = MoralRoTSelector(NUM_ATTN_HEAD, d_model, BATCH_SIZE, MG_MODEL, do_layernorm=do_layernorm_rot).to(device)

        num_training_steps = int(len(train_data) / BATCH_SIZE)
        warmup_steps = int(num_training_steps / 5)
        EVAL_STEPS = warmup_steps

        model_path = os.getcwd()+'/outputs/'+model_name+'/'+_author
        print(model_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        criterion = torch.nn.CrossEntropyLoss()
        # optimizer = get_std_opt(model, d_model*d_model, 1, warmup_steps)
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-6, betas=(0.9, 0.98), eps=1e-9)

        all_train_losses, all_train_accs, all_valid_losses, all_valid_accs = [], [], [], []
        best_epoch, best_valid_accs = 0, 0
        for epoch in range(NUM_EPOCHS):
            model.train()
            
            train_losses, train_accs, valid_losses, valid_accs = run_epoch(train_data, model, BATCH_SIZE,
                                    criterion, optimizer, str(epoch), EVAL_STEPS, eval_data, model_path)

            all_train_losses.append(train_losses)
            all_train_accs.append(train_accs)
            all_valid_losses.append(valid_losses)
            all_valid_accs.append(valid_accs)

            if sum(valid_accs)/len(valid_accs) >= best_valid_accs:
                best_epoch = epoch

        create_figure(model_path, NUM_EPOCHS, all_train_losses, all_train_accs, all_valid_losses, all_valid_accs)

        print('best epoch: %d'%best_epoch)
        best_epoch_model = MoralRoTSelector(NUM_ATTN_HEAD, d_model, BATCH_SIZE, MG_MODEL, do_layernorm=do_layernorm_rot).to(device)
        best_epoch_model.load_state_dict(torch.load(model_path+'/state_dict_'+str(best_epoch)+'.pt'))

        _author_test_f1, _, _author_tgts, _author_preds, = run_eval(test_data, best_epoch_model, BATCH_SIZE, criterion)
        print("Author %d, %s, Test F1: %.2f"%(idx, _author, (_author_test_f1*100)))

        _preds += _author_preds
        _tgts += _author_tgts

        # remove other state_dicts
        for file in os.listdir(model_path):
            filename = os.fsdecode(file)
            if filename.endswith('.pt'):
                _epoch = filename.split('/')[-1].split('.pt')[0].split('_')[-1]
                if int(_epoch) != best_epoch:
                    os.remove(model_path + '/' + filename)
    
    for name, param in model.named_parameters():
        print('name: ', name)
        print(type(param))
        print('param.shape: ', param.shape)
        print('param.requires_grad: ', param.requires_grad)
        print('=====')

    total_test_f1 = batch_f1(_tgts, _preds)
    print("Total Test F1: %.2f"%(total_test_f1*100))
