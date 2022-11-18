import math, copy

import torch
import transformers
from transformers import AutoTokenizer, AutoModel

class BaselineClassifier(torch.nn.Module):
    def __init__(self, d_model, nbatches, device, use_values=False, use_MGs=False, d_grounds=12,
                 n_classes=2, n_values=5, dropout=0.3, checkpoint="distilbert-base-uncased", MAX_LEN=32):
        super(BaselineClassifier, self).__init__()
        
        self.d_model = d_model
        self.n_classes = n_classes
        self.nbatches = nbatches
        self.n_values = n_values
        self.d_grounds = d_grounds
        self.device = device

        # self.dropout = dropout
        self.dropout = torch.nn.Dropout(p=dropout).to(device)
        self.MAX_LEN = MAX_LEN

        self.pre_classifier = torch.nn.Linear(d_model, d_model).to(device)
        self.classifier = torch.nn.Linear(d_model, n_classes).to(device)
        
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.encode_model = AutoModel.from_pretrained(checkpoint).to(device)

    def forward(self, batch):
        input_situ = self.tokenizer(batch['situation'], None, add_special_tokens=True, 
                                    max_length=self.MAX_LEN, padding=True, truncation=True, return_tensors='pt')
        output_situ = self.encode_model(input_ids=input_situ.input_ids.to(self.device), 
                                    attention_mask=input_situ.attention_mask.to(self.device))
        hidden_situ = output_situ.last_hidden_state[:,0,:] # getting the last hidden state of the <s> token representation

        # # what I've been doing:
        logits = self.classifier(hidden_situ)
        self.pred = torch.nn.functional.softmax(logits, dim=-1)
        return self.pred, batch['label']

        # # change logits:
        # logits = self.classifier(hidden_situ)
        # return logits.view(-1, self.n_classes), batch['label']

        # # pre_classifier and dropout
        # logits = self.classifier(self.dropout(nn.ReLU()(self.pre_classifier(hidden_situ))))
        # return logits.view(-1, self.n_classes), batch['label']



class BaselineClassifierGlobal(torch.nn.Module):
    def __init__(self, d_model, nbatches, use_values=False, use_MGs=False, d_grounds=12, device='cuda',
                 n_classes=2, n_values=5, dropout=0.3, checkpoint="distilbert-base-uncased", MAX_LEN=32):
        super(BaselineClassifierGlobal, self).__init__()
        
        self.d_model = d_model
        self.n_classes = n_classes
        self.nbatches = nbatches
        self.n_values = n_values
        self.d_grounds = d_grounds
        self.device = device

        # self.dropout = dropout
        self.dropout = torch.nn.Dropout(p=dropout).to(device)
        self.MAX_LEN = MAX_LEN

        self.pre_classifier = torch.nn.Linear(d_model+5, d_model).to(device)
        self.classifier = torch.nn.Linear(d_model, n_classes).to(device)
        
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.encode_model = AutoModel.from_pretrained(checkpoint).to(device)

    def author_encoding(self, number):
        bits = []
        while number > 0:
            bits.append(number%2)
            number = int(number/2)
        if len(bits) < 5:
            for i in range(5-len(bits)):
                bits.append(0)
        bits = bits[::-1]
        return torch.FloatTensor(bits)

    def forward(self, batch):
        self.nbatches = len(batch['situation'])

        input_situ = self.tokenizer(batch['situation'], None, add_special_tokens=True, 
                                    max_length=self.MAX_LEN, padding=True, truncation=True, return_tensors='pt')
        output_situ = self.encode_model(input_ids=input_situ.input_ids.to(self.device), 
                                    attention_mask=input_situ.attention_mask.to(self.device))
        hidden_situ = output_situ.last_hidden_state[:,0,:] # getting the last hidden state of the <s> token representation

        author_onehot = self.author_encoding(int(batch['author-index'][0])).to(self.device)
        input_author = author_onehot.repeat(self.nbatches,1)
        
        # # what I've been doing:
        self.logits = self.pre_classifier(torch.cat((input_author, hidden_situ), 1))
        self.logits = self.dropout(torch.nn.ReLU()(self.logits))
        self.pred = torch.nn.functional.softmax(self.classifier(self.logits), dim=-1)
        return self.pred, batch['label']


class BaselineClassifierTuned(torch.nn.Module):
    def __init__(self, d_model, nbatches, baseline_Model, device='cuda',
                 n_classes=2,dropout=0.3, checkpoint="distilbert-base-uncased", MAX_LEN=32):
        super(BaselineClassifierTuned, self).__init__()
        
        self.d_model = d_model
        self.n_classes = n_classes
        self.nbatches = nbatches
        self.device = device

        self.baseline_Model = baseline_Model

        self.pre_classifier = torch.nn.Linear(d_model, d_model).to(device)
        self.classifier = torch.nn.Linear(d_model, n_classes).to(device)
        
    def forward(self, batch):
        _, _ = self.baseline_Model(batch)
        self._curr_batch_situation = self.baseline_Model.situation
        pre_ = self.pre_classifier(self._curr_batch_situation)
        self.pred = torch.nn.functional.softmax(self.classifier(pre_), dim=-1)
        return self.pred, batch['label']

class BaselineClassifierTunedRoT(torch.nn.Module):
    def __init__(self, d_model, nbatches, baseline_Model, device='cuda',n_values=5,h=12,
                 n_classes=2,dropout=0.3, checkpoint="distilbert-base-uncased", MAX_LEN=32):
        super(BaselineClassifierTunedRoT, self).__init__()
        
        self.d_model = d_model
        self.h = h
        self.MAX_LEN = MAX_LEN
        self.n_classes = n_classes
        self.nbatches = nbatches
        self.device = device
        self.n_values = n_values

        self.baseline_Model = baseline_Model

        self.attn_layer = MultiHeadedAttention(h, d_model, device).to(device)

        self.pre_classifier = torch.nn.Linear(d_model*2, d_model).to(device)
        self.classifier = torch.nn.Linear(d_model, n_classes).to(device)
    
    def _get_moral_rot(self, batch):
        _rot_groups = [item for elem in batch['rot'] for item in elem.split('_____')]

        input_rot = self.baseline_Model.tokenizer(_rot_groups, None, add_special_tokens=True, 
                                   max_length=self.MAX_LEN, padding=True, truncation=True, return_tensors='pt')
        output_rot = self.baseline_Model.encode_model(input_ids=input_rot.input_ids.to(self.device), 
                                    attention_mask=input_rot.attention_mask.to(self.device))
        hidden_rot = output_rot.last_hidden_state[:,0,:]
        hidden_rot = torch.reshape(hidden_rot, (self.nbatches, self.n_values, self.d_model)) # (batch X n_values X d_model)
        
        return hidden_rot
    def forward(self, batch):
        _, _ = self.baseline_Model(batch)
        self._curr_batch_situation = self.baseline_Model.situation
        self.MoralRoT = self._get_moral_rot(batch)

        self.attn_results, self.attn = self.attn_layer(self.MoralRoT, self.MoralRoT, self.MoralRoT, normalize=True)
        self.weighted_value = self.attn_results.sum(dim=-2)

        self.logits = self.pre_classifier(torch.cat((self.weighted_value, self._curr_batch_situation), 1))
        self.pred = torch.nn.functional.softmax(self.classifier(self.logits), dim=-1)
        return self.pred, batch['label']

def clones(module, N):
    # Produce N identical layers.
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, dropout=None, do_normalize=False):
    # Compute 'Scaled Dot Product Attention'
    d_k = query.size(-1)
    
    if do_normalize:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    else:
        scores = torch.matmul(query, key.transpose(-2, -1))
        
    scores = scores.sum(dim=-1)
    p_attn = torch.nn.functional.softmax(scores, dim = -1) # -1
    p_attn = dropout(p_attn) if dropout is not None else p_attn
    p_attn = p_attn.unsqueeze(-1)
    return torch.mul(p_attn, value), p_attn

class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, h, d_model, device, dropout=0.1):
        # Take in model size and number of heads.
        super(MultiHeadedAttention, self).__init__()
        # We assume d_v always equals d_k
        self.d_model = d_model
        self.d_k = d_model // h
        self.h = h

        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout).to(device)

        self.linears = clones(torch.nn.Linear(d_model, d_model).to(device), 4) # self.linears = [query, key, value, output]
        
    def forward(self, query, key, value, normalize=False):
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, dropout=self.dropout, do_normalize=normalize)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x), self.attn

class MoralGroundModel(torch.nn.Module):
    def __init__(self, h, d_model, nbatches, all_authors, device,
                 n_classes=2, n_values=5, dropout=0.1, checkpoint="distilbert-base-uncased", MAX_LEN=32):
        super(MoralGroundModel, self).__init__()
        
        self.h = h
        self.d_model = d_model
        self.n_classes = n_classes
        self.nbatches = nbatches
        self.n_values = n_values
        self.device = device

        self.dropout = dropout
        self.MAX_LEN = MAX_LEN

        self.classifier = torch.nn.Linear(d_model*2, n_classes).to(device)
        self.MF_attn_layer = MultiHeadedAttention(h, d_model, device).to(device)

        self.all_authors = all_authors
        self.author_MF_attn_layers = clones(MultiHeadedAttention(h, d_model, device).to(device), len(all_authors))

        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.encode_model = AutoModel.from_pretrained(checkpoint).to(device)
    
    def _input_transform(self, batch):
        _situations = batch['situation']
        _rot_groups = [item for elem in batch['rot'] for item in elem.split('_____')]

        input_situ = self.tokenizer(_situations, None, add_special_tokens=True, 
                                    max_length=self.MAX_LEN, padding=True, truncation=True, return_tensors='pt')
        output_situ = self.encode_model(input_ids=input_situ.input_ids.to(self.device), 
                                    attention_mask=input_situ.attention_mask.to(self.device))
        hidden_situ = output_situ.last_hidden_state[:,0,:] # getting the last hidden state of the <s> token representation

        input_rot = self.tokenizer(_rot_groups, None, add_special_tokens=True, 
                                   max_length=self.MAX_LEN, padding=True, truncation=True, return_tensors='pt')
        output_rot = self.encode_model(input_ids=input_rot.input_ids.to(self.device), 
                                    attention_mask=input_rot.attention_mask.to(self.device))
        hidden_rot = output_rot.last_hidden_state[:,0,:]
        hidden_rot = torch.reshape(hidden_rot, (self.nbatches, self.n_values, self.d_model)) # (batch X n_values X d_model)
        
        return hidden_rot, hidden_situ
    
    def _get_redditor_moral_ground(self, batch):
        
        grounds = []
        for idx, _author in enumerate(batch['author']):
            _author_idx = self.all_authors.index(_author)
            _curr_moral_grounds = batch['moral-ground'][idx].split('_____') # should be length of 12

            input_grounds = self.tokenizer(_curr_moral_grounds, None, add_special_tokens=True, 
                                           max_length=self.MAX_LEN, padding=True, truncation=True, return_tensors='pt')
            output_grounds = self.encode_model(input_ids=input_grounds.input_ids.to(self.device), 
                                        attention_mask=input_grounds.attention_mask.to(self.device))
            hidden_grounds = output_grounds.last_hidden_state[:,0,:] # (n_moral_grounds_per_topic X hidden_dim)

            # Add the first 'batch' dimension for curr_situ and hidden_grounds
            _curr_situ = torch.reshape(self.situation[idx], (1, 1, self.d_model))
            hidden_grounds = torch.reshape(hidden_grounds, (1, hidden_grounds.shape[0], hidden_grounds.shape[1]))
            # _curr_attn_results, _curr_attn_weights = self.author_MF_attn_layers[_author_idx](hidden_grounds, _curr_situ, hidden_grounds)
            _curr_attn_results, _curr_attn_weights = self.author_MF_attn_layers[_author_idx](hidden_grounds, hidden_grounds, hidden_grounds)

            grounds.append(_curr_attn_results)

        return torch.cat(grounds) # output: [16 X 12 X 768] (nbatches X n_moral_grounds_per_topic X hidden_dim)

    def forward(self, batch, normalize=False):
        self.rot, self.situation = self._input_transform(batch)
        self.key_matrix = self._get_redditor_moral_ground(batch)
        self.attn_results, self.attn = self.MF_attn_layer(self.rot, self.key_matrix, self.rot, normalize=normalize)
        self.weighted_value = self.attn_results.sum(dim=-2)
        
        self.pred = torch.nn.functional.softmax(self.classifier(torch.cat((self.weighted_value, self.situation), 1)), dim=-1)
        return self.pred, batch['label']


class MoralGroundModel_singleAuthor(torch.nn.Module):
    def __init__(self, h, d_model, nbatches, device,
                 n_classes=2, n_values=5, dropout=0.3, checkpoint="distilbert-base-uncased", MAX_LEN=32):
        super(MoralGroundModel_singleAuthor, self).__init__()
        
        self.h = h
        self.d_model = d_model
        self.n_classes = n_classes
        self.nbatches = nbatches
        self.n_values = n_values
        self.device = device

        self.dropout = torch.nn.Dropout(p=dropout).to(device)
        self.MAX_LEN = MAX_LEN

        self.pre_classifier = torch.nn.Linear(d_model*2, d_model).to(device)
        self.classifier = torch.nn.Linear(d_model, n_classes).to(device)
        self.MF_attn_layer = MultiHeadedAttention(h, d_model, device).to(device)

        self.author_MF_attn_layers = MultiHeadedAttention(h, d_model, device).to(device)

        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.encode_model = AutoModel.from_pretrained(checkpoint).to(device)
    
    def _input_transform(self, batch):
        _situations = batch['situation']
        _rot_groups = [item for elem in batch['rot'] for item in elem.split('_____')]

        input_situ = self.tokenizer(_situations, None, add_special_tokens=True, 
                                    max_length=self.MAX_LEN, padding=True, truncation=True, return_tensors='pt')
        output_situ = self.encode_model(input_ids=input_situ.input_ids.to(self.device), 
                                    attention_mask=input_situ.attention_mask.to(self.device))
        hidden_situ = output_situ.last_hidden_state[:,0,:] # getting the last hidden state of the <s> token representation

        input_rot = self.tokenizer(_rot_groups, None, add_special_tokens=True, 
                                   max_length=self.MAX_LEN, padding=True, truncation=True, return_tensors='pt')
        output_rot = self.encode_model(input_ids=input_rot.input_ids.to(self.device), 
                                    attention_mask=input_rot.attention_mask.to(self.device))
        hidden_rot = output_rot.last_hidden_state[:,0,:]
        hidden_rot = torch.reshape(hidden_rot, (self.nbatches, self.n_values, self.d_model)) # (batch X n_values X d_model)
        
        return hidden_rot, hidden_situ
    
    def _get_redditor_moral_ground(self, batch):
        
        grounds = []
        for idx, _mg in enumerate(batch['moral-ground']):
            _curr_moral_grounds = _mg.split('_____') # should be length of 12

            input_grounds = self.tokenizer(_curr_moral_grounds, None, add_special_tokens=True, 
                                           max_length=self.MAX_LEN, padding=True, truncation=True, return_tensors='pt')
            output_grounds = self.encode_model(input_ids=input_grounds.input_ids.to(self.device), 
                                        attention_mask=input_grounds.attention_mask.to(self.device))
            hidden_grounds = output_grounds.last_hidden_state[:,0,:] # (n_moral_grounds_per_topic X hidden_dim)

            # Add the first 'batch' dimension for curr_situ and hidden_grounds
            _curr_situ = torch.reshape(self.situation[idx], (1, 1, self.d_model))
            hidden_grounds = torch.reshape(hidden_grounds, (1, hidden_grounds.shape[0], hidden_grounds.shape[1]))
            
            # _curr_attn_results, _curr_attn_weights = self.author_MF_attn_layers(hidden_grounds, _curr_situ, hidden_grounds)
            _curr_attn_results, _curr_attn_weights = self.author_MF_attn_layers(hidden_grounds, hidden_grounds, hidden_grounds)
            

            grounds.append(_curr_attn_results)

        return torch.cat(grounds) # output: [16 X 12 X 768] (nbatches X n_moral_grounds_per_topic X hidden_dim)

    def forward(self, batch, normalize=True):
        self.rot, self.situation = self._input_transform(batch)
        self.key_matrix = self._get_redditor_moral_ground(batch)
        self.attn_results, self.attn = self.MF_attn_layer(self.rot, self.key_matrix, self.rot, normalize=normalize)
        self.weighted_value = self.attn_results.sum(dim=-2)
        
        # pre_ = self.pre_classifier(torch.cat((self.weighted_value, self.situation), 1))
        # self.pred = torch.nn.functional.softmax(self.classifier(pre_), dim=-1)
        # return self.pred, batch['label']

        logits = self.classifier(self.dropout(nn.ReLU()(self.pre_classifier(torch.cat((self.weighted_value, self.situation), 1)))))
        return logits.view(-1, self.n_classes), batch['label']


class LayerNorm(torch.nn.Module):
    # "Construct a layernorm module (See citation for details)."
    def __init__(self, size_tuple, eps=1e-6, device='cuda'):
        super(LayerNorm, self).__init__()
        self.a_2 = torch.nn.Parameter(torch.ones(size_tuple)).to(device)
        self.b_2 = torch.nn.Parameter(torch.zeros(size_tuple)).to(device)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class MoralGroundTrainer(torch.nn.Module):
    def __init__(self, h, d_model, d_grounds, nbatches, MG_type, device='cuda', do_layernorm=False,
                 n_classes=2, n_values=5, dropout=0.1, checkpoint="distilbert-base-uncased", MAX_LEN=32):
        super(MoralGroundTrainer, self).__init__()
        
        self.h = h
        self.d_model = d_model
        self.d_grounds = d_grounds
        self.n_classes = n_classes
        self.nbatches = nbatches
        self.n_values = n_values
        self.device = device
        self.MG_type = MG_type
        self.do_layernorm = do_layernorm

        self.dropout = torch.nn.Dropout(p=dropout).to(device)
        self.MAX_LEN = MAX_LEN

        self.pre_classifier = torch.nn.Linear(d_model*2, d_model).to(device)
        self.classifier = torch.nn.Linear(d_model, n_classes).to(device)
        self.attn_layer = MultiHeadedAttention(h, d_model, device).to(device)
        
        if do_layernorm:
            self.mg_norm_layer = LayerNorm((nbatches, d_grounds, d_model))
            self.situ_norm_layer = LayerNorm((nbatches, 1, d_model))
            self.ff_norm_layer = LayerNorm((nbatches, d_model))

        if self.MG_type == 'random':
            self.random_MG = torch.nn.Parameter(torch.Tensor(d_grounds, d_model)).to(device)
            torch.nn.init.xavier_uniform_(self.random_MG, gain=torch.nn.init.calculate_gain('relu'))

        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.encode_model = AutoModel.from_pretrained(checkpoint).to(device)
    
    def _input_transform(self, batch):
        input_situ = self.tokenizer(batch['situation'], None, add_special_tokens=True, 
                                    max_length=self.MAX_LEN, padding=True, truncation=True, return_tensors='pt')
        output_situ = self.encode_model(input_ids=input_situ.input_ids.to(self.device), 
                                    attention_mask=input_situ.attention_mask.to(self.device))
        hidden_situ = output_situ.last_hidden_state[:,0,:] # getting the last hidden state of the <s> token representation
        # hidden_situ = torch.reshape(hidden_situ, (self.nbatches, 1, self.d_model))

        return hidden_situ
    
    def _get_redditor_moral_ground(self, batch):

        if self.MG_type == 'random':
            return self.random_MG.repeat(self.nbatches,1,1)

        elif self.MG_type == 'comment-based-cluster':
            grounds = []
            for _mg in batch['moral-ground-commCluster']:
                _curr_moral_grounds = _mg.split('_____') # should be length of d_grounds
                assert len(_curr_moral_grounds) == self.d_grounds

                input_grounds = self.tokenizer(_curr_moral_grounds, None, add_special_tokens=True, 
                                           max_length=self.MAX_LEN, padding=True, truncation=True, return_tensors='pt')
                output_grounds = self.encode_model(input_ids=input_grounds.input_ids.to(self.device), 
                                            attention_mask=input_grounds.attention_mask.to(self.device))
                hidden_grounds = output_grounds.last_hidden_state[:,0,:] # (n_moral_grounds_per_topic X hidden_dim)
                hidden_grounds = torch.reshape(hidden_grounds, (1, hidden_grounds.shape[0], hidden_grounds.shape[1]))
                
                grounds.append(hidden_grounds)
            return torch.cat(grounds) # should be [16 X 12 X 768] (nbatches X n_moral_grounds_per_topic X hidden_dim)
        
        elif self.MG_type == 'situation-based-cluster':
            grounds = []
            for _mg in batch['moral-ground']:
                _curr_moral_grounds = _mg.split('_____') # should be length of d_grounds
                assert len(_curr_moral_grounds) == self.d_grounds

                input_grounds = self.tokenizer(_curr_moral_grounds, None, add_special_tokens=True, 
                                           max_length=self.MAX_LEN, padding=True, truncation=True, return_tensors='pt')
                output_grounds = self.encode_model(input_ids=input_grounds.input_ids.to(self.device), 
                                            attention_mask=input_grounds.attention_mask.to(self.device))
                hidden_grounds = output_grounds.last_hidden_state[:,0,:] # (n_moral_grounds_per_topic X hidden_dim)
                hidden_grounds = torch.reshape(hidden_grounds, (1, hidden_grounds.shape[0], hidden_grounds.shape[1]))
                
                grounds.append(hidden_grounds)
            return torch.cat(grounds) # should be [16 X 12 X 768] (nbatches X n_moral_grounds_per_topic X hidden_dim)

    def forward(self, batch, self_attn=False, normalize=False):
        self.nbatches = len(batch['label'])
        self.situation = self._input_transform(batch)
        self.mg_matrix = self._get_redditor_moral_ground(batch)
        self.situation_addedDim = torch.reshape(self.situation, (self.nbatches, 1, self.d_model))
        if self_attn:
            self.attn_results, self.attn = self.attn_layer(self.mg_matrix, self.mg_matrix, self.mg_matrix, normalize=normalize)
            if self.do_layernorm:
                self.attn_results = self.mg_norm_layer(self.dropout(self.attn_results) + self.mg_matrix)
        else:
            self.attn_results, self.attn = self.attn_layer(self.mg_matrix, self.situation_addedDim, self.mg_matrix, normalize=normalize)
        self.weighted_value = self.attn_results.sum(dim=-2)

        #if self.do_layernorm:
        #    self.mg_norm_matrix = self.mg_norm_layer(self.mg_matrix)
        #    self.situ_norm = self.situ_norm_layer(self.situation_addedDim)
        #    if self_attn:
        #        self.norm_attn_results, _ = self.attn_layer(self.mg_norm_matrix, self.mg_norm_matrix, self.mg_norm_matrix, normalize=normalize)
        #    else:
        #        self.norm_attn_results, _ = self.attn_layer(self.mg_norm_matrix, self.situ_norm, self.mg_norm_matrix, normalize=normalize)
        #    self.attn_results += self.dropout(self.norm_attn_results)

        self.logits = self.pre_classifier(torch.cat((self.weighted_value, self.situation), 1))
        if self.do_layernorm:
            #self.norm_logits = self.pre_classifier(torch.cat((self.ff_norm_layer(self.weighted_value), self.ff_norm_layer(self.situation)), 1))
            self.logits = self.ff_norm_layer(self.weighted_value + self.dropout(self.logits))
        else:
            self.logits = self.dropout(torch.nn.ReLU()(self.logits))

        self.pred = torch.nn.functional.softmax(self.classifier(self.logits), dim=-1)
        return self.pred, batch['label']


class MoralGroundTrainerGlobal(torch.nn.Module):
    def __init__(self, h, d_model, d_grounds, nbatches, MG_type, device='cuda', do_layernorm=False,
                 n_classes=2, n_values=5, dropout=0.1, checkpoint="distilbert-base-uncased", MAX_LEN=32):
        super(MoralGroundTrainerGlobal, self).__init__()
        
        self.h = h
        self.d_model = d_model
        self.d_grounds = d_grounds
        self.n_classes = n_classes
        self.nbatches = nbatches
        self.n_values = n_values
        self.device = device
        self.MG_type = MG_type
        self.do_layernorm = do_layernorm

        self.dropout = torch.nn.Dropout(p=dropout).to(device)
        self.MAX_LEN = MAX_LEN

        self.pre_classifier = torch.nn.Linear(d_model*2, d_model).to(device)
        self.classifier = torch.nn.Linear(d_model, n_classes).to(device)

        self.attn_layer = clones(MultiHeadedAttention(h, d_model, device).to(device), 30)
        
        if do_layernorm:
            self.mg_norm_layer = LayerNorm((nbatches, d_grounds, d_model))
            self.situ_norm_layer = LayerNorm((nbatches, 1, d_model))
            self.ff_norm_layer = LayerNorm((nbatches, d_model))

        if self.MG_type == 'random':
            self.random_MG = torch.nn.Parameter(torch.Tensor(d_grounds, d_model)).to(device)
            torch.nn.init.xavier_uniform_(self.random_MG, gain=torch.nn.init.calculate_gain('relu'))

        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.encode_model = AutoModel.from_pretrained(checkpoint).to(device)
    
    def _input_transform(self, batch):
        input_situ = self.tokenizer(batch['situation'], None, add_special_tokens=True, 
                                    max_length=self.MAX_LEN, padding=True, truncation=True, return_tensors='pt')
        output_situ = self.encode_model(input_ids=input_situ.input_ids.to(self.device), 
                                    attention_mask=input_situ.attention_mask.to(self.device))
        hidden_situ = output_situ.last_hidden_state[:,0,:] # getting the last hidden state of the <s> token representation
        # hidden_situ = torch.reshape(hidden_situ, (self.nbatches, 1, self.d_model))

        return hidden_situ
    
    def _get_redditor_moral_ground(self, batch):

        if self.MG_type == 'random':
            return self.random_MG.repeat(self.nbatches,1,1)

        elif self.MG_type == 'comment-based-cluster':
            grounds = []
            for _mg in batch['moral-ground-commCluster']:
                _curr_moral_grounds = _mg.split('_____') # should be length of d_grounds
                assert len(_curr_moral_grounds) == self.d_grounds

                input_grounds = self.tokenizer(_curr_moral_grounds, None, add_special_tokens=True, 
                                           max_length=self.MAX_LEN, padding=True, truncation=True, return_tensors='pt')
                output_grounds = self.encode_model(input_ids=input_grounds.input_ids.to(self.device), 
                                            attention_mask=input_grounds.attention_mask.to(self.device))
                hidden_grounds = output_grounds.last_hidden_state[:,0,:] # (n_moral_grounds_per_topic X hidden_dim)
                hidden_grounds = torch.reshape(hidden_grounds, (1, hidden_grounds.shape[0], hidden_grounds.shape[1]))
                
                grounds.append(hidden_grounds)
            return torch.cat(grounds) # should be [16 X 12 X 768] (nbatches X n_moral_grounds_per_topic X hidden_dim)
        
        elif self.MG_type == 'situation-based-cluster':
            grounds = []
            for _mg in batch['moral-ground']:
                _curr_moral_grounds = _mg.split('_____') # should be length of d_grounds
                assert len(_curr_moral_grounds) == self.d_grounds

                input_grounds = self.tokenizer(_curr_moral_grounds, None, add_special_tokens=True, 
                                           max_length=self.MAX_LEN, padding=True, truncation=True, return_tensors='pt')
                output_grounds = self.encode_model(input_ids=input_grounds.input_ids.to(self.device), 
                                            attention_mask=input_grounds.attention_mask.to(self.device))
                hidden_grounds = output_grounds.last_hidden_state[:,0,:] # (n_moral_grounds_per_topic X hidden_dim)
                hidden_grounds = torch.reshape(hidden_grounds, (1, hidden_grounds.shape[0], hidden_grounds.shape[1]))
                
                grounds.append(hidden_grounds)
            return torch.cat(grounds) # should be (nbatches X n_moral_grounds_per_topic X hidden_dim)

    def forward(self, batch, self_attn=False, normalize=False):
        self.nbatches = len(batch['label'])
        self.situation = self._input_transform(batch)
        self.mg_matrix = self._get_redditor_moral_ground(batch)
        self.situation_addedDim = torch.reshape(self.situation, (self.nbatches, 1, self.d_model))
        if self_attn:
            self.attn_results, self.attn = self.attn_layer(self.mg_matrix, self.mg_matrix, self.mg_matrix, normalize=normalize)
            if self.do_layernorm:
                self.attn_results = self.mg_norm_layer(self.dropout(self.attn_results) + self.mg_matrix)
        else:
            self.attn_results, self.attn = self.attn_layer[int(batch['author-index'][0])](self.mg_matrix, self.situation_addedDim, self.mg_matrix, normalize=normalize)
        self.weighted_value = self.attn_results.sum(dim=-2)

        #if self.do_layernorm:
        #    self.mg_norm_matrix = self.mg_norm_layer(self.mg_matrix)
        #    self.situ_norm = self.situ_norm_layer(self.situation_addedDim)
        #    if self_attn:
        #        self.norm_attn_results, _ = self.attn_layer(self.mg_norm_matrix, self.mg_norm_matrix, self.mg_norm_matrix, normalize=normalize)
        #    else:
        #        self.norm_attn_results, _ = self.attn_layer(self.mg_norm_matrix, self.situ_norm, self.mg_norm_matrix, normalize=normalize)
        #    self.attn_results += self.dropout(self.norm_attn_results)

        self.logits = self.pre_classifier(torch.cat((self.weighted_value, self.situation), 1))
        if self.do_layernorm:
            #self.norm_logits = self.pre_classifier(torch.cat((self.ff_norm_layer(self.weighted_value), self.ff_norm_layer(self.situation)), 1))
            self.logits = self.ff_norm_layer(self.weighted_value + self.dropout(self.logits))
        else:
            self.logits = self.dropout(torch.nn.ReLU()(self.logits))

        self.pred = torch.nn.functional.softmax(self.classifier(self.logits), dim=-1)
        return self.pred, batch['label']



class MoralGroundTrainerBaseline(torch.nn.Module):
    def __init__(self, d_model, nbatches, device='cuda', 
                 n_classes=2, n_values=5, dropout=0.1, checkpoint="distilbert-base-uncased", MAX_LEN=32):
        super(MoralGroundTrainerBaseline, self).__init__()
        
        self.d_model = d_model
        self.n_classes = n_classes
        self.nbatches = nbatches
        self.device = device

        self.dropout = torch.nn.Dropout(p=dropout).to(device)
        self.MAX_LEN = MAX_LEN

        self.pre_classifier = torch.nn.Linear(d_model, d_model).to(device)
        self.classifier = torch.nn.Linear(d_model, n_classes).to(device)

        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.encode_model = AutoModel.from_pretrained(checkpoint).to(device)
    
    def _input_transform(self, batch):
        input_situ = self.tokenizer(batch['situation'], None, add_special_tokens=True, 
                                    max_length=self.MAX_LEN, padding=True, truncation=True, return_tensors='pt')
        output_situ = self.encode_model(input_ids=input_situ.input_ids.to(self.device), 
                                    attention_mask=input_situ.attention_mask.to(self.device))
        hidden_situ = output_situ.last_hidden_state[:,0,:] # getting the last hidden state of the <s> token representation

        return hidden_situ

    def forward(self, batch, normalize=True):
        self.situation = self._input_transform(batch)

        pre_ = self.pre_classifier(self.situation)
        self.pred = torch.nn.functional.softmax(self.classifier(pre_), dim=-1)
        return self.pred, batch['label']



class MoralRoTSelector(torch.nn.Module):
    def __init__(self, h, d_model, nbatches, MG_MODEL, device='cuda', do_layernorm = True, d_ground=6,
                 n_classes=2, n_values=5, dropout=0.1, checkpoint="distilbert-base-uncased", MAX_LEN=32):
        super(MoralRoTSelector, self).__init__()
        
        self.h = h
        self.d_ground = d_ground
        self.d_model = d_model
        self.n_classes = n_classes
        self.nbatches = nbatches
        self.n_values = n_values
        self.device = device
        self.MG_MODEL = MG_MODEL
        # MG_MODEL : given sitation, return weighted moral ground of this redditor. Concatenated with situation, predict judgment

        self.dropout = torch.nn.Dropout(p=dropout).to(device)
        self.MAX_LEN = MAX_LEN

        self.do_layernorm = do_layernorm
        if do_layernorm:
            self.rot_norm_layer = LayerNorm((nbatches, n_values, d_model))
            self.mg_norm_layer = LayerNorm((nbatches, d_ground, d_model))
            self.situ_norm_layer = LayerNorm((nbatches, 1, d_model))
            self.ff_norm_layer = LayerNorm((nbatches, d_model))

        self.pre_classifier = torch.nn.Linear(d_model*2, d_model).to(device)
        self.classifier = torch.nn.Linear(d_model, n_classes).to(device)
        self.attn_layer = MultiHeadedAttention(h, d_model, device).to(device)
    
    def _get_moral_rot(self, batch):
        _rot_groups = [item for elem in batch['rot'] for item in elem.split('_____')]

        input_rot = self.MG_MODEL.tokenizer(_rot_groups, None, add_special_tokens=True, 
                                   max_length=self.MAX_LEN, padding=True, truncation=True, return_tensors='pt')
        output_rot = self.MG_MODEL.encode_model(input_ids=input_rot.input_ids.to(self.device), 
                                    attention_mask=input_rot.attention_mask.to(self.device))
        hidden_rot = output_rot.last_hidden_state[:,0,:]
        hidden_rot = torch.reshape(hidden_rot, (self.nbatches, self.n_values, self.d_model)) # (batch X n_values X d_model)
        
        return hidden_rot

    def forward(self, batch, freeze_MG_attn=True, sum_MG=False, static_MG=True, self_attn=False, normalize=False, share_clf=False, wo_RoT=False):
        self.nbatches = len(batch['label'])
        if freeze_MG_attn:
            self.MG_MODEL.attn_layer.requires_grad = False
        _, _, = self.MG_MODEL.forward(batch)

        self._curr_batch_moral_ground = self.MG_MODEL.weighted_value if sum_MG else self.MG_MODEL.attn_results
        if static_MG:
            #self._curr_batch_moral_ground = self.MG_MODEL.mg_matrix
            self._curr_batch_moral_ground = torch.div(self.MG_MODEL.mg_matrix, self.d_ground)
        self._curr_batch_situation = self.MG_MODEL.situation

        self.MoralRoT = self._get_moral_rot(batch)

        self.attn_results, self.attn = self.attn_layer(self.MoralRoT, self._curr_batch_moral_ground, self.MoralRoT, normalize=normalize)
        if wo_RoT:
            self.attn_results, self.attn = self.attn_layer(self._curr_batch_moral_ground, self.MG_MODEL.situation_addedDim, self._curr_batch_moral_ground, normalize=normalize)
        if self_attn:
            self.attn_results, self.attn = self.attn_layer(self.MoralRoT, self.MoralRoT, self.MoralRoT, normalize=normalize)
        if self.do_layernorm:
            self.attn_results = self.rot_norm_layer(self.dropout(self.attn_results)+self.MoralRoT)
        self.weighted_value = self.attn_results.sum(dim=-2)

        self.logits = self.MG_MODEL.pre_classifier(torch.cat((self.weighted_value, self._curr_batch_situation), 1)) if share_clf else self.pre_classifier(torch.cat((self.weighted_value, self._curr_batch_situation), 1))
        self.logits = self.ff_norm_layer(self.weighted_value + self.dropout(self.logits)) if self.do_layernorm else self.dropout(torch.nn.ReLU()(self.logits))
        self.pred = torch.nn.functional.softmax(self.MG_MODEL.classifier(self.logits), dim=-1) if share_clf else torch.nn.functional.softmax(self.classifier(self.logits), dim=-1)
        return self.pred, batch['label']


class MoralRoTSelectorGlobal(torch.nn.Module):
    def __init__(self, h, d_model, nbatches, MG_MODEL, device='cuda', do_layernorm = False, d_ground=6,
                 n_classes=2, n_values=5, dropout=0.1, checkpoint="distilbert-base-uncased", MAX_LEN=32):
        super(MoralRoTSelectorGlobal, self).__init__()
        
        self.h = h
        self.d_ground = d_ground
        self.d_model = d_model
        self.n_classes = n_classes
        self.nbatches = nbatches
        self.n_values = n_values
        self.device = device
        self.MG_MODEL = MG_MODEL
        # MG_MODEL : given sitation, return weighted moral ground of this redditor. Concatenated with situation, predict judgment

        self.dropout = torch.nn.Dropout(p=dropout).to(device)
        self.MAX_LEN = MAX_LEN

        self.do_layernorm = do_layernorm
        if do_layernorm:
            self.rot_norm_layer = LayerNorm((nbatches, n_values, d_model))
            self.mg_norm_layer = LayerNorm((nbatches, d_ground, d_model))
            self.situ_norm_layer = LayerNorm((nbatches, 1, d_model))
            self.ff_norm_layer = LayerNorm((nbatches, d_model))

        self.pre_classifier = torch.nn.Linear(d_model*2, d_model).to(device)
        self.classifier = torch.nn.Linear(d_model, n_classes).to(device)
        # self.attn_layer = MultiHeadedAttention(h, d_model, device).to(device)
        self.attn_layer = clones(MultiHeadedAttention(h, d_model, device).to(device), 30)
    
    def _get_moral_rot(self, batch):
        _rot_groups = [item for elem in batch['rot'] for item in elem.split('_____')]

        input_rot = self.MG_MODEL.tokenizer(_rot_groups, None, add_special_tokens=True, 
                                   max_length=self.MAX_LEN, padding=True, truncation=True, return_tensors='pt')
        output_rot = self.MG_MODEL.encode_model(input_ids=input_rot.input_ids.to(self.device), 
                                    attention_mask=input_rot.attention_mask.to(self.device))
        hidden_rot = output_rot.last_hidden_state[:,0,:]
        hidden_rot = torch.reshape(hidden_rot, (self.nbatches, self.n_values, self.d_model)) # (batch X n_values X d_model)
        
        return hidden_rot

    def forward(self, batch, freeze_MG_attn=True, sum_MG=False, static_MG=False, self_attn=False, normalize=False, share_clf=False, wo_RoT=False):
        self.nbatches = len(batch['label'])
        if freeze_MG_attn:
            self.MG_MODEL.attn_layer.requires_grad = False
        _, _, = self.MG_MODEL.forward(batch)

        self._curr_batch_moral_ground = self.MG_MODEL.weighted_value if sum_MG else self.MG_MODEL.attn_results
        if static_MG:
            #self._curr_batch_moral_ground = self.MG_MODEL.mg_matrix
            self._curr_batch_moral_ground = torch.div(self.MG_MODEL.mg_matrix, self.d_ground)
        self._curr_batch_situation = self.MG_MODEL.situation

        self.MoralRoT = self._get_moral_rot(batch)

        self.attn_results, self.attn = self.attn_layer[int(batch['author-index'][0])](self.MoralRoT, self._curr_batch_moral_ground, self.MoralRoT, normalize=normalize)
        if wo_RoT:
            self.attn_results, self.attn = self.attn_layer[int(batch['author-index'][0])](self._curr_batch_moral_ground, self.MG_MODEL.situation_addedDim, self._curr_batch_moral_ground, normalize=normalize)
        if self_attn:
            self.attn_results, self.attn = self.attn_layer[int(batch['author-index'][0])](self.MoralRoT, self.MoralRoT, self.MoralRoT, normalize=normalize)
        if self.do_layernorm:
            self.attn_results = self.rot_norm_layer(self.dropout(self.attn_results)+self.MoralRoT)
        self.weighted_value = self.attn_results.sum(dim=-2)

        self.logits = self.MG_MODEL.pre_classifier(torch.cat((self.weighted_value, self._curr_batch_situation), 1)) if share_clf else self.pre_classifier(torch.cat((self.weighted_value, self._curr_batch_situation), 1))
        self.logits = self.ff_norm_layer(self.weighted_value + self.dropout(self.logits)) if self.do_layernorm else self.dropout(torch.nn.ReLU()(self.logits))
        self.pred = torch.nn.functional.softmax(self.MG_MODEL.classifier(self.logits), dim=-1) if share_clf else torch.nn.functional.softmax(self.classifier(self.logits), dim=-1)
        return self.pred, batch['label']


class JudgmentClassifier(torch.nn.Module):
    def __init__(self, d_model, nbatches, device='cuda', n_classes=2, dropout=0.1, checkpoint="distilbert-base-uncased", MAX_LEN=32):
        super(JudgmentClassifier, self).__init__()
        
        self.d_model = d_model
        self.n_classes = n_classes
        self.nbatches = nbatches
        self.device = device

        if dropout is not None:
            self.dropout = torch.nn.Dropout(p=dropout).to(device)
        else:
            self.dropout = None
        self.MAX_LEN = MAX_LEN

        self.pre_classifier = torch.nn.Linear(d_model*2, d_model).to(device) # getting rot and situation
        self.classifier = torch.nn.Linear(d_model, n_classes).to(device) #classify

        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.encode_model = AutoModel.from_pretrained(checkpoint).to(device)
    
    def _input_transform(self, batch):
        input_situ = self.tokenizer(batch['situation'], None, add_special_tokens=True, 
                                    max_length=self.MAX_LEN, padding=True, truncation=True, return_tensors='pt')
        output_situ = self.encode_model(input_ids=input_situ.input_ids.to(self.device), 
                                    attention_mask=input_situ.attention_mask.to(self.device))
        hidden_situ = output_situ.last_hidden_state[:,0,:] # getting the last hidden state of the <s> token representation

        input_rot = self.tokenizer(batch['rot'], None, add_special_tokens=True, 
                                    max_length=self.MAX_LEN, padding=True, truncation=True, return_tensors='pt')
        output_rot = self.encode_model(input_ids=input_situ.input_ids.to(self.device), 
                                    attention_mask=input_situ.attention_mask.to(self.device))
        hidden_rot = output_situ.last_hidden_state[:,0,:] # getting the last hidden state of the <s> token representation

        return hidden_situ, hidden_rot

    def forward(self, batch):
        self.situation, self.rot = self._input_transform(batch)
        
        self.logits = self.pre_classifier(torch.cat((self.situation, self.rot), 1))
        
        self.pred = torch.nn.functional.softmax(self.classifier(self.logits), dim=-1)
        return self.pred, batch['label']

        # self.pred = self.classifier(self.dropout(nn.ReLU()(self.logits)))
        # return self.pred(-1, self.n_classes), batch['label']



class RedditorSituationJudgment(torch.nn.Module):
    def __init__(self, d_model, nbatches, 
            device='cuda', n_classes=2, dropout=0.1, checkpoint="distilbert-base-uncased", MAX_LEN=32):
        super(RedditorSituationJudgment, self).__init__()
        
        self.d_model = d_model
        self.n_classes = n_classes
        self.nbatches = nbatches
        self.device = device

        if dropout is not None:
            self.dropout = torch.nn.Dropout(p=dropout).to(device)
        else:
            self.dropout = None
        
        self.MAX_LEN = MAX_LEN

        self.pre_classifier = torch.nn.Linear(d_model*2, d_model).to(device) # getting rot and situation
        self.classifier = torch.nn.Linear(d_model, n_classes).to(device) #classify

        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.encode_model = AutoModel.from_pretrained(checkpoint).to(device)
    
    def _input_transform(self, batch):
        input_situ = self.tokenizer(batch['situation'], None, add_special_tokens=True, 
                                    max_length=self.MAX_LEN, padding=True, truncation=True, return_tensors='pt')
        output_situ = self.encode_model(input_ids=input_situ.input_ids.to(self.device), 
                                    attention_mask=input_situ.attention_mask.to(self.device))
        hidden_situ = output_situ.last_hidden_state[:,0,:] # getting the last hidden state of the <s> token representation

        return hidden_situ

    def forward(self, batch):
        self.situation, self.rot = self._input_transform(batch)
        
        self.logits = self.pre_classifier(torch.cat((self.situation, self.rot), 1))
        
        self.pred = torch.nn.functional.softmax(self.classifier(self.logits), dim=-1)
        return self.pred, batch['label']


class NoamOpt:
    # "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        # "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        # "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        

class SimpleLossCompute:
    def __init__(self, criterion, opt=None):
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, pred, tgt):
        loss = self.criterion(pred, tgt)
        loss.backward()
        if self.opt is not None and isinstance(self.opt, torch.optim.Adam):
            self.opt.step()
            self.opt.zero_grad()
        elif self.opt is not None and isinstance(self.opt, NoamOpt):
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item()
