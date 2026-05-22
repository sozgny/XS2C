import torch
import numpy as np
import pandas as pd
import math
import time
import json
import base64
from io import BytesIO
import argparse
import re
import glob
import sys
import os 
import random
import networkx as nx
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor 
from datetime import datetime
from pytz import utc, timezone
from numpy import percentile
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from collections import defaultdict
from scipy.stats import rankdata, iqr, trim_mean
from sklearn.metrics import f1_score, mean_squared_error, classification_report, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score, precision_recall_curve, auc, average_precision_score, matthews_corrcoef, confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.nn.inits import glorot, zeros



###### UTILES ######

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_device(dev):
    global _device
    _device = dev

def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def get_var_list(list_dir): 
    var_file = open(list_dir, 'r')
    var_list = []
    for ft in var_file:
        var_list.append(ft.strip())

    return var_list


def construct_data(data, feature_map, labels=0):
    res = []
    for feature in feature_map:
        if feature in data.columns:
            res.append(data.loc[:, feature].values.tolist())
        else:
            print(feature, 'not exist in data')
            
    sample_n = len(res[0])
    if type(labels) == int:
        res.append([labels]*sample_n)
    elif len(labels) == sample_n:
        res.append(labels)

    return res


class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, mode='train', config = None):
        
        self.raw_data   = raw_data
        self.config     = config
        self.edge_index = edge_index
        self.mode       = mode

        x_data = raw_data[:-1]
        labels = raw_data[-1] 
        data   = x_data

        data   = torch.tensor(data).double()
        labels = torch.tensor(labels).double()

        self.x, self.y, self.labels = self.process(data, labels)
    
    def __len__(self):
        return len(self.x)

    def process(self, data, labels):
        x_arr, y_arr = [], []
        labels_arr = []

        slide_win, slide_stride = [self.config[k] for k in ['slide_win', 'slide_stride']]
        is_train = self.mode == 'train'

        node_num, total_time_len = data.shape

        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)
        
        for i in rang:
            ft  = data[:, i-slide_win:i]
            tar = data[:, i]

            x_arr.append(ft)
            y_arr.append(tar)
            labels_arr.append(labels[i])

        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()
        labels = torch.Tensor(labels_arr).contiguous()
        
        return x, y, labels
    
    def __getitem__(self, idx):
        feature = self.x[idx].double()
        y = self.y[idx].double()
        edge_index = self.edge_index.long()
        label = self.labels[idx].double()

        return feature, y, label, edge_index


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()

def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss

def load_graph(graph_dir):
    return torch.load(graph_dir).long().contiguous()

# For finetuning
def copy_and_freeze_selected(normal_model, fault_model, freeze_keys=[]):
    state_dict = normal_model.state_dict()
    
    for name, param in fault_model.named_parameters():
        if any(key in name for key in freeze_keys):
            param.requires_grad = False
        else:
            param.data.copy_(state_dict[name])
            param.requires_grad = True


def make_optimizer_only_trainable(model, config, lr=None):
    if lr is None:
        lr=0.001
    weight_decay = config.get('decay', 0.0)
    trainable= [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(trainable, lr=lr, weight_decay=weight_decay)

###### MODULES ######

class GraphLayer(MessagePassing): #Définir les couches de graphes en type "MessagePassing"
    def __init__(self, in_channels, out_channels, heads=1, concat=True,negative_slope=0.2, dropout=0, bias=True, inter_dim=-1,**kwargs):
        super(GraphLayer, self).__init__(aggr='add', **kwargs)  # Agrégation par la somme des features à la fin.

        self.override_alpha = None          # Tensor
        self.override_mask_single = None    # Bool tensor (E_single,)
        self.override_batch_num = None      # int
        self.override_rows = None   # Bool tensor (B,)



        self.in_channels = in_channels          # Dimension des features input
        self.out_channels = out_channels        # Dimension des features output
        self.heads = heads                      # Nombre des heads dans "multi-head attention"
        # #heads independent mécanismes d'attention réalisent la transformation et puis leurs features sont concaténés.
        
        self.concat = concat                    # Pour spécifier si les sorties des têtes sont concaténées ou non. 
        self.negative_slope = negative_slope    # LeakyReLU angle of the negative slope. (default: 0.2) LeakyRelu(x)= negative_slope*x si x est négatif.
        
        self.dropout = dropout
        # dropout = Probabilité d'abandon des coefficients d'attention normalisés qui expose chaque nœud à un voisinage échantillonné 
        ## de manière stochastique pendant l'apprentissage
        
        self.__alpha__ = None
        self.alpha_coefs = None
        
        self._logit_override = None
        self.last_logits = None
        
        self.lin = Linear(in_channels, heads * out_channels, bias=False) 
        
        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))     # coef. d'attention à apprendre pour noeud i
        # Explication: représente les paramètres d'attention utilisés pour calculer les poids d'attention entre les nœuds i et leurs voisins j dans le graphe.
                    ## Ces paramètres sont appris pendant l'entraînement du modèle et sont spécifiques à la relation entre le nœud i et ses voisins dans le contexte de l'attention.
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))     # coef. d'attention à apprendre pour noeud j
        
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))  # coef. d'attention à apprendre pour sensor embedding du noeud i
        # Explication: utilisés conjointement avec les embeddings (plongements) des nœuds lors du calcul des poids d'attention.
                    ## Ces paramètres permettent au modèle d'accorder une attention différenciée en fonction des caractéristiques d'embedding spécifiques des nœuds
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))  # coef. d'attention à apprendre pour sensor embedding du noeud j

        if bias and concat:             # Si multi-head attention avec concatenation
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:       # Sans concatenation (somme)
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self): # Initialisation des poids des paramètres 
        glorot(self.lin.weight)  # matrice de poids 
        glorot(self.att_i)       # matrice de poids
        glorot(self.att_j)       # matrice de poids
        
        zeros(self.att_em_i)
        zeros(self.att_em_j)

        zeros(self.bias)

    def set_alpha_override(self, alpha_fault, mask_single, batch_num, rows_mask=None):
        self.override_alpha = alpha_fault
        self.override_mask_single = mask_single
        self.override_batch_num = batch_num
        self.override_rows = rows_mask  # (B,) veya None


    def clear_alpha_override(self):
        self.override_alpha = None
        self.override_mask_single = None
        self.override_batch_num = None
        self.override_rows = None

    def set_logit_override(self, logit_fault, mask_single, batch_num, rows_mask):
        """
        logit_fault: Tensor (B*E_single, heads)  or (B*E_single, heads, 1) -> aşağıda squeeze ederiz
        mask_single: BoolTensor (E_single,)
        batch_num: int (B)
        rows_mask: BoolTensor (B,)  # hangi grafiklerde (rows) intervention var
        """
        self._logit_override = {
            "logit_fault": logit_fault,
            "mask_single": mask_single,
            "batch_num": batch_num,
            "rows_mask": rows_mask
        }

    def clear_logit_override(self):
        self._logit_override = None

    def forward(self, x, edge_index, embedding, return_attention_weights=True):
        """"""
        # x                        = caractéristiques des noeuds
        # edge_index               = liste des paires d'indices verticales qui représentent les aretes dans le graphe
        # embedding                = plongements des noeuds (s'il est utilisé)
        # return_attention_weights = pour spécifier si les poids d'attention doivent être retournés.
        
        
        # Appliquer la transformation linéaire aux caractéristiques d'entrée:
        if torch.is_tensor(x):
            x = self.lin(x) 
            x = (x, x)  
        else:
            x = (self.lin(x[0]), self.lin(x[1]))

        
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,num_nodes=x[1].size(self.node_dim)) # pour trouver l'attention du noeud lui-meme  

        # Appel initial (executer la propagation ) permettant de démarrer la propagation du message sur les aretes. Nécessite les indices des aretes.
        out = self.propagate(edge_index, x=x, embedding=embedding, edges=edge_index,return_attention_weights=return_attention_weights)
        
        # Agrégation des sorties des têtes d'attention et applique éventuellement un biais (bias) avant de renvoyer le résultat.           
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)     

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i, x_j, edge_index_i, size_i, embedding, edges, return_attention_weights):
        # Construit les messages par les noeuds voisins (j) au noeud i lors de la propagation. Récupère les arguments de `propagate()`
        # x_i : Caractéristiques des nœuds i (dimension : [batch_size * num_edges, heads, out_channels])
        ##      x_i est une représentation tensorielle des caractéristiques du nœud i étendue sur le lot (batch) de données, 
        # ce qui permet au GNN de traiter efficacement plusieurs instances de graphes en parallèle lors de l'apprentissage.
        # x_j : Caractéristiques des nœuds j (dimension : [batch_size * num_edges, heads, out_channels])
        # edge_index_i : Indices des arêtes des nœuds i (dimension : [2, num_edges])
        # size_i : Nombre de nœuds dans le graphe (dimension : scalar)
        # embedding : Plongements (embeddings) des nœuds (dimension : [num_nodes, embedding_dim])
        # edges : Indices des arêtes du graphe (dimension : [2, num_edges])
        # return_attention_weights : Indicateur pour retourner les poids d'attention (boolean)
        
        """
        print("edge_index_i:", edge_index_i)
        print("size_i:", size_i)
        print("x_i old shape:", x_i.shape)
        print("x_j old shape:", x_j.shape)
        print("edges:", edges)
        print("edges shape:", edges.shape)
        
        
        print("att_i:", self.att_i)
        print("att_j:", self.att_j)
        print("att_em_i:", self.att_em_i)
        print("att_em_j:", self.att_em_j)
        """
        # Réorganiser les caractéristiques des nœuds pour l'opération matricielle:
        x_i = x_i.view(-1, self.heads, self.out_channels)  # input feature du noeud i 
        x_j = x_j.view(-1, self.heads, self.out_channels)  # input features des noeuds j

        # Concaténer les caractéristiques des nœuds avec les embeddings si disponibles:
        if embedding is not None:
            embedding_i = embedding[edge_index_i]  # Plongements des noeuds i
            embedding_j = embedding[edges[0]]      # Plongements des noeuds j (source des aretes)
            
            # Répéter les plongements pour correspondre aux dimensions des messages
            embedding_i = embedding_i.unsqueeze(1).repeat(1,self.heads,1)  # [batch_size * num_edges, heads, embedding_dim]
            embedding_j = embedding_j.unsqueeze(1).repeat(1,self.heads,1)  # [batch_size * num_edges, heads, embedding_dim]

            # Equation 6 (g_i et g_j) Concaténer les caractéristiques avec les plongements:
            key_i = torch.cat((x_i, embedding_i), dim=-1) # g_i => [batch_size * num_edges, heads, out_channels + embedding_dim]
            key_j = torch.cat((x_j, embedding_j), dim=-1) # g_j => [batch_size * num_edges, heads, out_channels + embedding_dim]


        # Concaténer les coefficients d'attention avec les plongements correspondants
        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)  # [1, heads, out_channels + embedding_dim]
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)  # [1, heads, out_channels + embedding_dim]

        # Equation 7 Calculer l'attention pondérée entre les paires de nœuds:
        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)  # [batch_size * num_edges, heads]
        
        
        alpha = alpha.view(-1, self.heads, 1)            # [batch_size * num_edges, heads, 1]
        e = F.leaky_relu(alpha, self.negative_slope).squeeze(-1) # [batch_size * num_edges, heads]
        
                # DEBUG / cache: softmax öncesi logits'i sakla

        # 2) LOGIT OVERRIDE (softmax'tan önce!)
        if self._logit_override is not None:
            logit_fault = self._logit_override["logit_fault"]
            mask_single = self._logit_override["mask_single"]
            B = self._logit_override["batch_num"]
            rows_mask = self._logit_override["rows_mask"]

            # logit_fault shape standardize
            if logit_fault.dim() == 3 and logit_fault.size(-1) == 1:
                logit_fault = logit_fault.squeeze(-1)  # (B*E_single, heads)

            # --- batched-edge mask üret ---
            # E_total = B * E_single
            # mask_single: (E_single,) -> repeat -> (B*E_single,)
            mask_batched = mask_single.repeat(B).to(e.device)

            # --- sadece fault satırları (grafikler) için uygula ---
            # Her edge’in hangi grafiğe ait olduğunu bulmamız lazım.
            # Batched graph'ta node id'ler b*node_num offset'li.
            # edge_index_i hedef node id'leri -> graph_id = edge_index_i // node_num.
            # node_num’ı size_i’dan yakalayabiliriz: size_i == B*node_num
            node_num = size_i // B
            graph_id = (edge_index_i // node_num).long()  # (E_total,)

            row_mask_per_edge = rows_mask[graph_id].to(e.device)  # (E_total,)

            final_mask = mask_batched & row_mask_per_edge  # (E_total,)

            # override
            # (safety) boyut uyuşmazlığı varsa assert ile yakala
            if logit_fault.shape != e.shape:
                raise RuntimeError(f"logit_fault shape {logit_fault.shape} != logits shape {e.shape}")

            #e[final_mask] = logit_fault[final_mask]
            ## CUTTING EDGES 
            e[final_mask] = -50.0   # logits very negative => alpha ~ 0 after softmax
        self.last_logits = e.detach()

        self.node_dim=0
        
        # Equation 8: 
        alpha = softmax(e, edge_index_i, num_nodes=size_i) # [batch_size * num_edges, heads, 1]
        if (self.override_alpha is not None) and (self.override_mask_single is not None):
            B = int(self.override_batch_num)
            E_total = alpha.size(0)
            E_single = E_total // B

            # alpha: (B*E, heads, 1)  --> reshape
            alpha_ = alpha.view(B, E_single, self.heads, 1)
            aF_    = self.override_alpha.view(B, E_single, self.heads, 1)

            m = self.override_mask_single  # (E_single,) bool
            rows = self.override_rows      # (B,) bool veya None

            if rows is None:
                mask2 = m.unsqueeze(0).expand(B, E_single)          # (B,E)
            else:
                mask2 = rows.unsqueeze(1) & m.unsqueeze(0)          # (B,E)

            mask4 = mask2.unsqueeze(-1).unsqueeze(-1)               # (B,E,1,1)

            alpha_ = torch.where(mask4, aF_, alpha_)                # override in-place effect

            alpha = alpha_.view(B * E_single, self.heads, 1)



        
        
        
        
        # Enregistrer les poids d'attention si nécessaire pour le retour
        if return_attention_weights:
            self.__alpha__ = alpha # Stocker les coefficients d'attention

        # Appliquer le dropout aux poids d'attention
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)  # [batch_size * num_edges, heads, 1]
        '''print("coefs alpha taille:", alpha.shape)
        print("edge index i taile:", edge_index_i.shape)
        print("edges taile:", edges.shape)'''
        
        self.alpha_coefs = alpha
        # Multiplication des caractéristiques des nœuds j par les poids d'attention
        return x_j * alpha.view(-1, self.heads, 1)  # Equation 5 sans RELU 
        # [batch_size * num_edges, heads, out_channels]

        
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num = 512):
        super(OutLayer, self).__init__()
        modules = []

        for i in range(layer_num):
            if i == layer_num-1:
                modules.append(nn.Linear( in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear( layer_in_num, inter_num ))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0,2,1)
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)

        return out



class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()


        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):

        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index
        self.alpha_coefs = self.gnn.alpha_coefs
        out = self.bn(out)
        
        return self.relu(out)


###### MODELS ##############

class normalModel(nn.Module):
    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, topk=20): #topk=20 defaut

        super(normalModel, self).__init__()

        self.edge_index_sets = edge_index_sets # les indices des aretes
        device = get_device()
        #print("edge index sets:", self.edge_index_sets)
        edge_index = edge_index_sets[0]
        #print("edge index :", edge_index)

        # Couche embedding
        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)

        # Construire les couches de GNN
        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim+embed_dim, heads=1) for i in range(edge_set_num)
        ])

        # Couche de sortie
        self.out_layer = OutLayer(dim*edge_set_num, node_num, out_layer_num, inter_num = out_layer_inter_dim)
        
        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None
        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None
        self.dp = nn.Dropout(0.2)
        
        self.learned_graphs = {}

        # Initialisation des poids
        self.init_params()
    
    def init_params(self): # Initialisation des poids de la couche embedding
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))


    def forward(self, data, org_edge_index):
        # "data" est un tenseur en format  (batch_num / node_num / all_feature)
        
        x = data.clone().detach()  # les données à utiliser pour la suite
        edge_index_sets = self.edge_index_sets

        device = data.device

        batch_num, node_num, all_feature = x.shape
        #print("batch num:",batch_num)
        #print("data shape:", x.shape)
        x = x.view(-1, all_feature).contiguous()
        # batch_num   = nombre des individus ? 
        # node_num    = nombre de noeuds
        # all_feature = nombre des features de chaque noeud
        # Remarque: dans le contexte des GNNs, le concept de "batch" est utilisé pour regrouper les nœuds et leurs voisins 
        ###         lors de l'exécution des opérations sur des données de graphe. 
        ###   Chaque batch représente un sous-ensemble de nœuds et de connexions du graphe, 
        ###   ce qui permet au modèle de capturer efficacement les informations locales et de traiter les structures de graphe de manière parallèle et régulière.
        
        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            #print("i:",i)
            #print("edge_index:",edge_index)
            edge_num = edge_index.shape[1]  # edge_index = pour déterminer les voisins d'un noeud (i)
            cache_edge_index = self.cache_edge_index_sets[i]   # Contrôle d'indices des aretes déjà calculés ou cachés
            # Pour faire compatible les edge_index avec batch_num et et node_num
            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num*batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)
            
            batch_edge_index = self.cache_edge_index_sets[i]
            
            # Tous les embeddings des noeuds
            all_embeddings = self.embedding(torch.arange(node_num).to(device))
            weights_arr = all_embeddings.detach().clone()
            all_embeddings = all_embeddings.repeat(batch_num, 1)

            
            gated_edge_index = edge_index
            
            
            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device) 
            #print('batch_gated_edge_index:', batch_gated_edge_index.shape)
            #print('batch_edge_index:', batch_edge_index.shape)
            
            self.current_gated_edge_index = gated_edge_index

            # Appel couche GNN => pour obtenir nouvelles vecteurs de features des noeuds en considérant les relations entre eux.
            gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings) # vecteurs de features 
            gcn_outs.append(gcn_out)

        #print(self.learned_graphs)
        # Chaque GNN transforme les caractéristiques des nœuds en tenant compte des relations et des structures spécifiques représentées par son ensemble d'index d'arete. 
        # Ainsi, chaque GNN produit une sortie qui capture des informations importantes sur les nœuds en fonction de sa vue particulière du graphe.
        
        # Concaténer les sorties des couches GNN:
        ## Représentation combinée des nœuds qui intègre les informations extraites par tous les GNNs appliqués aux différents ensembles d'index d'arete.
        x = torch.cat(gcn_outs, dim=1) 
        x = x.view(batch_num, node_num, -1)

        # Multiplier chaque noeud avec son embedding EQUATION 9
        indexes = torch.arange(0,node_num).to(device)
        #print("dimension x:",x.shape)
        #print("dimension embedding:",self.embedding)
        
        out = torch.mul(x, self.embedding(indexes))
        #print("dimension out:",out.shape)
        # Arranger la sortie & appliquer Dropout et OutLayer:
        out = out.permute(0,2,1) # Échanger les dimensions 1 et 2 
        ## Ex: si out avait la forme (batch_size, nombre de nœuds, nombre de caractéristiques), 
        ###    après cette opération, il aura la forme (batch_size, nombre de caractéristiques, nombre de nœuds).
        
        out = F.relu(self.bn_outlayer_in(out)) # Batch normalisation
        out = out.permute(0,2,1)               # Après normalisation les dimensions de out sont à nouveau permutées pour revenir à sa forme d'origine (batch_size, nombre de nœuds, nombre de caractéristiques).
        out = self.dp(out)                     # désactive aléatoirement certains neurones de out pendant l'entraînement pour éviter le surapprentissage. 
        out = self.out_layer(out)              # La couche de sortie finale
        out = out.view(-1, node_num)           # chaque ligne correspond à une entrée (exemple) dans le lot, et chaque colonne correspond à une valeur de sortie pour un nœud particulier.
   

        return out
    


class faultModel(nn.Module):
    def __init__(self, edge_index_sets, node_num, embeddings, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1):

        super(faultModel, self).__init__()

        self.edge_index_sets = edge_index_sets 
        device = get_device()
        edge_index = edge_index_sets[0]
        embed_dim = dim

        # Freeze embeddings, use initializations from normalModel
        weight = embeddings.weight.detach().clone()
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=True).to(device)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)

        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim+embed_dim, heads=1) for i in range(edge_set_num)
        ])
        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.out_layer = OutLayer(dim*edge_set_num, node_num, out_layer_num, inter_num = out_layer_inter_dim)
        
        self.node_embedding = None
        self.dp = nn.Dropout(0.2)


    def forward(self, data, org_edge_index):
    
        x = data.clone().detach()
        edge_index_sets = self.edge_index_sets
        device = data.device
        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()
        gcn_outs = []
        
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]  
            cache_edge_index = self.cache_edge_index_sets[i]   

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num*batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)
            
            batch_edge_index = self.cache_edge_index_sets[i]
            node_ids = torch.arange(node_num, device=self.embedding.weight.device)
            all_embeddings = self.embedding(node_ids)
            weights_arr = all_embeddings.detach().clone()
            all_embeddings = all_embeddings.repeat(batch_num, 1)
            
            gated_edge_index = edge_index
            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device) 
            self.current_gated_edge_index = gated_edge_index 
            
            gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num*batch_num, embedding=all_embeddings) 
            gcn_outs.append(gcn_out)

        
        x = torch.cat(gcn_outs, dim=1) 
        x = x.view(batch_num, node_num, -1)
        indexes = torch.arange(0,node_num).to(device)

        out = torch.mul(x, self.embedding(indexes))
        out = out.permute(0,2,1) 
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0,2,1)               
        out = self.dp(out)                      
        out = self.out_layer(out)              
        out = out.view(-1, node_num)           

        return out
    

#Training
def train(model=None, save_path='', config={}, train_dataloader=None, val_dataloader=None, train_dataset=None):
    seed = config['seed']
    lr = config.get('lr', 0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config['decay'])

    train_loss_list = []
    val_loss_list = []
    epoch_train_loss_list = []

    device = get_device()

    best_val_loss = float("inf")
    best_train_loss = float("inf")

    epoch = config['epoch']
    early_stop_win = config.get('early_stop_win', 15)
    stop_improve_count = 0

    model.train()

    for i_epoch in range(1, epoch + 1):
        acu_loss = 0.0
        model.train()

        for x, labels, attack_labels, edge_index in train_dataloader:
            x, labels, edge_index = [
                item.float().to(device) for item in [x, labels, edge_index]]

            optimizer.zero_grad()
            out = model(x, edge_index).float().to(device)
            loss = loss_func(out, labels)
            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.item())
            acu_loss += loss.item()

        mean_train_loss = acu_loss / len(train_dataloader)
        epoch_train_loss_list.append(mean_train_loss)

        print(f"Epoch ({i_epoch}/{epoch}) | Train Loss: {mean_train_loss:.8f}", flush=True)

        if val_dataloader is not None:
            val_loss, val_result = test(model, val_dataloader)
            val_loss_list.append(val_loss)

            print(f"Epoch ({i_epoch}/{epoch}) | Val Loss: {val_loss:.8f}", flush=True)

            if val_loss < best_val_loss:
                torch.save(model.state_dict(), save_path)
                best_val_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1

            
            if stop_improve_count >= early_stop_win:
                print(f"Early stopping at epoch {i_epoch}")
                break
                

        else:
            if mean_train_loss < best_train_loss:
                torch.save(model.state_dict(), save_path)
                best_train_loss = mean_train_loss

    plt.figure()
    plt.plot(epoch_train_loss_list, label='Training Loss')

    if len(val_loss_list) > 0:
        plt.plot(val_loss_list, label='Validation Loss')

    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()

    final_best_loss = best_val_loss if val_dataloader is not None else best_train_loss
    return train_loss_list, final_best_loss



#Testing
def test(model, dataloader):
    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()

    test_loss_list = []
    now = time.time()

    test_predicted_list = []
    test_ground_list = []
    test_labels_list = []

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    test_len = len(dataloader)

    model.eval()

    i = 0
    acu_loss = 0
    for x, y, labels, edge_index in dataloader:
        x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]
        
        with torch.no_grad():
            predicted = model(x, edge_index).float().to(device)
            loss = loss_func(predicted, y)
            labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)
        
        test_loss_list.append(loss.item())
        acu_loss += loss.item()
        
        i += 1

    test_predicted_list = t_test_predicted_list.tolist()        
    test_ground_list = t_test_ground_list.tolist()        
    test_labels_list = t_test_labels_list.tolist()      
    avg_loss = sum(test_loss_list)/len(test_loss_list)

    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]



#################### EVAL FUNCTIONS ###################
def get_pred_errors(test_res, val_res): 
    test_predict, test_gt = test_res
    val_predict, val_gt = val_res

    test_delta = np.abs(np.subtract(np.array(test_predict).astype(np.float64), np.array(test_gt).astype(np.float64)))
    
    return test_delta

def get_full_pred_errors(test_result, val_result): 
    np_test_result = np.array(test_result)
    np_val_result = np.array(val_result)

    all_scores =  None
    all_normals = None
    feature_num = np_test_result.shape[-1]
    labels = np_test_result[2, :, 0].tolist()

    for i in range(feature_num):
        test_re_list = np_test_result[:2,:,i]
        val_re_list = np_val_result[:2,:,i]

        scores = get_pred_errors(test_re_list, val_re_list)
        normal_dist = get_pred_errors(val_re_list, val_re_list)
    
        if all_scores is None:
            all_scores = scores
            all_normals = normal_dist
        else:
            all_scores = np.vstack((all_scores, scores))
            all_normals = np.vstack((all_normals, normal_dist))

    return all_scores, all_normals



def get_sensor_anomality_quantile_target_ppr(gt_labels, test_result, val_result, target_tick_ppr, q_lo, q_hi, max_iter, tol):
    max_iter = int(max_iter)
    test_errors, val_errors = get_full_pred_errors(test_result, val_result)
    S, T = np.asarray(val_errors).shape

    lo, hi = float(q_lo), float(q_hi)
    q_star, ppr_star = hi, None 
    pred_val_sensors_hi = []
    
    for s in range(S):
        verr = np.asarray(val_errors[s]).reshape(-1)
        tau_hi = np.quantile(verr, q_hi)
        pred_val_sensors_hi.append((verr > tau_hi).astype(int))
        
    tick_pred_val_hi = np.array(pred_val_sensors_hi).any(axis=0).astype(int)
    ppr_hi = tick_pred_val_hi.mean()
    
    for _ in range(max_iter):
        q = (lo + hi) / 2.0
        taus = []
        pred_val_sensors = []
        for s in range(S):
            verr = np.asarray(val_errors[s]).reshape(-1)
            tau = np.quantile(verr, q)
            taus.append(float(tau))
            pred_val_sensors.append((verr > tau).astype(int))

        pred_val_sensors = np.array(pred_val_sensors)
        tick_pred_val = (pred_val_sensors.any(axis=0)).astype(int)
        ppr = tick_pred_val.mean()
        
        if ppr > target_tick_ppr:   
            lo = q
            
        else:                       
            hi = q
            q_star, ppr_star = q, ppr

        if (hi - lo) < tol:
            break
        
    if ppr_star is None:
        ppr_star = ppr_hi
        
    thresholds = []
    pred_sensor_labels = []
    for s in range(S):
        verr = np.asarray(val_errors[s]).reshape(-1)
        terr = np.asarray(test_errors[s]).reshape(-1)
        tau = np.quantile(verr, q_star)
        thresholds.append(float(tau))
        pred_sensor_labels.append((terr > tau).astype(int))

    pred_sensor_labels = np.array(pred_sensor_labels)
    print(f"[ADAPTIVE-QUANTILE] target_tick_ppr={target_tick_ppr:.4f} -> "
        f"q*={q_star:.5f}, val_tick_ppr≈{(ppr_star if ppr_star is not None else np.nan):.5f}")
    return thresholds, pred_sensor_labels


def get_test_performance_quantile_target_ppr(gt_labels, test_result, val_result, target_tick_ppr, q_lo, q_hi, max_iter, tol, k):
    thresholds, pred_sensor_labels = get_sensor_anomality_quantile_target_ppr(gt_labels, test_result, val_result, target_tick_ppr, q_lo, q_hi, max_iter, tol)

    pred_sensor_labels = np.array(pred_sensor_labels)
    pred_labels = (pred_sensor_labels.sum(axis=0) >= k).astype(int)

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    macro_f1 = f1_score(gt_labels, pred_labels, average='macro')
    mcc_coef =  matthews_corrcoef(gt_labels, pred_labels)

    return macro_f1, mcc_coef, classification_report(gt_labels, pred_labels)


def _gt_to_1d(gt, T_pred):
    gt = np.asarray(gt)
    if gt.ndim == 2:
        if gt.shape[1] == 1:
            gt = gt[:, 0]
        else:
            gt = gt[:, 0]
    if gt.shape[0] != T_pred:
        raise ValueError(f"gt length {gt.shape[0]} != T_pred {T_pred}. "
                         "gt not aligned with prediction ticks.")
    return gt.astype(int)


def select_fault_sensors_by_separation(pred_sensor_labels, gt_tick_labels, var_names=None, 
                                       min_selected=1, max_selected=None,label_fault_value=1, return_debug=True):
    
    P = np.asarray(pred_sensor_labels)
    S, T_pred = P.shape

    gt = _gt_to_1d(gt_tick_labels, T_pred)
    fault_mask = (gt == label_fault_value)
    L = int(fault_mask.sum())
    
    counts = P[:, fault_mask].sum(axis=1).astype(float) 
    freqs  = counts / max(L, 1)

    order = np.argsort(counts)[::-1]
    counts_sorted = counts[order]
    freqs_sorted  = freqs[order]

    if S == 1:
        cut_k = 1
    else:
        gaps = counts_sorted[:-1] - counts_sorted[1:]
        cut_k = int(np.argmax(gaps) + 1) 
        
    cut_k = max(cut_k, int(min_selected))
    
    if max_selected is not None:
        cut_k = min(cut_k, int(max_selected))
    selected_idx = order[:cut_k]
    
    if var_names is None:
        selected_names = [f"s{i}" for i in selected_idx]
    else:
        selected_names = [var_names[i] for i in selected_idx]

    selected_idx = selected_idx[np.argsort(counts[selected_idx])[::-1]]
    selected_names = [var_names[i] if var_names is not None else f"s{i}" for i in selected_idx]

    if return_debug:
        dbg = {
            "fault_len_L": L,
            "counts_sorted": counts_sorted.astype(int).tolist(),
            "freqs_sorted": freqs_sorted.tolist(),
            "sorted_names": [var_names[i] if var_names is not None else f"s{i}" for i in order]
        }
        if S > 1:
            dbg["gaps_sorted"] = (counts_sorted[:-1] - counts_sorted[1:]).tolist()
            dbg["cut_k"] = int(len(selected_idx))

        return selected_idx, selected_names, counts.astype(int), freqs, dbg

    return selected_idx, selected_names, counts.astype(int), freqs


def find_attack_blocks(gt_labels):
    y = np.asarray(gt_labels).astype(int)

    padded = np.r_[0, y, 0]
    diff = np.diff(padded)

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    return list(zip(starts, ends))


def summarize_attack_blocks(gt_for_pred, pred_sensor_labels, k=15):
    blocks = find_attack_blocks(gt_for_pred)
    rows = []

    for block_id, (s, e) in enumerate(blocks, start=1):
        P_block = pred_sensor_labels[:, s:e+1]
        tick_counts = P_block.sum(axis=0)

        rows.append({
            "block_id": block_id,
            "start": int(s),
            "end": int(e),
            "length": int(e - s + 1),
            "total_sensor_alarms": int(P_block.sum()),
            "max_sensors_per_tick": int(tick_counts.max()),
            f"detected_ticks_k{k}": int((tick_counts >= k).sum()),
            f"detected_rate_k{k}": float((tick_counts >= k).mean()),
        })

    return pd.DataFrame(rows)


def get_selected_attack_window(
    attack_summary,
    block_id,
    slide_win
):
    row = attack_summary[attack_summary["block_id"] == block_id].iloc[0]

    s = int(row["start"])
    e = int(row["end"])

    raw_s = s + slide_win
    raw_e = e + slide_win

    return raw_s, raw_e


############### NORMAL-FAULT MODEL COMPARISON

def _to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

def _reduce_heads(alpha, head_reduce="sum"):
    a = _to_np(alpha)
    if a.ndim >= 2:
        a = a.reshape(a.shape[0], -1) 
        if head_reduce == "mean":
            a = a.mean(axis=1)
        else:
            a = a.sum(axis=1)
    return a.astype(np.float32)

@torch.no_grad()
def _extract_edges_alpha_per_t(model, X, layer_idx=0, head_reduce="sum",
                               edge_attr_names=("edge_index_1", "att_weight_1")):
    model.eval()
    S, N, w = X.shape
    X = X.to(next(model.parameters()).device)
    _ = model(X, None)

    layer = model.gnn_layers[layer_idx]
    ei_all = _to_np(getattr(layer, edge_attr_names[0])).astype(int)  
    alpha_all = _reduce_heads(getattr(layer, edge_attr_names[1]), head_reduce=head_reduce) 

    src_all, dst_all = ei_all[0], ei_all[1]

    edges_s, alpha_s = [], []
    for t in range(S):
        off = t * N
        m = (src_all >= off) & (src_all < off + N) & (dst_all >= off) & (dst_all < off + N)
        src = (src_all[m] - off).astype(int)
        dst = (dst_all[m] - off).astype(int)
        a   = alpha_all[m].astype(np.float32)
        edges_s.append(np.vstack([src, dst])) 
        alpha_s.append(a)                     
    return edges_s, alpha_s, N, S

def _project_to_base(base_edge_index, edges_t, alpha_t, agg="sum"):
    base = _to_np(base_edge_index).astype(int)
    ei_t = _to_np(edges_t).astype(int)
    a_t  = _to_np(alpha_t).reshape(-1)

    acc = defaultdict(list)
    for s, d, a in zip(ei_t[0], ei_t[1], a_t):
        acc[(int(s), int(d))].append(float(a))

    alpha_full = np.zeros((base.shape[1],), dtype=np.float32)
    for k, (s, d) in enumerate(base.T):
        vals = acc.get((int(s), int(d)), [])
        if not vals:
            alpha_full[k] = 0.0
        else:
            alpha_full[k] = float(np.mean(vals) if agg == "mean" else np.sum(vals))
    return alpha_full

#@torch.no_grad()
def alpha_change_pipeline(
    normal_model,
    fault_model,
    X_normal,         
    X_fault,          
    base_edge_index,  
    slide_win,        
    fault_start,      # raw index in normal test where fault begins
    layer_idx=0,
    head_reduce="sum",
    proj_agg="sum",
    eps=1e-8):
    
    base_ei = _to_np(base_edge_index).astype(int)
    Xn = X_normal[fault_start + slide_win:].float()
    Xf = X_fault[slide_win:].float()

    edges_n, alpha_n, Nn, Sn = _extract_edges_alpha_per_t(normal_model, Xn, layer_idx, head_reduce)
    edges_f, alpha_f, Nf, Sf = _extract_edges_alpha_per_t(fault_model,  Xf, layer_idx, head_reduce)

    E0 = base_ei.shape[1]
    alpha_norm  = np.zeros((Sn, E0), dtype=np.float32)
    alpha_fault = np.zeros((Sf, E0), dtype=np.float32)

    for t in range(Sn):
        alpha_norm[t] = _project_to_base(base_ei, edges_n[t], alpha_n[t], agg=proj_agg)
    for t in range(Sf):
        alpha_fault[t] = _project_to_base(base_ei, edges_f[t], alpha_f[t], agg=proj_agg)

    S = min(alpha_norm.shape[0], alpha_fault.shape[0])
    alpha_norm  = alpha_norm[:S]
    alpha_fault = alpha_fault[:S]

    mn = alpha_norm.mean(axis=0)
    mf = alpha_fault.mean(axis=0)

    mean_diff = np.abs(mf-mn)
    ratio = mf / (mn + eps)
    pct_change = np.abs(100.0 * (mf - mn) / (mn + eps)) 
    log_ratio  = np.abs(np.log((mf + eps) / (mn + eps)))  

    out = {
        "mn": mn, "mf": mf,
        "mean_diff": mean_diff,
        "ratio": ratio,
        "pct_change": pct_change,
        "log_ratio": log_ratio,
        "base_edge_index": base_ei,
        "S": S,
    }
    return out




Edge = Tuple[int, int]
def save_case_study_report(
    output_dir: Union[str, Path],
    title: str,
    edge_index: Union[np.ndarray, Sequence[Sequence[int]]],
    feature_names: Sequence[str],

    quantile: Optional[float] = None,
    graph_method: Optional[str] = None,
    case_description: str = "",
    train_data_name: str = "",
    test_data_name: str = "",

    normalization_info: Optional[Dict[str, Any]] = None,
    normal_model_info: Optional[Dict[str, Any]] = None,
    fault_model_info: Optional[Dict[str, Any]] = None,

    identified_symptoms: Optional[Sequence[Union[int, Dict[str, Any]]]] = None,
    kept_sources: Optional[Sequence[int]] = None,
    path_sets: Optional[Sequence[Sequence[Sequence[Edge]]]] = None,
    conflict_sets: Optional[Sequence[Sequence[Edge]]] = None,
    diagnosis_sets: Optional[Sequence[Sequence[Edge]]] = None,
    ranked_results: Optional[Sequence[Dict[str, Any]]] = None,

    heatmap_fig=None,
    heatmap_png_path: Optional[Union[str, Path]] = None,
    extra_notes: str = "",
    html_filename: str = "case_report.html",
    json_filename: str = "case_report.json",
) -> Dict[str, Any]:
    
    def _to_numpy_edge_index(eidx):
        arr = np.asarray(eidx)
        if arr.ndim != 2 or arr.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E).")
        return arr.astype(int)

    def _safe_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, tuple):
            return list(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def _edge_to_named(edge: Edge) -> Dict[str, Any]:
        u, v = int(edge[0]), int(edge[1])
        return {
            "src_idx": u,
            "dst_idx": v,
            "src": str(feature_names[u]),
            "dst": str(feature_names[v]),
            "edge_str": f"{feature_names[u]} -> {feature_names[v]}",
        }

    def _edge_list_to_named(edges: Sequence[Edge]) -> List[Dict[str, Any]]:
        return [_edge_to_named(e) for e in edges]

    def _path_to_named(path: Sequence[Edge], reverse_for_display: bool = True) -> List[Dict[str, Any]]:
        path_clean = [tuple(map(int, e)) for e in path]
        if reverse_for_display:
            path_clean = list(reversed(path_clean))
        return [_edge_to_named(e) for e in path_clean]

    def _fig_to_base64(fig) -> str:
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def _image_file_to_base64(img_path: Union[str, Path]) -> str:
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _summarize_graph(eidx: np.ndarray) -> Dict[str, Any]:
        src = eidx[0]
        dst = eidx[1]
        edges = [(int(src[i]), int(dst[i])) for i in range(eidx.shape[1])]

        indeg = Counter(dst.tolist())
        outdeg = Counter(src.tolist())
        self_loops = [(u, v) for (u, v) in edges if u == v]

        node_rows = []
        for i in range(len(feature_names)):
            node_rows.append({
                "node_idx": int(i),
                "sensor": str(feature_names[i]),
                "in_degree": int(indeg.get(i, 0)),
                "out_degree": int(outdeg.get(i, 0)),
            })

        return {
            "num_nodes": int(len(feature_names)),
            "num_edges": int(len(edges)),
            "max_in_degree": int(max((r["in_degree"] for r in node_rows), default=0)),
            "max_out_degree": int(max((r["out_degree"] for r in node_rows), default=0)),
            "suggested_topk": int(max((r["in_degree"] for r in node_rows), default=0)),
            "self_loop_included": bool(len(self_loops) > 0),
            "num_self_loops": int(len(self_loops)),
            "node_degree_table": node_rows,
            "edge_table": _edge_list_to_named(edges),
        }

    def _normalize_identified_symptoms(symptoms) -> List[Dict[str, Any]]:
        if symptoms is None:
            return []

        normalized = []
        for s in symptoms:
            if isinstance(s, dict):
                if "symptom_idx" in s:
                    idx = int(s["symptom_idx"])
                    name = s.get("symptom_name", str(feature_names[idx]))
                    normalized.append({
                        "symptom_idx": idx,
                        "symptom_name": str(name)
                    })
                else:
                    normalized.append(s)
            else:
                idx = int(s)
                normalized.append({
                    "symptom_idx": idx,
                    "symptom_name": str(feature_names[idx])
                })
        return normalized

    def _build_incoming_subgraph_records(eidx: np.ndarray, symptom_indices: Sequence[int]) -> List[Dict[str, Any]]:
        src = eidx[0]
        dst = eidx[1]
        all_edges = [(int(src[i]), int(dst[i])) for i in range(eidx.shape[1])]

        records = []
        for s in symptom_indices:
            incoming = [edge for edge in all_edges if edge[1] == int(s)]
            records.append({
                "symptom_idx": int(s),
                "symptom_name": str(feature_names[int(s)]),
                "incoming_edges": _edge_list_to_named(incoming),
                "num_incoming_edges": int(len(incoming)),
            })
        return records

    def _build_symptom_conflict_records(
        kept_sources_,
        path_sets_,
        conflict_sets_,
    ) -> List[Dict[str, Any]]:
        if kept_sources_ is None or path_sets_ is None or conflict_sets_ is None:
            return []

        records = []
        for symptom_idx, paths, cset in zip(kept_sources_, path_sets_, conflict_sets_):
            symptom_idx = int(symptom_idx)
            cset_sorted = sorted([tuple(map(int, e)) for e in cset])

            record = {
                "symptom_idx": symptom_idx,
                "symptom_name": str(feature_names[symptom_idx]),
                "num_paths": int(len(paths)),
                "paths": [],
                "conflict_set_size": int(len(cset_sorted)),
                "conflict_set_edges": _edge_list_to_named(cset_sorted),
            }

            for k, p in enumerate(paths, start=1):
                p_clean = [tuple(map(int, e)) for e in p]
                record["paths"].append({
                    "path_id": int(k),
                    "path_length": int(len(p_clean)),
                    "edges": _path_to_named(p_clean, reverse_for_display=True),
                })

            records.append(record)

        return records

    def _build_diagnosis_records(diags) -> List[Dict[str, Any]]:
        if diags is None:
            return []

        records = []
        for i, diag in enumerate(diags, start=1):
            diag_clean = [tuple(map(int, e)) for e in diag]
            records.append({
                "diagnosis_id": int(i),
                "cardinality": int(len(diag_clean)),
                "edges": _edge_list_to_named(diag_clean),
            })
        return records

    def _build_ranked_results_records(ranked_) -> List[Dict[str, Any]]:
        if ranked_ is None:
            return []

        out = []
        for rank, item in enumerate(ranked_, start=1):
            diag = [tuple(map(int, e)) for e in item.get("diagnosis", [])]
            missing = [tuple(map(int, e)) for e in item.get("missing_edges", [])]

            edge_scores = []
            for edge, score in item.get("edge_scores", []):
                edge = tuple(map(int, edge))
                edge_scores.append({
                    "edge": _edge_to_named(edge),
                    "score": None if score is None else float(score)
                })

            out.append({
                "rank": int(rank),
                "diagnosis_score": None if item.get("diagnosis_score", -np.inf) == -np.inf else float(item.get("diagnosis_score")),
                "diagnosis": _edge_list_to_named(diag),
                "diagnosis_named": item.get("diagnosis_named", [e["edge_str"] for e in _edge_list_to_named(diag)]),
                "edge_scores": edge_scores,
                "missing_edges": _edge_list_to_named(missing),
                "missing_edges_named": item.get("missing_edges_named", [e["edge_str"] for e in _edge_list_to_named(missing)]),
            })
        return out

    def _edge_list_html(edges: List[Dict[str, Any]]) -> str:
        if not edges:
            return "<i>None</i>"
        items = "".join(
            f"<li>{e['edge_str']}</li>"
            for e in edges
        )
        return f"<ul>{items}</ul>"

    def _render_html(report: Dict[str, Any], embedded_heatmap_b64: Optional[str] = None) -> str:
        symptoms_html = ""
        for s in report["identified_symptoms"]:
            symptoms_html += f"<li>{s['symptom_name']}</li>"

        graph_edges_html = _edge_list_html(report["graph_info"]["edge_table"])

        incoming_subgraphs_html = ""
        for rec in report["incoming_subgraph_records"]:
            incoming_subgraphs_html += f"""
            <div class="card">
                <h3>Symptom: {rec['symptom_name']}</h3>
                <p><b>Number of incoming edges:</b> {rec['num_incoming_edges']}</p>
                <p><b>Incoming neighborhood:</b></p>
                {_edge_list_html(rec['incoming_edges'])}
            </div>
            """

        conflict_sections = ""
        for rec in report["symptom_conflict_records"]:
            paths_html = ""
            for p in rec["paths"]:
                paths_html += f"""
                <div style="margin-left:20px; margin-bottom:12px;">
                    <b>Path {p['path_id']}</b> (length={p['path_length']})
                    {_edge_list_html(p['edges'])}
                </div>
                """

            conflict_sections += f"""
            <div class="card">
                <h3>Symptom: {rec['symptom_name']}</h3>
                <p><b>Number of paths:</b> {rec['num_paths']}</p>
                <p><b>Conflict set size:</b> {rec['conflict_set_size']}</p>
                <p><b>Conflict set edges:</b></p>
                {_edge_list_html(rec['conflict_set_edges'])}
                <p><b>Paths:</b></p>
                {paths_html}
            </div>
            """

        diag_html = ""
        for d in report["diagnosis_records"]:
            diag_html += f"""
            <div class="card">
                <h3>Diagnosis {d['diagnosis_id']}</h3>
                <p><b>Cardinality:</b> {d['cardinality']}</p>
                {_edge_list_html(d['edges'])}
            </div>
            """

        ranked_html = ""
        for r in report["ranked_results_records"]:
            score_str = "NOT FOUND" if r["diagnosis_score"] is None else f"{r['diagnosis_score']:.4f}"

            edge_scores_html = ""
            if r["edge_scores"]:
                items = ""
                for es in r["edge_scores"]:
                    s = "NOT FOUND" if es["score"] is None else f"{es['score']:.4f}"
                    items += f"<li>{es['edge']['edge_str']} (score={s})</li>"
                edge_scores_html = f"<ul>{items}</ul>"
            else:
                edge_scores_html = "<i>None</i>"

            ranked_html += f"""
            <div class="card">
                <h3>Rank {r['rank']}</h3>
                <p><b>Diagnosis score:</b> {score_str}</p>
                <p><b>Diagnosis edges:</b></p>
                {_edge_list_html(r['diagnosis'])}
                <p><b>Edge scores:</b></p>
                {edge_scores_html}
                <p><b>Missing edges:</b></p>
                {_edge_list_html(r['missing_edges'])}
            </div>
            """

        heatmap_html = ""
        if embedded_heatmap_b64 is not None:
            heatmap_html = f"""
            <div class="card">
                <h2>Attention Change Heatmap</h2>
                <img src="data:image/png;base64,{embedded_heatmap_b64}" style="max-width:100%; border:1px solid #ccc;" />
            </div>
            """

        html = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <title>{report['title']}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 30px;
                    line-height: 1.5;
                    color: #222;
                }}
                h1, h2 {{
                    color: #143a66;
                }}
                .card {{
                    border: 1px solid #ddd;
                    border-radius: 10px;
                    padding: 16px;
                    margin-bottom: 20px;
                    background: #fafafa;
                }}
                pre {{
                    white-space: pre-wrap;
                    word-wrap: break-word;
                    background: #f4f4f4;
                    padding: 10px;
                    border-radius: 8px;
                }}
                ul {{
                    margin-top: 5px;
                }}
            </style>
        </head>
        <body>
            <h1>{report['title']}</h1>

            <div class="card">
                <h2>General Information</h2>
                <p><b>Case description:</b> {report['case_description'] if report['case_description'] else "<i>Not provided</i>"}</p>
                <p><b>Train data:</b> {report['train_data_name'] if report['train_data_name'] else "<i>Not provided</i>"}</p>
                <p><b>Test data:</b> {report['test_data_name'] if report['test_data_name'] else "<i>Not provided</i>"}</p>
                <p><b>Quantile:</b> {report['quantile']}</p>
                <p><b>Graph method:</b> {report['graph_method']}</p>
            </div>

            <div class="card">
                <h2>Normalization</h2>
                <pre>{json.dumps(report['normalization_info'], indent=2, ensure_ascii=False)}</pre>
            </div>

            <div class="card">
                <h2>Normal Model Info</h2>
                <pre>{json.dumps(report['normal_model_info'], indent=2, ensure_ascii=False)}</pre>
            </div>

            <div class="card">
                <h2>Fault Model Info</h2>
                <pre>{json.dumps(report['fault_model_info'], indent=2, ensure_ascii=False)}</pre>
            </div>

            <div class="card">
                <h2>Graph Summary</h2>
                <p><b>Number of nodes:</b> {report['graph_info']['num_nodes']}</p>
                <p><b>Number of edges:</b> {report['graph_info']['num_edges']}</p>
                <p><b>Max in-degree:</b> {report['graph_info']['max_in_degree']}</p>
                <p><b>Suggested topk:</b> {report['graph_info']['suggested_topk']}</p>
                <p><b>Self-loops included:</b> {report['graph_info']['self_loop_included']}</p>
                <p><b>Number of self-loops:</b> {report['graph_info']['num_self_loops']}</p>
                <p><b>All graph edges:</b></p>
                {graph_edges_html}
            </div>

            <div class="card">
                <h2>Identified Symptoms</h2>
                <ul>{symptoms_html}</ul>
            </div>

            <h2>Incoming Neighborhood Subgraphs</h2>
            {incoming_subgraphs_html if incoming_subgraphs_html else "<i>No symptom subgraphs provided.</i>"}

            <h2>Conflict Sets by Symptom</h2>
            {conflict_sections if conflict_sections else "<i>No conflict sets provided.</i>"}

            <h2>Diagnosis Sets</h2>
            {diag_html if diag_html else "<i>No diagnosis sets provided.</i>"}

            <h2>Ranked Diagnosis Sets</h2>
            {ranked_html if ranked_html else "<i>No ranked diagnosis sets provided.</i>"}

            {heatmap_html}

            <div class="card">
                <h2>Extra Notes</h2>
                <p>{report['extra_notes']}</p>
            </div>
        </body>
        </html>
        """
        return html

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    edge_index_np = _to_numpy_edge_index(edge_index)
    graph_info = _summarize_graph(edge_index_np)
    identified_symptoms_norm = _normalize_identified_symptoms(identified_symptoms)

    if kept_sources is not None:
        incoming_subgraph_records = _build_incoming_subgraph_records(edge_index_np, kept_sources)
    else:
        incoming_subgraph_records = []

    symptom_conflict_records = _build_symptom_conflict_records(
        kept_sources_=kept_sources,
        path_sets_=path_sets,
        conflict_sets_=conflict_sets,
    )

    diagnosis_records = _build_diagnosis_records(diagnosis_sets)
    ranked_results_records = _build_ranked_results_records(ranked_results)

    report_dict = {
        "title": title,
        "case_description": case_description,
        "train_data_name": train_data_name,
        "test_data_name": test_data_name,
        "quantile": quantile,
        "graph_method": graph_method,
        "normalization_info": normalization_info if normalization_info is not None else {},
        "normal_model_info": normal_model_info if normal_model_info is not None else {},
        "fault_model_info": fault_model_info if fault_model_info is not None else {},
        "graph_info": graph_info,
        "identified_symptoms": identified_symptoms_norm,
        "incoming_subgraph_records": incoming_subgraph_records,
        "symptom_conflict_records": symptom_conflict_records,
        "diagnosis_records": diagnosis_records,
        "ranked_results_records": ranked_results_records,
        "extra_notes": extra_notes,
    }

    embedded_heatmap_b64 = None
    if heatmap_fig is not None:
        embedded_heatmap_b64 = _fig_to_base64(heatmap_fig)
    elif heatmap_png_path is not None and Path(heatmap_png_path).exists():
        embedded_heatmap_b64 = _image_file_to_base64(heatmap_png_path)

    with open(output_dir / json_filename, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False, default=_safe_json)

    html = _render_html(report_dict, embedded_heatmap_b64=embedded_heatmap_b64)
    with open(output_dir / html_filename, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] JSON saved to: {output_dir / json_filename}")
    print(f"[OK] HTML saved to: {output_dir / html_filename}")

    return report_dict



