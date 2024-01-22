import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import pickle
import numpy as np
import random

from rdkit import Chem
from rdkit.Chem import AllChem

from AttentiveFP.getFeatures_aromaticity_rm import save_smiles_dicts, get_smiles_dicts, get_smiles_array, moltosvg_highlight

from mymodel import mymodel

task_name = 'bitterness'
tasks = ['bitterness']

raw_filename = "./data/bitterness_processed.csv"
feature_filename = raw_filename.replace('.csv','.pickle')
filename = raw_filename.replace('.csv','')
prefix_filename = raw_filename.split('/')[-1].replace('.csv','')
smiles_tasks_df = pd.read_csv(raw_filename)
smilesList = smiles_tasks_df.cano_smiles.values
if os.path.isfile(feature_filename):
    feature_dicts = pickle.load(open(feature_filename, "rb" ))
else:
    feature_dicts = save_smiles_dicts(smilesList,filename)

for i in range(len(smiles_tasks_df)):
    smiles_tasks_df['cano_smiles'][i] = Chem.MolToSmiles(Chem.MolFromSmiles(smiles_tasks_df['cano_smiles'][i]), isomericSmiles=True)

drop_list = ['[Ca+2].[OH-].[OH-]',
'[Br-].[Li+]',
'[Br-].[Br-].[Mg+2]',
'[Cl-].[Cl-].[Mg+2]',
'[Br-].[Na+]',
'[Be+2].[Cl-].[Cl-]',
'[Cl-].[NH4+]',
'[Cl-].[Li+]',
'[Cl-].[K+]',
'[Cl-].[Na+]']

for i in range(len(smiles_tasks_df)):
    if smiles_tasks_df.cano_smiles[i] in drop_list:
        smiles_tasks_df = smiles_tasks_df.drop(i)

smiles_tasks_df = smiles_tasks_df.reset_index(drop=True)

batch_size = 64

ProteinBERT_parameters = {'num_tokens' : 26,
                          'num_annotation' : 8943,
                          'dim' : 512,
                          'dim_global' : 256,
                          'depth' : 6,
                          'narrow_conv_kernel' : 9,
                            'wide_conv_kernel' : 9,
                            'wide_conv_dilation' : 5,
                            'attn_heads' : 8,
                            'attn_dim_head' : 64,
                            'attn_qk_activation' : nn.Tanh(),
                            'local_to_global_attn' : False,
                            'local_self_attn' : False,
                            'num_global_tokens' : 1,
                            'glu_conv' : False
}

FP_parameters = {'radius' : 2,
                    'T' : 2,
                    'input_feature_dim' : 39,
                    'input_bond_dim' : 10,
                    'fingerprint_dim' : 512,
                    'output_units_num' : 512,
                    'p_dropout' : 0.2
    }

test_df = smiles_tasks_df.sample(frac=1/5,random_state=i)
train_df = smiles_tasks_df.drop(test_df.index)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array([smilesList[0]],feature_dicts)
num_atom_features = x_atom.shape[-1]
num_bond_features = x_bonds.shape[-1]
loss_function = nn.CrossEntropyLoss()

model = mymodel(ProteinBERT_parameters, FP_parameters, 1)

def train(model, dataset, optimizer, loss_function):
    model.train()
    valList = np.arange(0,dataset.shape[0])
    np.random.shuffle(valList)
    batch_list = []

    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i+batch_size]
        batch_list.append(batch)
    
    for counter, train_batch in enumerate(batch_list):
        batch_df = dataset.loc[train_batch,:]
        smiles_list = batch_df.cano_smiles.values
        y_val = batch_df.label.values
        
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,feature_dicts)
        atoms_prediction, mol_prediction = model(torch.Tensor(x_atom).to(torch.device('cuda')),torch.Tensor(x_bonds).to(torch.device('cuda')),
                                                torch.LongTensor(x_atom_index).to(torch.device('cuda')),torch.LongTensor(x_bond_index).to(torch.device('cuda')),
                                                torch.Tensor(x_mask).to(torch.device('cuda')))
        
        model.zero_grad()
        loss = loss_function(mol_prediction, torch.Tensor(y_val).view(-1,1).to(torch.device('cuda')))     
        loss.backward()
        optimizer.step()

    optimizer.zero_grad()
    pred = model(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask)
    loss = loss_function(pred, y)
    loss.backward()
    optimizer.step()
    return loss.item()


