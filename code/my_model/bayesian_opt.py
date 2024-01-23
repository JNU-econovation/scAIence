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
import time

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.metrics import r2_score

from AttentiveFP.getFeatures_aromaticity_rm import save_smiles_dicts, get_smiles_dicts, get_smiles_array, moltosvg_highlight

from mymodel import MyModel, AttentiveFP, ChemInfo
from chemical_info.chem_features_pipeline import total_chem_array
from sklearn.metrics import accuracy_score, precision_recall_curve, auc, roc_auc_score

from bayes_opt import BayesianOptimization

task_name = 'bitterness'
tasks = ['bitterness']

raw_filename = "./data/bitterness_processed.csv"
feature_filename = raw_filename.replace('.csv','.pickle')
filename = raw_filename.replace('.csv','')
cheminfo_file = raw_filename.replace('.csv','_cheminfo')
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

if os.path.isfile(cheminfo_file+'.pickle'):
    cheminfo_dict = pickle.load(open(cheminfo_file+'.pickle', "rb" ))
    print("cheminfo_dict loaded!")
else:
    cheminfo_dict = {}
    for i in range(len(smiles_tasks_df)):
        cheminfo_dict[smiles_tasks_df.cano_smiles[i]] = total_chem_array([smiles_tasks_df.cano_smiles[i]], vec_len=50)
    pickle.dump(cheminfo_dict, open(cheminfo_file+'.pickle', "wb" ))
    print("cheminfo_dict saved!")

batch_size = 128

def get_cheminfo_array(smiles_list):
    cheminfo_array = []
    for smiles in smiles_list:
        cheminfo_array.append(cheminfo_dict[smiles].reshape(-1))
    return np.array(cheminfo_array)


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
            # TODO: make function that extract chemical info
            
            x_chemical_info = get_cheminfo_array(smiles_list)

            param_list = [x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, x_chemical_info]
            for i in range(len(param_list)):
                param_list[i] = torch.Tensor(param_list[i]).to(torch.device('cuda'))
            x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, x_chemical_info = param_list

            
            _ , prediction = model(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, x_chemical_info)
            
            model.zero_grad()
            loss = loss_function(prediction, torch.Tensor(y_val).view(-1,1).to(torch.device('cuda')))     
            loss.backward()
            optimizer.step()


def eval(model, dataset):
    model.eval()
    eval_MAE_list = []
    eval_MSE_list = []
    eval_CROSS_ENTROPY_list = []
    
    
    y_val_list = []
    y_pred_list = []
#     np.random.seed(8)
    valList = np.arange(0,dataset.shape[0])
    #shuffle them
#     np.random.shuffle(valList)
    batch_list = []
    for i in range(0, dataset.shape[0], batch_size):
        batch = valList[i:i+batch_size]
        batch_list.append(batch)
    for counter, eval_batch in enumerate(batch_list):
        batch_df = dataset.loc[eval_batch,:]
        smiles_list = batch_df.cano_smiles.values
#         print(batch_df)
        y_val = batch_df.label.values
        
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list,feature_dicts)
        x_chemical_info = get_cheminfo_array(smiles_list)
        # print(x_atom.shape)
        # print(x_chemical_info.shape)
        # raise
        param_list = [x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, x_chemical_info]
        for i in range(len(param_list)):
            param_list[i] = torch.Tensor(param_list[i]).to(torch.device('cuda'))
        x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, x_chemical_info = param_list
        
        _ , prediction = model(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, x_chemical_info)
        MAE = F.l1_loss(prediction, torch.Tensor(y_val).view(-1,1).to(torch.device('cuda')), reduction='none')        
        MSE = F.mse_loss(prediction, torch.Tensor(y_val).view(-1,1).to(torch.device('cuda')), reduction='none')
        CROSS_ENTROPY = F.binary_cross_entropy_with_logits(prediction, torch.Tensor(y_val).view(-1,1).to(torch.device('cuda')), reduction='none')

        
        y_val_list.extend(y_val)
        y_pred_list.extend(np.array(prediction.data.cpu().numpy()))

#         print(x_mask[:2],atoms_prediction.shape, mol_prediction,MSE)
        eval_MAE_list.extend(MAE.data.cpu().numpy())
        eval_MSE_list.extend(MSE.data.cpu().numpy())
        eval_CROSS_ENTROPY_list.extend(CROSS_ENTROPY.data.cpu().numpy())

        

    r2 = r2_score(y_val_list,y_pred_list)
    ACC = accuracy_score(y_val_list, np.array(y_pred_list)>0.5)
    
    reshape_pred = prediction.data.cpu().numpy().reshape(-1)
    
    PRECISION, RECALL, _ = precision_recall_curve(y_val_list, y_pred_list, pos_label=1)
    AUPRC = auc(RECALL, PRECISION)
    AUROC = roc_auc_score(y_val_list, y_pred_list)

    return np.array(eval_MAE_list).mean(), np.array(eval_MSE_list).mean(), np.array(eval_CROSS_ENTROPY_list).mean(), r2,\
            ACC, PRECISION, RECALL, AUPRC, AUROC


# best_param ={}
# best_param["train_epoch"] = 0
# best_param["test_epoch"] = 0
# best_param["train_MSE"] = 9e8
# best_param["test_MSE"] = 9e8
# best_param["train_CROSS_ENTROPY"] = 9e8
# best_param["test_CROSS_ENTROPY"] = 9e8

# round_num = 5


def objective(**params):
    # global kargs

    kargs = {
        'cheminfo_input_dim' : 50 + 8,
        # 'cheminfo_input_dim' : params['cheminfo_input_dim'] + 8,
        'cheminfo_output_dim' : params['cheminfo_output_dim'],
        'radius' : 2,
        'T' : 2,
        'input_feature_dim' : 39,
        'input_bond_dim' : 10,
        'fingerprint_dim' : params['fingerprint_dim'],
        'output_units_num' : params['output_units_num'],
        'p_dropout' : params['p_dropout']
    }

    test_df = smiles_tasks_df.sample(frac=1/5,random_state=i)
    train_df = smiles_tasks_df.drop(test_df.index)
    test_df = test_df.sample(frac=1,)
    train_df = train_df.sample(frac=1)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)



    x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, smiles_to_rdkit_list = get_smiles_array([smilesList[0]],feature_dicts)
    x_chemical_info = total_chem_array([smilesList[0]], vec_len=50)
    num_atom_features = x_atom.shape[-1]
    num_bond_features = x_bonds.shape[-1]




    # x_chemical_info = total_chem_array([smilesList[0]], vec_len=params['chem_dim'])

    # Update the parameters in kargs with the values suggested by Bayesian optimization

    weight_decay = params['weight_decay']
    learning_rate = params['learning_rate']

    # The rest of your code remains unchanged

    # Replace the values you want to optimize with the corresponding values in kargs
    batch_size = 128

    model = MyModel(1, **kargs).cuda()
    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), 10**-learning_rate, weight_decay=10**-weight_decay)


    best_param ={}
    best_param["train_epoch"] = 0
    best_param["test_epoch"] = 0
    best_param["train_MSE"] = 9e8
    best_param["test_MSE"] = 9e8
    best_param["train_CROSS_ENTROPY"] = 9e8
    best_param["test_CROSS_ENTROPY"] = 9e8

    for epoch in range(800):
        # if epoch % 100 == 0:
        #     print(epoch, " started")
        train_MAE, train_MSE, train_CROSS_ENTROPY, train_r2, train_acc, train_pre, train_re, train_auprc, train_auroc = eval(model, train_df)
        test_MAE, test_MSE, test_CROSS_ENTROPY, test_r2, test_acc, test_pre, test_re, test_auprc, test_auroc = eval(model, test_df)
        # ... (rest of your existing code)
        if train_CROSS_ENTROPY < best_param["train_CROSS_ENTROPY"]:
            best_param["train_epoch"] = epoch
            best_param["train_CROSS_ENTROPY"] = train_CROSS_ENTROPY
        if test_CROSS_ENTROPY < best_param["test_CROSS_ENTROPY"]:
            best_param["test_epoch"] = epoch
            best_param["test_CROSS_ENTROPY"] = test_CROSS_ENTROPY
            if np.sqrt(test_CROSS_ENTROPY) < 0.5:
                torch.save(model, './saved_models/model_'+prefix_filename+'_'+str(epoch)+'.pt')
                print("model saved!")

        if (epoch - best_param["train_epoch"] >3) and (epoch - best_param["test_epoch"] >20):     
            break
        train(model, train_df, optimizer, loss_function)
    # Return the value you want to minimize (in this case, test_CROSS_ENTROPY)
    return -best_param["test_CROSS_ENTROPY"]

# Define the parameter space for Bayesian optimization
param_space = {
    # 'cheminfo_input_dim': (50, 100),
    'cheminfo_output_dim': (8, 20),
    'fingerprint_dim': (32, 128),
    'output_units_num': (8, 16),
    'p_dropout': (0.2, 0.4),
    'weight_decay': (3, 5),
    'learning_rate': (2, 3)
}

# Perform Bayesian optimization
opt_result = BayesianOptimization(objective, param_space, verbose=2, random_state=42)
# opt_result.set_gp_params(kernel=None, alpha=1e-5)
opt_result.set_gp_params(alpha=1e-3, n_restarts_optimizer=5)
opt_result.maximize(
    init_points=1,
    n_iter=30
)

# maximize(init_points=2, n_iter=10, acq='ei', xi=0.01)

# Print the best parameters found by Bayesian optimization
print("Best Parameters:", opt_result.max)