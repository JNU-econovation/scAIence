import rdkit
from rdkit import Chem
import pandas as pd
import numpy as np
from rdkit.Chem import QED

from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import UIntSparseIntVect
import re
from gensim.models import Word2Vec

from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec


def mol_from_smiles(smiles_list):
    none_list = []
    mols = [rdkit.Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    
    for idx, mol in enumerate(mols):
        if mol is None:
            none_list.append(smiles_list[idx])
    
    # print("SMILES of None molecules: ", none_list)
        
    return mols


def get_chem_df(mols):
    qed_list = []
    for mol in mols:
        qed_list.append(QED.properties(mol))
        
    # make a dataframe of QED values
    qed_df = pd.DataFrame(qed_list)
    return qed_df

# ----------------------------------------------------------


def mgfp_tokens(mols, radius=3):
    mgfp_list = []
    for mol in mols:
        # print(type(mol))
        mgfp_list.append(rdkit.Chem.rdMolDescriptors.GetMorganFingerprint(mol, radius))         
    # fp_identifier = []
    fp_idf_tokens = []
    max_len = 0
    
    for mgfp in mgfp_list:
        fp_idf = list(map(str, list(UIntSparseIntVect.GetNonzeroElements(mgfp).keys())))
        fp_idf_tokens.extend(fp_idf)
        atom_len = UIntSparseIntVect.GetTotalVal(mgfp)
        if atom_len > max_len:
            max_len = atom_len
        
    fp_idf_tokens = list(set(fp_idf_tokens))
    
    return [fp_idf_tokens]




def sentences2vec(model, sentence, vec_len, unseen=None):
    """modified from """
    
    """Generate vectors for each sentence (list) in a list of sentences. Vector is simply a
    sum of vectors for individual words.
    
    Parameters
    ----------
    sentences : list, array
        List with sentences
    model : word2vec.Word2Vec
        Gensim word2vec model
    unseen : None, str
        Keyword for unseen words. If None, those words are skipped.
        https://stats.stackexchange.com/questions/163005/how-to-set-the-dictionary-for-text-analysis-using-neural-networks/163032#163032

    Returns
    -------
    np.array
    """

    keys = set(list(model.wv.key_to_index.keys()))
    vec = []
    # print(keys)
    if unseen:
        unseen_vec = model.wv.word_vec(unseen)


    if unseen:
        vec.append(sum([model.wv.get_vector(y) if y in set(sentence) & keys
                    else unseen_vec for y in sentence]))
    else:
        vec.append(sum([model.wv.get_vector(y) for y in sentence 
                        if y in set(sentence) & keys]))

    return np.array(vec)



def make_fp_df(mols, vec_len):
    fp_df = pd.DataFrame(columns=['sentence'])
    fp_df = pd.concat([fp_df, pd.DataFrame(columns=range(vec_len))], axis=1)
    # fp_df = pd.DataFrame(columns=range(vec_len))
    # print(fp_df)
    # mols = mol_from_smiles(smiles_list)
    count = 0
    
    tokens = mgfp_tokens(mols, radius=3)
    model =  Word2Vec(tokens, vector_size=vec_len, window=5, min_count=1, workers=4)

    for mol in mols:
        mol_sen = MolSentence(mol2alt_sentence(mol, 1))
        mol_vec = sentences2vec(model, mol_sen, vec_len=vec_len, unseen=None)
        # fp_df = fp_df.append({'sentence': mol_sen, 'mol2vec': mol_vec}, ignore_index=True)
        fp_df = fp_df.append({'sentence': mol_sen}, ignore_index=True)
        # print(mol_vec)
        # raise
        # # add mol2vec to fp_df2
        fp_df.iloc[count, 1:] = list(mol_vec[0])
        count += 1
    fp_df = fp_df.drop(columns=['sentence'], axis=1).reset_index(drop=True)
    return fp_df

# --------------------------------------------------------

def total_chem_array(smiles_list, vec_len):
    # mol = rdkit.Chem.MolFromSmiles("CC(C)C12CC2C(C)C(=O)C1")
    # # temp = rdkit.Chem.rdMolDescriptors.GetMorganFingerprint(mol, 2, bitinfo=None)
    # temp = Chem.rdMolDescriptors.GetMorganFingerprint(mol, 3, bitInfo={})
    s = 'CC(C)C12CC2C(C)C(=O)C1'

    mol = Chem.MolFromSmiles(s)

    temp = rdkit.Chem.rdMolDescriptors.GetMorganFingerprint(mol, 3)
    
    
    # print(temp, "==================================================")

    mols = mol_from_smiles(smiles_list)
    qed_df = get_chem_df(mols)
    # print(qed_df)
    fp_df = make_fp_df(mols, vec_len=vec_len)
    chem_df = pd.concat([qed_df, fp_df], axis=1)
    
    # convert chem_df to numpy array
    chem_numpy = chem_df.to_numpy(float)
    
    return chem_numpy