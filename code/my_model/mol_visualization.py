from AttentiveFP.getFeatures_aromaticity_rm import get_smiles_array, get_smiles_dicts


def mol_image_gen(smiles, path):
    feature_dicts_adversarial = get_smiles_dicts([smiles])
    (
        x_atom,
        x_bonds,
        x_atom_index,
        x_bond_index,
        x_mask,
        smiles_to_rdkit_list,
    ) = get_smiles_array([smiles], feature_dicts_adversarial)

    
