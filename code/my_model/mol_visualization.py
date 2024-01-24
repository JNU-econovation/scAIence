from AttentiveFP.getFeatures_aromaticity_rm import get_smiles_array, get_smiles_dicts
import numpy as np
import matplotlib
import matplotlib.cm as cm

from rdkit import Chem

def min_max_norm(dataset):
    if isinstance(dataset, list):
        norm_list = list()
        min_value = min(dataset)
        max_value = max(dataset)

        for value in dataset:
            tmp = (value - min_value) / (max_value - min_value)
            norm_list.append(tmp)
    else:
        raise TypeError("dataset should be a list")
    return norm_list


def mol_image_gen(smiles, viz_arrs, path, **kwargs):

    mol_feature_sorted = []
    out_feature_sorted = []
    out_weight_sorted = []

    feature_dicts_adversarial = get_smiles_dicts([smiles])
    (
        x_atom,
        x_bonds,
        x_atom_index,
        x_bond_index,
        x_mask,
        smiles_to_rdkit_list,
    ) = get_smiles_array([smiles], feature_dicts_adversarial)

    (
        atom_feature_viz,
        atom_attention_weight_viz,
        mol_feature_viz,
        mol_feature_unbounded_viz,
        mol_attention_weight_viz,
    ) = viz_arrs

    atom_feature = np.stack([atom_feature_viz[L].cpu().detach().numpy() for L in range(kwargs['radius']+1)])
    atom_weight = np.stack([mol_attention_weight_viz[t].cpu().detach().numpy() for t in range(kwargs['T'])])
    mol_feature = np.stack([mol_feature_viz[t].cpu().detach().numpy() for t in range(kwargs['T'])])

    mol_feature_sorted.extend([mol_feature[:,i,:] for i in range(mol_feature.shape[1])])

    i = 0
    atom_num = i
    ind_mask = x_mask[i]
    ind_atom = smiles_to_rdkit_list[smiles]
    ind_feature = atom_feature[:, i]
    ind_weight = atom_weight[:, i]
    out_feature = []
    out_weight = []

    for j, one_or_zero in enumerate(list(ind_mask)):
        if one_or_zero == 1.0:
            out_feature.append(ind_feature[:, j])
            out_weight.append(ind_weight[:, j])
    
    out_feature_sorted.extend([out_feature[m]] for m in np.argsort(ind_atom))
    out_weight_sorted.extend([out_weight[m]] for m in np.argsort(ind_atom))

    mol = Chem.MolFromSmiles(smiles)

    weight_norm = min_max_norm([out_weight[m][0] for m in np.argsort(ind_atom)])

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1.28)
    cmap = cm.get_cmap('Oranges')
    plt_colors = cm.ScalarMappable(norm=norm, cmap=cmap)
    atom_colors = {}
    weight_norm = np.array(weight_norm).flatten()
    threshold = weight_norm[np.argsort(weight_norm)[-len(ind_atom)//3]]
    weight_norm = np.where(weight_norm < threshold, 0, weight_norm)

    for i in range(len(ind_atom)):
        atom_colors[i] = plt_colors.to_rgba(float(weight_norm[i]))
    Chem.rdDepictor.Compute2DCoords(mol)

    drawer = Chem.Draw.rdMolDraw2D.MolDraw2DCairo(400, 400)
    drawer.SetFontSize(1.0)
    op = drawer.drawOptions()

    mol = Chem.Draw.rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(mol, highlightAtoms=range(0,len(ind_atom)), highlightBonds = [],
                        highlightAtomColors=atom_colors)
    
    drawer.FinishDrawing()
    svg = drawer.WriteDrawingText(path)
    # svg = SVG(svg)
    # svg.save(path)

    print("image saved at ", path)