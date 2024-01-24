# from chemical_info.chemical_info import ChemicalInfo
# from AttentiveFP.AttentiveLayers import Fingerprint


import torch
import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self, output_dim, **kargs):
        ## ChemicalInfo_parameters
        cheminfo_input_dim = kargs["cheminfo_input_dim"]
        cheminfo_output_dim = kargs["cheminfo_output_dim"]

        ## FP_parameters
        radius = kargs["radius"]
        T = kargs["T"]
        input_feature_dim = kargs["input_feature_dim"]
        input_bond_dim = kargs["input_bond_dim"]
        fingerprint_dim = kargs["fingerprint_dim"]
        output_units_num = kargs["output_units_num"]
        p_dropout = kargs["p_dropout"]

        # super(Fingerprint, self).__init__()
        super().__init__()

        self.output_dim = output_dim

        ## AttentiveFP
        # graph attention for atom embedding
        self.atom_fc = nn.Linear(input_feature_dim, fingerprint_dim)
        self.neighbor_fc = nn.Linear(
            input_feature_dim + input_bond_dim, fingerprint_dim
        )
        self.GRUCell = nn.ModuleList(
            [nn.GRUCell(fingerprint_dim, fingerprint_dim) for r in range(radius)]
        )
        self.align = nn.ModuleList(
            [nn.Linear(2 * fingerprint_dim, 1) for r in range(radius)]
        )
        self.attend = nn.ModuleList(
            [nn.Linear(fingerprint_dim, fingerprint_dim) for r in range(radius)]
        )
        # graph attention for molecule embedding
        self.mol_GRUCell = nn.GRUCell(fingerprint_dim, fingerprint_dim)
        self.mol_align = nn.Linear(2 * fingerprint_dim, 1)
        self.mol_attend = nn.Linear(fingerprint_dim, fingerprint_dim)

        self.mol_norm = nn.BatchNorm1d(fingerprint_dim)

        self.dropout = nn.Dropout(p=p_dropout)
        self.mol_output = nn.Linear(fingerprint_dim, output_units_num)

        self.radius = radius
        self.T = T

        ## ChemicalInfo
        self.fc1 = nn.Linear(cheminfo_input_dim, 50)
        self.bn1 = nn.BatchNorm1d(50)

        self.fc2 = nn.Linear(50, 40)
        self.bn2 = nn.BatchNorm1d(40)

        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(40, 32)
        self.bn3 = nn.BatchNorm1d(32)

        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(32, 20)
        self.bn4 = nn.BatchNorm1d(20)

        self.dropout4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(20, cheminfo_output_dim)

        # last layer
        self.output = nn.Linear(output_units_num + cheminfo_output_dim, self.output_dim)
        # self.output = nn.Linear(cheminfo_output_dim, self.output_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        x_atom,
        x_bond,
        x_atom_index,
        x_bond_index,
        x_mask,
        x_chemical_info,
        has_gpu=True,
    ):
        ## AttentiveFP
        x_mask = x_mask.unsqueeze(2)
        batch_size, mol_length, num_atom_feat = x_atom.size()
        atom_feature = F.leaky_relu(self.atom_fc(x_atom))

        atom_feature_viz = []
        atom_feature_viz.append(self.atom_fc(x_atom))

        if not has_gpu:
            x_bond_index = x_bond_index.type(torch.LongTensor)
            x_atom_index = x_atom_index.type(torch.LongTensor)

        else:
            x_bond_index = x_bond_index.type(torch.cuda.LongTensor)
            x_atom_index = x_atom_index.type(torch.cuda.LongTensor)

        bond_neighbor = [x_bond[i][x_bond_index[i]] for i in range(batch_size)]
        bond_neighbor = torch.stack(bond_neighbor, dim=0)
        atom_neighbor = [x_atom[i][x_atom_index[i]] for i in range(batch_size)]
        atom_neighbor = torch.stack(atom_neighbor, dim=0)
        # then concatenate them
        neighbor_feature = torch.cat([atom_neighbor, bond_neighbor], dim=-1)
        neighbor_feature = F.leaky_relu(self.neighbor_fc(neighbor_feature))

        # generate mask to eliminate the influence of blank atoms
        attend_mask = x_atom_index.clone()
        attend_mask[attend_mask != mol_length - 1] = 1
        attend_mask[attend_mask == mol_length - 1] = 0

        if not has_gpu:
            attend_mask = attend_mask.type(torch.FloatTensor).unsqueeze(-1)
        else:
            attend_mask = attend_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        softmax_mask = x_atom_index.clone()
        softmax_mask[softmax_mask != mol_length - 1] = 0
        softmax_mask[
            softmax_mask == mol_length - 1
        ] = -9e8  # make the softmax value extremly small

        if not has_gpu:
            softmax_mask = softmax_mask.type(torch.FloatTensor).unsqueeze(-1)
        else:
            softmax_mask = softmax_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        (
            batch_size,
            mol_length,
            max_neighbor_num,
            fingerprint_dim,
        ) = neighbor_feature.shape
        atom_feature_expand = atom_feature.unsqueeze(-2).expand(
            batch_size, mol_length, max_neighbor_num, fingerprint_dim
        )
        feature_align = torch.cat([atom_feature_expand, neighbor_feature], dim=-1)

        align_score = F.leaky_relu(self.align[0](self.dropout(feature_align)))
        #             print(attention_weight)
        align_score = align_score + softmax_mask
        attention_weight = F.softmax(align_score, -2)
        #             print(attention_weight)
        attention_weight = attention_weight * attend_mask
        #         print(attention_weight)

        atom_attention_weight_viz = []
        atom_attention_weight_viz.append(attention_weight)

        neighbor_feature_transform = self.attend[0](self.dropout(neighbor_feature))
        #             print(features_neighbor_transform.shape)
        context = torch.sum(torch.mul(attention_weight, neighbor_feature_transform), -2)
        #             print(context.shape)
        context = F.elu(context)
        context_reshape = context.view(batch_size * mol_length, fingerprint_dim)
        atom_feature_reshape = atom_feature.view(
            batch_size * mol_length, fingerprint_dim
        )
        atom_feature_reshape = self.GRUCell[0](context_reshape, atom_feature_reshape)
        atom_feature = atom_feature_reshape.view(
            batch_size, mol_length, fingerprint_dim
        )

        # do nonlinearity
        activated_features = F.relu(atom_feature)
        atom_feature_viz.append(activated_features)

        for d in range(self.radius - 1):
            # bonds_indexed = [x_bond[i][torch.cuda.LongTensor(x_bond_index)[i]] for i in range(batch_size)]
            neighbor_feature = [
                activated_features[i][x_atom_index[i]] for i in range(batch_size)
            ]

            # neighbor_feature is a list of 3D tensor, so we need to stack them into a 4D tensor first
            neighbor_feature = torch.stack(neighbor_feature, dim=0)
            atom_feature_expand = activated_features.unsqueeze(-2).expand(
                batch_size, mol_length, max_neighbor_num, fingerprint_dim
            )

            feature_align = torch.cat([atom_feature_expand, neighbor_feature], dim=-1)

            align_score = F.leaky_relu(self.align[d + 1](self.dropout(feature_align)))
            #             print(attention_weight)
            align_score = align_score + softmax_mask
            attention_weight = F.softmax(align_score, -2)
            #             print(attention_weight)
            attention_weight = attention_weight * attend_mask
            #             print(attention_weight)
            atom_attention_weight_viz.append(attention_weight)
            neighbor_feature_transform = self.attend[d + 1](
                self.dropout(neighbor_feature)
            )
            #             print(features_neighbor_transform.shape)
            context = torch.sum(
                torch.mul(attention_weight, neighbor_feature_transform), -2
            )
            #             print(context.shape)
            context = F.elu(context)
            context_reshape = context.view(batch_size * mol_length, fingerprint_dim)
            #             atom_feature_reshape = atom_feature.view(batch_size*mol_length, fingerprint_dim)
            atom_feature_reshape = self.GRUCell[d + 1](
                context_reshape, atom_feature_reshape
            )
            atom_feature = atom_feature_reshape.view(
                batch_size, mol_length, fingerprint_dim
            )

            # do nonlinearity
            activated_features = F.relu(atom_feature)
            atom_feature_viz.append(activated_features)

        mol_feature_unbounded_viz = []
        mol_feature_unbounded_viz.append(torch.sum(atom_feature * x_mask, dim=-2))

        mol_feature = torch.sum(activated_features * x_mask, dim=-2)

        activated_features_mol = F.relu(mol_feature)

        mol_feature_viz = []
        mol_feature_viz.append(mol_feature)

        mol_attention_weight_viz = []
        mol_softmax_mask = x_mask.clone()
        mol_softmax_mask[mol_softmax_mask == 0] = -9e8
        mol_softmax_mask[mol_softmax_mask == 1] = 0

        if not has_gpu:
            mol_softmax_mask = mol_softmax_mask.type(torch.FloatTensor)
        else:
            mol_softmax_mask = mol_softmax_mask.type(torch.cuda.FloatTensor)

        for t in range(self.T):
            mol_prediction_expand = activated_features_mol.unsqueeze(-2).expand(
                batch_size, mol_length, fingerprint_dim
            )
            mol_align = torch.cat([mol_prediction_expand, activated_features], dim=-1)
            mol_align_score = F.leaky_relu(self.mol_align(mol_align))
            mol_align_score = mol_align_score + mol_softmax_mask
            mol_attention_weight = F.softmax(mol_align_score, -2)
            mol_attention_weight = mol_attention_weight * x_mask
            #             print(mol_attention_weight.shape,mol_attention_weight)
            mol_attention_weight_viz.append(mol_attention_weight)

            activated_features_transform = self.mol_attend(
                self.dropout(activated_features)
            )
            #             aggregate embeddings of atoms in a molecule
            mol_context = torch.sum(
                torch.mul(mol_attention_weight, activated_features_transform), -2
            )
            #             print(mol_context.shape,mol_context)
            mol_context = F.elu(mol_context)
            mol_context = self.mol_norm(mol_context)
            mol_feature = self.mol_GRUCell(mol_context, mol_feature)
            #             print(mol_feature.shape,mol_feature)

            mol_feature_unbounded_viz.append(mol_feature)
            # do nonlinearity
            activated_features_mol = F.relu(mol_feature)
            mol_feature_viz.append(activated_features_mol)

        mol_prediction = self.mol_output(self.dropout(mol_feature))

        ## ChemicalInfo
        chemical_info = F.relu(self.fc1(x_chemical_info))
        chemical_info = self.bn1(chemical_info)

        chemical_info = F.relu(self.fc2(chemical_info))
        chemical_info = self.bn2(chemical_info)

        chemical_info = self.dropout2(chemical_info)

        chemical_info = F.relu(self.fc3(chemical_info))
        chemical_info = self.bn3(chemical_info)

        chemical_info = self.dropout3(chemical_info)

        chemical_info = F.relu(self.fc4(chemical_info))
        chemical_info = self.bn4(chemical_info)

        chemical_info = self.dropout4(chemical_info)

        chemical_info = self.fc5(chemical_info)

        # print(mol_prediction.shape, chemical_info.shape)

        x = torch.cat((mol_prediction, chemical_info), dim=1)
        x = self.output(x)
        y = self.sigmoid(x)

        return (
            (
                atom_feature_viz,
                atom_attention_weight_viz,
                mol_feature_viz,
                mol_feature_unbounded_viz,
                mol_attention_weight_viz,
            ),
            x,
            y,
        )


class AttentiveFP(nn.Module):
    def __init__(self, output_dim, **kargs):
        ## ChemicalInfo_parameters
        cheminfo_input_dim = kargs["cheminfo_input_dim"]
        cheminfo_output_dim = kargs["cheminfo_output_dim"]

        ## FP_parameters
        radius = kargs["radius"]
        T = kargs["T"]
        input_feature_dim = kargs["input_feature_dim"]
        input_bond_dim = kargs["input_bond_dim"]
        fingerprint_dim = kargs["fingerprint_dim"]
        output_units_num = kargs["output_units_num"]
        p_dropout = kargs["p_dropout"]

        # super(Fingerprint, self).__init__()
        super().__init__()

        self.output_dim = output_dim

        ## AttentiveFP
        # graph attention for atom embedding
        self.atom_fc = nn.Linear(input_feature_dim, fingerprint_dim)
        self.neighbor_fc = nn.Linear(
            input_feature_dim + input_bond_dim, fingerprint_dim
        )
        self.GRUCell = nn.ModuleList(
            [nn.GRUCell(fingerprint_dim, fingerprint_dim) for r in range(radius)]
        )
        self.align = nn.ModuleList(
            [nn.Linear(2 * fingerprint_dim, 1) for r in range(radius)]
        )
        self.attend = nn.ModuleList(
            [nn.Linear(fingerprint_dim, fingerprint_dim) for r in range(radius)]
        )
        # graph attention for molecule embedding
        self.mol_GRUCell = nn.GRUCell(fingerprint_dim, fingerprint_dim)
        self.mol_align = nn.Linear(2 * fingerprint_dim, 1)
        self.mol_attend = nn.Linear(fingerprint_dim, fingerprint_dim)

        self.mol_norm = nn.BatchNorm1d(fingerprint_dim)

        self.dropout = nn.Dropout(p=p_dropout)
        self.mol_output = nn.Linear(fingerprint_dim, output_units_num)

        self.radius = radius
        self.T = T

        # self.output = nn.Linear(output_units_num + cheminfo_output_dim, self.output_dim)
        self.output = nn.Linear(output_units_num, self.output_dim)

        # self.output = nn.Linear(cheminfo_output_dim, self.output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, x_atom, x_bond, x_atom_index, x_bond_index, x_mask, x_chemical_info
    ):
        x_mask = x_mask.unsqueeze(2)
        batch_size, mol_length, num_atom_feat = x_atom.size()
        atom_feature = F.leaky_relu(self.atom_fc(x_atom))

        x_bond_index = x_bond_index.type(torch.cuda.LongTensor)
        x_atom_index = x_atom_index.type(torch.cuda.LongTensor)
        bond_neighbor = [x_bond[i][x_bond_index[i]] for i in range(batch_size)]
        bond_neighbor = torch.stack(bond_neighbor, dim=0)
        atom_neighbor = [x_atom[i][x_atom_index[i]] for i in range(batch_size)]
        atom_neighbor = torch.stack(atom_neighbor, dim=0)
        # then concatenate them
        neighbor_feature = torch.cat([atom_neighbor, bond_neighbor], dim=-1)
        neighbor_feature = F.leaky_relu(self.neighbor_fc(neighbor_feature))

        # generate mask to eliminate the influence of blank atoms
        attend_mask = x_atom_index.clone()
        attend_mask[attend_mask != mol_length - 1] = 1
        attend_mask[attend_mask == mol_length - 1] = 0
        attend_mask = attend_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        softmax_mask = x_atom_index.clone()
        softmax_mask[softmax_mask != mol_length - 1] = 0
        softmax_mask[
            softmax_mask == mol_length - 1
        ] = -9e8  # make the softmax value extremly small
        softmax_mask = softmax_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        (
            batch_size,
            mol_length,
            max_neighbor_num,
            fingerprint_dim,
        ) = neighbor_feature.shape
        atom_feature_expand = atom_feature.unsqueeze(-2).expand(
            batch_size, mol_length, max_neighbor_num, fingerprint_dim
        )
        feature_align = torch.cat([atom_feature_expand, neighbor_feature], dim=-1)

        align_score = F.leaky_relu(self.align[0](self.dropout(feature_align)))
        #             print(attention_weight)
        align_score = align_score + softmax_mask
        attention_weight = F.softmax(align_score, -2)
        #             print(attention_weight)
        attention_weight = attention_weight * attend_mask
        #         print(attention_weight)
        neighbor_feature_transform = self.attend[0](self.dropout(neighbor_feature))
        #             print(features_neighbor_transform.shape)
        context = torch.sum(torch.mul(attention_weight, neighbor_feature_transform), -2)
        #             print(context.shape)
        context = F.elu(context)
        context_reshape = context.view(batch_size * mol_length, fingerprint_dim)
        atom_feature_reshape = atom_feature.view(
            batch_size * mol_length, fingerprint_dim
        )
        atom_feature_reshape = self.GRUCell[0](context_reshape, atom_feature_reshape)
        atom_feature = atom_feature_reshape.view(
            batch_size, mol_length, fingerprint_dim
        )

        # do nonlinearity
        activated_features = F.relu(atom_feature)

        for d in range(self.radius - 1):
            # bonds_indexed = [x_bond[i][torch.cuda.LongTensor(x_bond_index)[i]] for i in range(batch_size)]
            neighbor_feature = [
                activated_features[i][x_atom_index[i]] for i in range(batch_size)
            ]

            # neighbor_feature is a list of 3D tensor, so we need to stack them into a 4D tensor first
            neighbor_feature = torch.stack(neighbor_feature, dim=0)
            atom_feature_expand = activated_features.unsqueeze(-2).expand(
                batch_size, mol_length, max_neighbor_num, fingerprint_dim
            )

            feature_align = torch.cat([atom_feature_expand, neighbor_feature], dim=-1)

            align_score = F.leaky_relu(self.align[d + 1](self.dropout(feature_align)))
            #             print(attention_weight)
            align_score = align_score + softmax_mask
            attention_weight = F.softmax(align_score, -2)
            #             print(attention_weight)
            attention_weight = attention_weight * attend_mask
            #             print(attention_weight)
            neighbor_feature_transform = self.attend[d + 1](
                self.dropout(neighbor_feature)
            )
            #             print(features_neighbor_transform.shape)
            context = torch.sum(
                torch.mul(attention_weight, neighbor_feature_transform), -2
            )
            #             print(context.shape)
            context = F.elu(context)
            context_reshape = context.view(batch_size * mol_length, fingerprint_dim)
            #             atom_feature_reshape = atom_feature.view(batch_size*mol_length, fingerprint_dim)
            atom_feature_reshape = self.GRUCell[d + 1](
                context_reshape, atom_feature_reshape
            )
            atom_feature = atom_feature_reshape.view(
                batch_size, mol_length, fingerprint_dim
            )

            # do nonlinearity
            activated_features = F.relu(atom_feature)

        mol_feature = torch.sum(activated_features * x_mask, dim=-2)

        activated_features_mol = F.relu(mol_feature)

        mol_softmax_mask = x_mask.clone()
        mol_softmax_mask[mol_softmax_mask == 0] = -9e8
        mol_softmax_mask[mol_softmax_mask == 1] = 0
        mol_softmax_mask = mol_softmax_mask.type(torch.cuda.FloatTensor)

        for t in range(self.T):
            mol_prediction_expand = activated_features_mol.unsqueeze(-2).expand(
                batch_size, mol_length, fingerprint_dim
            )
            mol_align = torch.cat([mol_prediction_expand, activated_features], dim=-1)
            mol_align_score = F.leaky_relu(self.mol_align(mol_align))
            mol_align_score = mol_align_score + mol_softmax_mask
            mol_attention_weight = F.softmax(mol_align_score, -2)
            mol_attention_weight = mol_attention_weight * x_mask
            #             print(mol_attention_weight.shape,mol_attention_weight)
            activated_features_transform = self.mol_attend(
                self.dropout(activated_features)
            )
            #             aggregate embeddings of atoms in a molecule
            mol_context = torch.sum(
                torch.mul(mol_attention_weight, activated_features_transform), -2
            )
            #             print(mol_context.shape,mol_context)
            mol_context = F.elu(mol_context)
            mol_context = self.mol_norm(mol_context)
            mol_feature = self.mol_GRUCell(mol_context, mol_feature)
            #             print(mol_feature.shape,mol_feature)

            # do nonlinearity
            activated_features_mol = F.relu(mol_feature)

        mol_prediction = self.mol_output(self.dropout(mol_feature))

        x = self.output(mol_prediction)
        y = self.sigmoid(x)

        return x, y


class ChemInfo(nn.Module):
    def __init__(self, output_dim, **kargs):
        ## ChemicalInfo_parameters
        cheminfo_input_dim = kargs["cheminfo_input_dim"]
        cheminfo_output_dim = kargs["cheminfo_output_dim"]

        # super(Fingerprint, self).__init__()
        super().__init__()

        self.output_dim = output_dim

        ## ChemicalInfo
        self.fc1 = nn.Linear(cheminfo_input_dim, 50)
        self.bn1 = nn.BatchNorm1d(50)

        self.fc2 = nn.Linear(50, 40)
        self.bn2 = nn.BatchNorm1d(40)

        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(40, 32)
        self.bn3 = nn.BatchNorm1d(32)

        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(32, 20)
        self.bn4 = nn.BatchNorm1d(20)

        self.dropout4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(20, cheminfo_output_dim)

        # last layer

        self.output = nn.Linear(cheminfo_output_dim, self.output_dim)

        # self.output = nn.Linear(cheminfo_output_dim, self.output_dim)

        self.sigmoid = nn.Sigmoid()

    def forward(
        self, x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, x_chemical_info
    ):
        ## ChemicalInfo
        chemical_info = F.relu(self.fc1(x_chemical_info))
        chemical_info = self.bn1(chemical_info)

        chemical_info = F.relu(self.fc2(chemical_info))
        chemical_info = self.bn2(chemical_info)

        chemical_info = self.dropout2(chemical_info)

        chemical_info = F.relu(self.fc3(chemical_info))
        chemical_info = self.bn3(chemical_info)

        chemical_info = self.dropout3(chemical_info)

        chemical_info = F.relu(self.fc4(chemical_info))
        chemical_info = self.bn4(chemical_info)

        chemical_info = self.dropout4(chemical_info)

        chemical_info = self.fc5(chemical_info)

        #     x = torch.cat((mol_prediction, chemical_info), dim=1)
        x = self.output(chemical_info)
        y = self.sigmoid(x)

        return x, y
