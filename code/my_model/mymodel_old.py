from proteinBERT.proteinBERT import ProteinBERT
from AttentiveFP.AttentiveLayers import Fingerprint
import torch
import torch.nn as nn

class mymodel(ProteinBERT, Fingerprint, nn.Module):
  def __init__(self, ProteinBERT_parameters, FP_parameters, output_dim):
      super(mymodel, self).__init__(self, ProteinBERT_parameters, FP_parameters, output_dim)
      self.output_dim = output_dim

      # ProteinBERT_parameters
      num_tokens = ProteinBERT_parameters['num_tokens']
      num_annotation = ProteinBERT_parameters['num_annotation']
      dim = ProteinBERT_parameters['dim']
      dim_global = ProteinBERT_parameters['dim_global']
      depth = ProteinBERT_parameters['depth']
      narrow_conv_kernel = ProteinBERT_parameters['narrow_conv_kernel']
      wide_conv_kernel = ProteinBERT_parameters['wide_conv_kernel']
      wide_conv_dilation = ProteinBERT_parameters['wide_conv_dilation']
      attn_heads = ProteinBERT_parameters['attn_heads']
      attn_dim_head = ProteinBERT_parameters['attn_dim_head']
      attn_qk_activation = ProteinBERT_parameters['attn_qk_activation']
      local_to_global_attn = ProteinBERT_parameters['local_to_global_attn']
      local_self_attn = ProteinBERT_parameters['local_self_attn']
      num_global_tokens = ProteinBERT_parameters['num_global_tokens']
      glu_conv = ProteinBERT_parameters['glu_conv']
      
      # FP_parameters
      radius = FP_parameters['radius']
      T = FP_parameters['T']
      input_feature_dim = FP_parameters['input_feature_dim']
      input_bond_dim = FP_parameters['input_bond_dim']
      fingerprint_dim = FP_parameters['fingerprint_dim']
      output_units_num = FP_parameters['output_units_num']
      p_dropout = FP_parameters['p_dropout']

      self.proteinBERT = ProteinBERT(num_tokens, num_annotation, dim, dim_global, depth,
        narrow_conv_kernel,  wide_conv_kernel,  wide_conv_dilation,  attn_heads, attn_dim_head,
        attn_qk_activation, local_to_global_attn,  local_self_attn, num_global_tokens, glu_conv)
      self.fingerprint = Fingerprint(radius, T, input_feature_dim, input_bond_dim,\
            fingerprint_dim, output_units_num, p_dropout)
      
      self.output = nn.Linear(output_units_num + dim, output_dim)
      self.sigmoid = nn.Sigmoid()
  
  def forward(self, protein, protein_mask, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask):
      proteinBERT_output = self.proteinBERT(protein, protein_mask)
      fingerprint_output = self.fingerprint(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask)
      output = torch.cat([proteinBERT_output, fingerprint_output], dim=1)
      output = self.output(output)
      output = self.sigmoid(output)
      return output

  

    


