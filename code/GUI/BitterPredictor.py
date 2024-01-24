import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append("../my_model/")

from mymodel import MyModel
from chemical_info.chem_features_pipeline import total_chem_array
from AttentiveFP.getFeatures_aromaticity_rm import get_smiles_dicts, get_smiles_array

import rdkit
from rdkit import Chem
from rdkit.Chem import Draw

import tkinter as tk
import tkinter.ttk as ttk
import pandas as pd
import json
from PIL import ImageTk, Image


import torch
from src import *


IMG_SIZE = (400, 400)


window = tk.Tk()
window.title("BitterFusionNet")
window.geometry("700x700+100+100")

page1 = tk.Frame(window, width=700, height=700, background="white")
page1.pack(side="top", anchor="center")

logo_img = imagePrepro("./images/logo.png")
loding_img = imagePrepro("./images/loading.gif")


img_frame = tk.Label(page1, width=IMG_SIZE[0], height=IMG_SIZE[1]+100, image=logo_img)
img_frame.pack(side="top", anchor="center")


def startAction(event):
    s = start_input.get().replace(" ", "")

    page1.destroy()
    page2.pack(side="top", anchor="center")


start_input = tk.Entry(page1)
start_input.bind("<Return>", startAction)
start_input.pack(side="bottom", anchor="center")


page2 = tk.Frame(window, width=700, height=700)

img_box = tk.LabelFrame(page2, width=IMG_SIZE[0], height=IMG_SIZE[1]+100, borderwidth=0)
img_box.pack(side="top", anchor="center")



img_frame = tk.Label(img_box, width=IMG_SIZE[0], height=IMG_SIZE[1]+100, image=logo_img)
img_frame.pack(side="top", anchor="center")

img_caption = tk.Label(img_box, text="분자 이름", font=("Helvetica", 20))
img_caption.pack(side="bottom", anchor="center")




def isBitterSMILES(event):
    smiles = entry.get()

    img_frame.configure(image=loding_img)

    try:
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
    except:
        bitter_noti.configure(text="it's not SMILES")
        entry.delete(0, len(entry.get()))
        return
    
    #TODO: smiles를 분자 이름으로 바꾸기
    img_caption.configure(text=smiles)

    kargs = {
        "cheminfo_input_dim": 58,
        "cheminfo_output_dim": 8,
        "radius": 2,
        "T": 2,
        "input_feature_dim": 39,
        "input_bond_dim": 10,
        "fingerprint_dim": 95,
        "output_units_num": 14,
        "p_dropout": 0.3,
    }

    model = MyModel(1, **kargs)
    model = torch.load(
        "../data/model_bitterness_processed_158.pt", map_location=torch.device("cpu")
    )

    feature_dicts = get_smiles_dicts([smiles])
    (
        x_atom,
        x_bonds,
        x_atom_index,
        x_bond_index,
        x_mask,
        smiles_to_rdkit_list,
    ) = get_smiles_array([smiles], feature_dicts)
    x_chemical_info = total_chem_array([smiles], vec_len=50)

    param_list = [x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, x_chemical_info]
    for i in range(len(param_list)):
        if i == 2 or i == 3:
            param_list[i] = torch.LongTensor(param_list[i]).to(torch.device("cpu"))
        else:
            param_list[i] = torch.Tensor(param_list[i]).to(torch.device("cpu"))
    x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, x_chemical_info = param_list

    model.eval()

    _, prediction = model(
        x_atom,
        x_bonds,
        x_atom_index,
        x_bond_index,
        x_mask,
        x_chemical_info,
        has_gpu=False,
    )

    if prediction[0][0] > 0.5:
        result = "Bitter"
    else:
        result = "Not bitter"

    bitter_noti.configure(text=result)

    entry.delete(0, len(entry.get()))

    mol_img = Draw.MolToImage(Chem.MolFromSmiles(smiles))
    mol_img.save("./images/temp_img.png")
    mol_img = imagePrepro("./images/temp_img.png")
    img_frame.image = mol_img

    img_frame.configure(image=mol_img)


def exit_program(window):
    window.destroy()


bottom_box = tk.LabelFrame(page2, borderwidth=0)
bottom_box.pack(side="bottom", anchor="center", pady=10)

label_frame = tk.LabelFrame(bottom_box, borderwidth=0)
label_frame.pack(side="top", anchor="center")

entry = tk.Entry(label_frame)
entry.bind("<Return>", isBitterSMILES)
entry.pack(side="top", anchor="center")

bitter_noti = tk.Label(label_frame, text="canonical SMILES를 입력하세요", anchor="w")
bitter_noti.pack(side="bottom", anchor="center")


window.mainloop()
