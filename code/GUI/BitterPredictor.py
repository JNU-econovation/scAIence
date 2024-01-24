import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append("../my_model/")

from mymodel import MyModel
from mol_visualization import mol_image_gen
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

examples = tk.Text(img_box, height=4, font=("Pretendard", 10, "bold"), borderwidth=0, pady=10)
examples.insert(1.0, "Caffeine, Aspartame, Sucrose, Glucose, Fructose, Ethanol, Acetic acid, Sodium chloride\n")
examples.insert(1.0, "Examples\n")
examples.tag_configure("center", justify='center')
examples.tag_add("center", 1.0, "end")
examples.pack(side="bottom", anchor="center")   

examples.configure(bg=img_box.cget('bg'),  font=("Pretendard", 10, "bold"), state="disabled")

img_caption = tk.Label(img_box, text="화합물 이름 또는 canonical SMILES를 입력하세요", font=("Pretendard", 15, "bold"))
img_caption.pack(side="bottom", anchor="center")


def isBitterSMILES(event):
    input = entry.get()

    img_frame.configure(image=loding_img, width=IMG_SIZE[0], height=IMG_SIZE[1]+100)

    try:
        # if input is the name of compound, convert it to SMILES
        smiles = name_to_smiles(input)
        smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
    except:
        try:
            # if input is SMILES, passthrough rdkit
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(input), isomericSmiles=True)
        except:
            # if input is neither SMILES nor compound name, show error message
            bitter_noti.configure(text="it's not SMILES or compound name")
            entry.delete(0, len(entry.get()))
            return
    
    img_caption.configure(text=input)

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

    viz_arrs, _, prediction = model(
        x_atom,
        x_bonds,
        x_atom_index,
        x_bond_index,
        x_mask,
        x_chemical_info,
        has_gpu=False,
    )

    if prediction[0][0] > 0.5:
        result = "BITTER"
    else:
        result = "NOT BITTER"

    bitter_noti.configure(text=result, font=("Pretendard", 20))

    entry.delete(0, len(entry.get()))

    # mol_img = Draw.MolToImage(Chem.MolFromSmiles(smiles))
    # mol_img.save("./images/temp_img.png")
    mol_image_gen(smiles, viz_arrs, "./images/temp_img.png", **kargs)
    mol_img = imagePrepro("./images/temp_img.png")
    img_frame.image = mol_img

    img_frame.configure(image=mol_img)


def exit_program(window):
    window.destroy()


bottom_box = tk.LabelFrame(page2, borderwidth=0)
bottom_box.pack(side="bottom", anchor="center", pady=5)

label_frame = tk.LabelFrame(bottom_box, borderwidth=0)
label_frame.pack(side="top", anchor="center")


entry = tk.Entry(label_frame)
entry.bind("<Return>", isBitterSMILES)
entry.pack(side="top", anchor="center", pady=5)


bitter_noti = tk.Label(label_frame, text="", anchor="w", font=("Pretendard", 20, "bold"))
bitter_noti.pack(side="bottom", anchor="center")

window.mainloop()
