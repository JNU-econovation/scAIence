from PIL import ImageTk, Image
import requests
import pubchempy as pcp


IMG_SIZE = (400, 400)


def imagePrepro(dir):

    img = Image.open(dir)
    new_size = list(IMG_SIZE)
    if img.size[0] * 3 / 2 > img.size[1]:
        ratio = IMG_SIZE[0] / img.size[0]
        new_size[1] = int(img.size[1] * ratio)
    else:
        ratio = IMG_SIZE[1] / img.size[1]
        new_size[0] = int(img.size[0] * ratio)

    img_resize = img.resize(new_size, Image.LANCZOS)
    img = ImageTk.PhotoImage(img_resize)
    return img

def imagePrepro_img(img):

    new_size = list(IMG_SIZE)
    if img.size[0] * 3 / 2 > img.size[1]:
        ratio = IMG_SIZE[0] / img.size[0]
        new_size[1] = int(img.size[1] * ratio)
    else:
        ratio = IMG_SIZE[1] / img.size[1]
        new_size[0] = int(img.size[0] * ratio)

    img_resize = img.resize(new_size, Image.LANCZOS)
    img = ImageTk.PhotoImage(img_resize)
    return img


def exit_program(window):
    window.destroy()


def name_to_smiles(name):
    c = pcp.get_compounds(name, 'name')
    try:
        return c[0].canonical_smiles
    except:
        return False
    

def smiles_to_name(smiles):
    c = pcp.get_compounds(smiles, 'smiles')
    try:
        c = pcp.get_compounds(c[0].synonyms[0], 'name')
        return c[0].synonyms[0]
    except:
        return False
