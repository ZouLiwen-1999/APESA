# -*- coding: utf-8 -*-
import nibabel as nib
# from utils import *
from skimage import morphology
import networkx as nx
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np
import os
join=os.path.join
from tqdm import tqdm 

def vis(pred_file,mask_file,save_path):
    name=pred_file.split('/')[-1]#文件名
    pred_nii=nib.load(pred_file)
    pred=pred_nii.get_fdata()
    mask_nii=nib.load(mask_file)
    mask=mask_nii.get_fdata()
    TP=np.multiply(mask, pred)
    FP=((pred-mask)==1).astype(np.uint8)
    FN=((mask-pred)==1).astype(np.uint8)
    mask[mask==1]=0
    mask[FN==1]=1
    save_nii = nib.Nifti1Image(mask, mask_nii.affine)
    nib.save(save_nii, join(save_path,name.replace('_0001','')))


if __name__ == '__main__':
    label_path=''
    pred_path=''
    save_path=''
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    names=os.listdir(pred_path)
    for name in tqdm(names):
        if 'nii' in name:
            pred_file=join(pred_path,name)
            mask_file=join(label_path,name.replace('_0001',''))
            vis(pred_file,mask_file,save_path)



