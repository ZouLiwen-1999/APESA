import nibabel as nib
import os
join = os.path.join
ld=os.listdir
import numpy as np
from tqdm import tqdm
from utils import *

ori_path='./predsTs_f0_v2/'
ref_path='./predsTs_f0_v2_CA_RC2'
save_path = './predsTs_f0_v2_post/'


if not os.path.exists(save_path):
	os.makedirs(save_path)

for file in tqdm(ld(ori_path)):
	if 'IPMN_dingshiqing_A.nii' in file:
		ori_nii = nib.load(join(ori_path, file))
		ori=ori_nii.get_fdata()

		ref_nii = nib.load(join(ref_path, file))
		ref=ref_nii.get_fdata()
		ori[ref==1]=1
		save_nii = nib.Nifti1Image(ori, ori_nii.affine)
		nib.save(save_nii, join(save_path, file))
		# except:
		# 	print(name,'error!')