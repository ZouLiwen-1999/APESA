import nibabel as nib
import os
join = os.path.join
ld=os.listdir
import numpy as np
from tqdm import tqdm
from utils import *

ori_path='./predsTs_f0_v2_CA_RC/'
save_path = './predsTs_f0_v2_CA_RC2/'


if not os.path.exists(save_path):
	os.makedirs(save_path)

for file in tqdm(ld(ori_path)):
	if 'IPMN_dingshiqing_A.nii' in file:
		ori_nii = nib.load(join(ori_path, file))
		ori=ori_nii.get_fdata()
		ori1=np.uint8(ori==1)
		if np.sum(ori1)>0:
			LargestCC_mask = np.uint8(getLargestCC(ori1))#获取最大连通分量
			ori1[LargestCC_mask==1]=2
			ori[ori1==1]=2
		save_nii = nib.Nifti1Image(ori, ori_nii.affine)
		nib.save(save_nii, join(save_path, file))
		# except:
		# 	print(name,'error!')