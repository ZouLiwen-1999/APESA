import nibabel as nb
import numpy as np
from skimage import measure #, morphology
import os
import warnings
warnings.filterwarnings("ignore")
ld=os.listdir
join=os.path.join
ex=os.path.exists
md=os.makedirs
from tqdm import tqdm 
import shutil
import copy 

def getLargestCC(segmentation):
    mask = measure.label(segmentation)
    largestCC = mask == np.argmax(np.bincount(mask.flat)[1:])+1
    return np.uint8(largestCC)

img_path=''
lab_path=''
coarse_path=''
trunk_path=''
this_img_path=''
this_lab_path=''

for path in [this_img_path,this_lab_path,trunk_path]:
    if not ex(path):
        md(path)

trunk_dsc=[]
for case in tqdm(ld(coarse_path)):
    print(case)
    if 'nii' in case and case not in ld(this_lab_path):
        #读取粗分割结果
        coarse_nii=nb.load(join(coarse_path,case))
        coarse_data=coarse_nii.get_fdata()
        trunk=getLargestCC(coarse_data)#获取粗分割主干
        trunk_nii = nb.Nifti1Image(trunk, coarse_nii.affine)
        #读取GT
        lab_nii=nb.load(join(lab_path,case))
        lab_data=lab_nii.get_fdata()

        dsc=dice_equation(trunk, lab_data)
        if dsc>=0:
            trunk_dsc.append(dsc)
        

        #复制图像
        ori_img=join(img_path,case.replace('.nii','_0000.nii'))
        dst_img=join(this_img_path,case.replace('.nii','_0000.nii'))
        shutil.copyfile(ori_img,dst_img)

        #保存truck1作为引导信息
        nb.save(trunk_nii, join(this_img_path,case.replace('.nii','_0001.nii')))
        nb.save(trunk_nii, join(trunk_path,case))

        #保存truck1相对于gt的可生长区域
        grow_region=lab_data-trunk
        grow_region=np.uint8(grow_region==1)
        grow_region_nii = nb.Nifti1Image(grow_region, coarse_nii.affine)
        nb.save(grow_region_nii, join(this_lab_path,case))
        print('Mean DSC of trunk 1:',sum(trunk_dsc)/len(trunk_dsc)) 

 
