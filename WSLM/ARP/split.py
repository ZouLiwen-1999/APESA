'''
从多器官masks中挑选出指定种类的mask,
'''
import nibabel as nib
import os
join = os.path.join
mask_path='./predsTs_f0_v2/'  
save_path = './predsTs_f0_v2_CA/'
# mask_path='./8_output_bduct/label/'  
# save_path = './8_output_bduct/label/'  
if not os.path.exists(save_path):
	os.makedirs(save_path)
cases=os.listdir(mask_path)
ns=len(cases)
n=0
for case in cases:
	n+=1
	if case not in os.listdir(save_path):
		
		print(n,'/',ns)
		mask_nii = nib.load(join(mask_path, case))
		mask=mask_nii.get_fdata()
		mask[mask!=2]=0
		mask[mask==2]=1
		# mask[mask==3]=0
		# mask[mask==4]=0
		# mask[mask==5]=0
		# mask[mask==6]=0
		save_nii = nib.Nifti1Image(mask, mask_nii.affine)
		nib.save(save_nii, join(save_path, case))