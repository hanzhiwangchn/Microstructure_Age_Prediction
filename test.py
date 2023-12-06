import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns

# def unzip_wand_image():
#     for filename in os.listdir('314_wand_mri_extracted'):
#         print(f'{filename}')
#         sub_id = filename.split("_")[0]
#         with open(os.path.join('314_wand_mri_extracted', filename, 'brain.nii'), 'rb') as f_in:
#             with open(os.path.join('314_wand_mri_ex', f"{sub_id}_brain.nii"), 'wb') as f_out:
#                         shutil.copyfileobj(f_in, f_out)

# unzip_wand_image()

# def get_img(image_dir, file_name):
#     """return a normalized hr image"""
#     image = nib.load(image_dir + file_name)
#     image = np.array(image.dataobj).astype(np.float32)
#     image /= np.max(image)
#     return image

# image = get_img(image_dir='/Users/hanzhiwang/Datasets/wand_t1w/out/sub-01187_T1w/', file_name='brain.nii')
# print(image.shape)

# plt.imshow(np.rot90(image[120, :, :]), cmap='gray')
# plt.savefig('original_HR.jpg', bbox_inches='tight')
# plt.imshow(np.rot90(image[:, 128, :]), cmap='gray')
# plt.savefig('synthetic_LR.jpg', bbox_inches='tight')
