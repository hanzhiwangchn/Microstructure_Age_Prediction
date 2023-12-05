import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# def save_wand_image_to_npy():
#     """base function to prepare each image modality"""
#     # File path to read the text file
#     file_path = 'wand_tract_training_ids_sorted.txt'
#     with open(file_path, 'r') as file:
#         sub_id_list = [line.strip() for line in file]
#     print(sub_id_list)

#     image_list = []
#     for each_sub in sub_id_list:
#         image = nib.load(os.path.join('314_wand_mri_ex_MNI', f'{each_sub}_brain_mni.nii.gz'))
#         image = np.array(image.dataobj)
#         image_list.append(image)

#     assert len(image_list) == 123

#     image_array = np.stack(image_list, axis=0)
#     print(image_array.shape)

#     np.save('wand_t1w.npy', image_array)

def function():
    data = np.load('/Users/hanzhiwang/Datasets/wand_t1w_cropped.npy')
    data_small = np.float32(data)
    print(data.dtype)
    print(data_small.dtype)
    np.save('/Users/hanzhiwang/Datasets/wand_t1w_cropped_float32.npy', data_small)

function()






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
