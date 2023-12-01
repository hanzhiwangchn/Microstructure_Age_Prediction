import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


def get_img(image_dir, file_name):
    """return a normalized hr image"""
    image = nib.load(image_dir + file_name)
    image = np.array(image.dataobj).astype(np.float32)
    image /= np.max(image)
    return image

image = get_img(image_dir='/Users/hanzhiwang/Datasets/wand_t1w/out/sub-01187_T1w/', file_name='brain.nii')
print(image.shape)

plt.imshow(np.rot90(image[120, :, :]), cmap='gray')
plt.savefig('original_HR.jpg', bbox_inches='tight')
plt.imshow(np.rot90(image[:, 128, :]), cmap='gray')
plt.savefig('synthetic_LR.jpg', bbox_inches='tight')
