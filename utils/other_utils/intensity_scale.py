import os
import numpy as np
import SimpleITK as sitk
import seaborn as sns
import matplotlib.pyplot as plt


def LoadImage(path_img):
    try:
        img_itk, voxel_size = Load_itk_image(path_img)
    except FileNotFoundError:
        print('ERROR: File not found', path_img)

    img_numpy = sitk.GetArrayFromImage(img_itk[0])
    img_load = img_numpy.transpose()

    # remove background = 0
    img_load[img_load == 0] = np.nan

    return img_load


def Load_itk_image(path_img):
    # Read header
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(path_img)
    file_reader.ReadImageInformation()
    dim_total = file_reader.GetSize()
    img_vol_ITK = []

    if len(dim_total) == 4:
        dim_vol = (dim_total[0:3])
        # make volume for all images
        img_load_all = sitk.ReadImage(path_img, sitk.sitkFloat32)

        img_vol_ITK = []
        for vol_n in range(dim_total[3]):
            # Extract vol out of object
            size = list(dim_total)
            size[3] = 0
            index = [0, 0, 0, vol_n]
            extractor = sitk.ExtractImageFilter()
            extractor.SetSize(size)
            extractor.SetIndex(index)

            img_vol_ITK.append(extractor.Execute(img_load_all))

    else:
        img_ITK = sitk.ReadImage(path_img, sitk.sitkFloat32)
        img_vol_ITK.append(img_ITK)
        dim_vol = dim_total

    # Voxelsize in mm. forth dimension has no meaning in spacing.
    voxel_size = img_vol_ITK[0].GetSpacing()
    return img_vol_ITK, voxel_size


def NormalizeMinMax(img):
    # Normalization for Volumes
    minVal = np.nanmin(img)
    maxVal = np.nanmax(img)
 
    img_normalized = (img - minVal) * (1 / (maxVal - minVal))
    return img_normalized


def NormalizeZScore(img):
    # Normalization for Volumes
    meanVal = np.nanmean(img)
    stdVal = np.nanstd(img)

    img_normalized = (img - meanVal) * (1 / stdVal)
    return img_normalized


def Norm_image(vol_path, normalization_technique):
    imglist = []
    imgClist = []
    imgNlist = []
    imgNZlist = []
    for file in vol_path:
        img_load = LoadImage(file)
        imglist.append(img_load.flatten())
          
        if 'MinMax' in normalization_technique:
            imgNlist.append(NormalizeMinMax(img_load).flatten())
        if 'Z-Score' in normalization_technique:
            imgNZlist.append(NormalizeZScore(img_load).flatten())

    plt.figure(11)
    fig, ax = plt.subplots(1, 1)
    for i, file in enumerate(vol_path):
        sns.histplot(data=imglist[i].flatten(), kde=False, label=os.path.basename(file)[0:25], 
                     log_scale=False, element="step", fill=False, bins=500, legend=True).set(title='Original')
    ax.legend()
    plt.savefig("Original.png")
    
    if 'MinMax' in normalization_technique:
        fig, ax = plt.subplots(1, 1)
        for i, file in enumerate(vol_path):
            sns.histplot(data=imgNlist[i].flatten(), kde=False, label=os.path.basename(file)[0:25], 
                         log_scale=False, element="step", fill=False, bins=500, legend=True).set(title='Min-Max')
        ax.legend()
        plt.savefig("MinMax.png")
            
    if 'Z-Score' in normalization_technique:
        fig, ax = plt.subplots(1, 1)
        for i, file in enumerate(vol_path):
            sns.histplot(data=imgNZlist[i].flatten(), kde=False, label=os.path.basename(file)[0:25], 
                         log_scale=False, element="step", fill=False, bins=500, legend=True).set(title='Z-Score')
        ax.legend()
        plt.savefig("Zscore.png")


Norm_image(vol_path=[],
           normalization_technique=['Z-Score'])
