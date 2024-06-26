import re, os, json, gzip, shutil, logging
import numpy as np
import pandas as pd
import nibabel as nib

# WAND_FULL_IMAGE_DIR stores each image modality and corresponding age. 
# Different modalities have different ids. 
# WAND_COMPACT_IMAGE_DIR stores a compact version of the above dataset. 
# All modalities have the same ids.
WAND_IMAGE_DIR = '/cubric/collab/314_wand/bids/'
WAND_FULL_IMAGE_DIR = '/cubric/data/c1809127/314_wand_full/'
WAND_COMPACT_IMAGE_DIR = '/cubric/data/c1809127/314_wand_compact/'

logger = logging.getLogger(__name__)


# ------------------- WAND Age Clean Up---------------------
def prep_wand_age(save_dir_name, file_name):
    """clean wand age file"""
    # clean wand age file
    df = pd.read_csv(os.path.join(save_dir_name, file_name)).dropna()
    df['subject_id'] = df['Subject'].apply(lambda x: re.split('[_-]', x)[-1])
    filter_1 = df['subject_id'].str.contains('[x]')
    filter_2 = df['Subject'].str.startswith('1')
    df = df[~(filter_1 | filter_2)].reset_index(drop='true')
    df['subject_age'] = df['DOB'].apply(lambda x: 2023 - int(x.split('/')[-1]))
    df.to_csv(os.path.join(save_dir_name, 'wand_age_clean.csv'))

    # extract id and age to json
    age_dict = dict(sorted(dict(zip(df['subject_id'], df['subject_age'])).items()))
    with open(os.path.join(save_dir_name, 'wand_age.json'), 'w') as f:
        json.dump(age_dict, f)


# ------------------- WAND Image ---------------------    
def build_wand_image_modality_dir_dict():
    """build a dict with keys being image modality and values being corresponding dirs"""
    image_modality_dir_dict = dict()
    image_modality_dir_dict['KFA_DKI'] = 'derivatives/DKI_dipywithgradcorr/preprocessed/'
    # image_modality_dir_dict['ICVF_NODDI'] = 'derivatives/NODDI_MDT/preprocessed/'
    image_modality_dir_dict['FA_CHARMED'] = 'derivatives/CHARMED/preprocessed/'
    image_modality_dir_dict['RD_CHARMED'] = 'derivatives/CHARMED/preprocessed/'
    image_modality_dir_dict['MD_CHARMED'] = 'derivatives/CHARMED/preprocessed/'
    image_modality_dir_dict['AD_CHARMED'] = 'derivatives/CHARMED/preprocessed/'
    image_modality_dir_dict['FRtot_CHARMED'] = 'derivatives/CHARMED/preprocessed/'
    # image_modality_dir_dict['MWF_mcDESPOT'] = 'derivatives/mcDESPOT/preprocessed/'

    return image_modality_dir_dict


def build_wand_image_modality_fullname_dict():
    """build a dict with keys being image modality and values being corresponding file fullname"""
    image_modality_fullname_dict = dict()
    image_modality_fullname_dict['KFA_DKI'] = 'kurtosis_fractional_anisotropy_KFA.nii'
    # image_modality_fullname_dict['ICVF_NODDI'] = 'w_ic.w.nii'
    image_modality_fullname_dict['FA_CHARMED'] = 'CHARMED_denoisedMPPCA_driftCo_TED_gibbsCorrSubVoxShift_dtFit_nonlinear_FA.nii'
    image_modality_fullname_dict['RD_CHARMED'] = 'CHARMED_denoisedMPPCA_driftCo_TED_gibbsCorrSubVoxShift_dtFit_nonlinear_RD.nii'
    image_modality_fullname_dict['MD_CHARMED'] = 'CHARMED_denoisedMPPCA_driftCo_TED_gibbsCorrSubVoxShift_dtFit_nonlinear_MD.nii'
    image_modality_fullname_dict['AD_CHARMED'] = 'CHARMED_denoisedMPPCA_driftCo_TED_gibbsCorrSubVoxShift_AD_MRtrix.nii'
    image_modality_fullname_dict['FRtot_CHARMED'] = 'CHARMED_denoisedMPPCA_driftCo_TED_gibbsCorrSubVoxShift_charmed_standard_FRtot.nii'
    # image_modality_fullname_dict['MWF_mcDESPOT'] = 'mcDESPOT_3C_f_m.nii'

    return image_modality_fullname_dict


def prep_all_wand_images(args):
    """main function to prepare wand images"""
    # prepare dict for reference
    # image modality -> dir
    image_modality_dir_dict = build_wand_image_modality_dir_dict()
    # image modality -> fullname
    image_modality_fullname_dict = build_wand_image_modality_fullname_dict()
    # subject-id -> age
    with open(args.wand_age_dir) as f:
        wand_id_age_dict = json.load(f)
    # option -> output_path
    output_image_dir_dict = {'full': WAND_FULL_IMAGE_DIR, 'compact': WAND_COMPACT_IMAGE_DIR}

    # prepare matched id list
    # there are some mismatches on the subject ids between age csv file and cluster data
    id_list_with_age = set(wand_id_age_dict.keys())
    logger.info(f'number of ids with age: {len(id_list_with_age)}')

    # compact version 
    if args.option == 'compact':
        # iterate all modalities, get all ids in their folders and find the common ones
        image_modality_ids_dict = dict()
        for modality in image_modality_dir_dict.keys():
            full_image_dir = os.path.join(WAND_IMAGE_DIR, image_modality_dir_dict[modality])
            # list all available sub-id folder
            id_list_with_image = set([i.split('-')[-1] for i in os.listdir(path=full_image_dir)])
            image_modality_ids_dict[modality] = id_list_with_image
            logger.info(f'modality: {modality}; number of ids: {len(id_list_with_image)}')

        # find common ids for all modalities
        id_list_with_image = set.intersection(*image_modality_ids_dict.values())
        logger.info(f'number of common ids for all modalities: {len(id_list_with_image)}')
        # find common ids between age and modalities
        id_list_available = id_list_with_age.intersection(id_list_with_image)
        logger.info(f'number of common ids for all modalities and age: {len(id_list_available)}')
 
        # In compact version, id_list_available is updated for all modalities to make sure 
        # all modalities are of the same length as .npy file
        for modality in image_modality_dir_dict.keys():
            id_list_available = unzip_wand_image(image_modality_dir_dict=image_modality_dir_dict, 
                                                 image_modality_fullname_dict=image_modality_fullname_dict, 
                                                 image_modality=modality, output_image_dir=output_image_dir_dict[args.option], 
                                                 id_list_available=id_list_available)
        
        # After unzipping .gz file and updating id_list_available, save .nii to .npy
        for modality in image_modality_dir_dict.keys():
            save_wand_image_to_npy(image_modality=modality, wand_id_age_dict=wand_id_age_dict, 
                                   output_image_dir=output_image_dir_dict[args.option], id_list_available=id_list_available)
        
        # ordering only matters in compact version
        id_ordering_check(image_modality_dir_dict=image_modality_dir_dict, 
                          output_image_dir=output_image_dir_dict[args.option])

    # full version 
    elif args.option == 'full': 
        # In the full version, each modality will be compared with age id to find common ids
        for modality in image_modality_dir_dict.keys():
            # in full version, id_list_available is based on age id and current modality id
            full_image_dir = os.path.join(WAND_IMAGE_DIR, image_modality_dir_dict[modality])
            # list all available sub-id folder
            id_list_with_image = set([i.split('-')[-1] for i in os.listdir(path=full_image_dir)])
            logger.info(f'modality: {modality}; number of ids: {len(id_list_with_image)}')
            # find common ids between age and current modality
            id_list_available = id_list_with_age.intersection(id_list_with_image)
            logger.info(f'number of common ids for {modality} and age: {len(id_list_available)}')

            # ensure all ids should have corresponding file
            id_list_available = unzip_wand_image(image_modality_dir_dict=image_modality_dir_dict, 
                                                 image_modality_fullname_dict=image_modality_fullname_dict, 
                                                 image_modality=modality, output_image_dir=output_image_dir_dict[args.option], 
                                                 id_list_available=id_list_available)

            save_wand_image_to_npy(image_modality=modality, wand_id_age_dict=wand_id_age_dict, 
                                   output_image_dir=output_image_dir_dict[args.option], id_list_available=id_list_available)


def unzip_wand_image(image_modality_dir_dict, image_modality_fullname_dict, image_modality, output_image_dir, id_list_available):
    # unzip matched images to .nii format
    full_image_dir = os.path.join(WAND_IMAGE_DIR, image_modality_dir_dict[image_modality])
    os.makedirs(os.path.join(output_image_dir, image_modality), exist_ok=True)
    
    # some sub-id folder are empty for unknown reason
    id_list_with_missing_files = []
    for i in id_list_available:
        subject_folder = f'sub-{i}'
        try:
            # NOTE ICVF_NODDI file does not have sub-id prefix in the beginning
            if image_modality == 'ICVF_NODDI':
                with gzip.open(os.path.join(full_image_dir, subject_folder, f'{image_modality_fullname_dict[image_modality]}.gz'), 'rb') as f_in:
                    with open(os.path.join(output_image_dir, image_modality, f"{subject_folder}_{image_modality_fullname_dict[image_modality]}"), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                with gzip.open(os.path.join(full_image_dir, subject_folder, f'{subject_folder}_{image_modality_fullname_dict[image_modality]}.gz'), 'rb') as f_in:
                    with open(os.path.join(output_image_dir, image_modality, f"{subject_folder}_{image_modality_fullname_dict[image_modality]}"), 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
        except:
            id_list_with_missing_files.append(i)
    logger.info(f'{image_modality} has {len(id_list_with_missing_files)} missing files: {id_list_with_missing_files}')
    
    for j in id_list_with_missing_files:
        id_list_available.remove(j)
    
    return id_list_available


def save_wand_image_to_npy(image_modality, wand_id_age_dict, output_image_dir, id_list_available):
    """base function to prepare each image modality"""
    # a resultant dict to store age and corresponding images
    res_dict = dict((k, [int(wand_id_age_dict[k])]) for k in sorted(id_list_available))
    for j in os.listdir(path=os.path.join(output_image_dir, image_modality)):
        if j.endswith('.nii'):
            subject_id = j.split('-')[-1].split('_')[0]
            # Due to the way we unzip file, some .nii files may not be in id available list
            if subject_id not in id_list_available:
                continue
            img = nib.load(os.path.join(output_image_dir, image_modality, j))
            img = np.array(img.dataobj)

            res_dict[subject_id].append(np.float32(img))
    
    id_array = np.array(list(res_dict.keys()))
    age_array = np.array([i[0] for i in res_dict.values()])
    img_array = np.array([i[1] for i in res_dict.values()])

    rnd_check = len(id_array) // 2
    assert wand_id_age_dict[id_array[rnd_check]] == age_array[rnd_check]

    if image_modality == 'ICVF_NODDI':
        img_array = np.squeeze(img_array, axis=-1)

    logger.info(f'{image_modality} images has shape {img_array.shape}; age has shape {age_array.shape}')
    np.save(os.path.join(output_image_dir, image_modality, f'subject_id_{image_modality}.npy'), id_array)
    np.save(os.path.join(output_image_dir, image_modality, f'subject_age_{image_modality}.npy'), age_array)
    np.save(os.path.join(output_image_dir, image_modality, f'subject_images_{image_modality}.npy'), img_array)


def id_ordering_check(image_modality_dir_dict, output_image_dir):
    """we want to check if id and age across all modalities have the same ordering"""
    dict1 = dict()
    for modality in image_modality_dir_dict.keys():
        id_array = np.load(os.path.join(output_image_dir, modality, f'subject_id_{modality}.npy'))
        dict1[modality] = id_array
    
    first_modality = list(image_modality_dir_dict.keys())[0]
    for modality in image_modality_dir_dict.keys():
        assert np.array_equal(dict1[first_modality], dict1[modality])

    logger.info('id ordering check passed')


# ------------------- WAND Tract ---------------------   
def prep_tract_data(args):
    """prepare tract metrics value for age prediction"""
    # load age json file
    with open(args.wand_age_dir) as f:
        wand_id_age_dict = json.load(f)
    id_list_of_age = set(wand_id_age_dict.keys())

    # remove subject if their tract of interest is not 29 and find common ids in all 8 tract metrics 
    id_dict = dict()
    tract_fullname_dict = build_wand_tract_metric_fullname_dict()
    # check if a subject has 29 tract regions
    for tract_metric in tract_fullname_dict.keys():
        df = pd.read_csv(os.path.join(args.wand_tract_dir_prefix, tract_fullname_dict[tract_metric]), 
                         sep=" ", header=None, names=['sub_id', 'segments', 'metrics', 'values'])
        df_full_segments = df.groupby("sub_id").filter(lambda x: len(x) == 29)
        id_dict[tract_metric] = set([i.split('-')[-1] for i in set(df_full_segments.sub_id.to_list())])
    # find common ids in all 8 tract metrics
    common_id = set.intersection(*(set(val) for val in id_dict.values()))
    
    # find common id with age file
    common_id = sorted(common_id.intersection(id_list_of_age))

    # save age array to file
    age = np.array([int(wand_id_age_dict[i]) for i in sorted(common_id)])
    np.save(os.path.join(args.wand_tract_data_dir, 'tract_age_compact.npy'), age)
    
    # extract tract data
    common_id = sorted(['sub-' + i for i in common_id])

    # save selected subject id to txt file
    file_path = 'wand_tract_training_ids_sorted.txt'
    with open(file_path, 'w') as file:
        for item in common_id:
            file.write("%s\n" % item)

    # prepare tract data
    full_dataset = []
    for tract_metric in tract_fullname_dict.keys():
        df = pd.read_csv(os.path.join(args.wand_tract_dir_prefix, tract_fullname_dict[tract_metric]), 
                         sep=" ", header=None, names=['sub_id', 'segments', 'metrics', 'values'])
        
        df_selected = df[df['sub_id'].isin(common_id)]
        tract_values = np.array(df_selected['values']).reshape(-1, 29)
        full_dataset.append(tract_values)
    
    full_dataset = np.stack(full_dataset, axis=0)
    full_dataset = np.transpose(full_dataset, (1, 2, 0))
    # save tract array to file
    np.save(os.path.join(args.wand_tract_data_dir, 'tract_value_compact.npy'), full_dataset)


def build_wand_tract_metric_fullname_dict():
    """
    build a dict with keys being tract metric and values being corresponding file fullname
    QMT is not being used since it contains subjects
    """
    tract_fullname_dict = dict()
    tract_fullname_dict['KFA_DKI'] = 'allsubs_kurtosis_fractional_anisotropy_KFA.txt'
    tract_fullname_dict['ICVF_NODDI'] = 'allsubs_ICVF.txt'
    tract_fullname_dict['AD_CHARMED'] = 'allsubs_fractional_anisotropy_AD.txt'
    tract_fullname_dict['FA_CHARMED'] = 'allsubs_fractional_anisotropy_FA.txt'
    tract_fullname_dict['RD_CHARMED'] = 'allsubs_radial_diffusivity_RD.txt'
    tract_fullname_dict['MD_CHARMED'] = 'allsubs_mean_diffusivity_MD.txt'
    tract_fullname_dict['FRtot_CHARMED'] = 'allsubs_CHARMED_FRtot.txt'
    tract_fullname_dict['MWF_mcDESPOT'] = 'allsubs_mcDESPOT_3C_f_m_mcf.txt'
    # tract_fullname_dict['QMT'] = 'allsubs_QMT_f_b_mcf.txt'

    return tract_fullname_dict
