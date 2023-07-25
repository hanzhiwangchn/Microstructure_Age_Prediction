import re, os, json, gzip, shutil, logging
import numpy as np
import pandas as pd
import nibabel as nib

# WAND_FULL_IMAGE_DIR stores each image modality and corresponding age. Different modalities 
# have different ids. 
# WAND_NPY_IMAGE_DIR stores a compact version of above dataset, where all modalities have the 
# same ids. It is obvious smaller than the above dataset.
WAND_IMAGE_DIR = '/cubric/collab/314_wand/bids/'
WAND_FULL_IMAGE_DIR = '/cubric/data/c1809127/314_wand_backup/'
WAND_COMPACT_IMAGE_DIR = '/cubric/data/c1809127/314_wand_compact/'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def prep_wand_age(file_dir):
    """clean wand age file"""
    # clean wand age file
    df = pd.read_csv(file_dir).dropna()
    df['subject_id'] = df['Subject'].apply(lambda x: re.split('[_-]', x)[-1])
    filter_1 = df['subject_id'].str.contains('[x]')
    filter_2 = df['Subject'].str.startswith('1')
    df = df[~(filter_1 | filter_2)].reset_index(drop='true')
    df['subject_age'] = df['DOB'].apply(lambda x: 2023 - int(x.split('/')[-1]))
    df.to_csv('wand_age_clean.csv')

    # extract id and age to json
    age_dict = dict(sorted(dict(zip(df['subject_id'],df['subject_age'])).items()))
    with open('wand_age.json', 'w') as f:
        json.dump(age_dict, f)


def build_wand_image_modality_dir_dict():
    """build a dict with keys being image modality and values being corresponding dirs"""
    image_dir_dict = dict()
    image_dir_dict['KFA_DKI'] = 'derivatives/DKI_dipywithgradcorr/preprocessed/'
    image_dir_dict['ICVF_NODDI'] = 'derivatives/NODDI/preprocessed/'
    image_dir_dict['FA_CHARMED'] = 'derivatives/CHARMED/preprocessed/'
    image_dir_dict['RD_CHARMED'] = 'derivatives/CHARMED/preprocessed/'
    image_dir_dict['MD_CHARMED'] = 'derivatives/CHARMED/preprocessed/'
    image_dir_dict['AD_CHARMED'] = 'derivatives/CHARMED/preprocessed/'
    image_dir_dict['FRtot_CHARMED'] = 'derivatives/CHARMED/preprocessed/'
    image_dir_dict['MWF_mcDESPOT'] = 'derivatives/mcDESPOT/preprocessed/'

    return image_dir_dict


def build_wand_image_modality_fullname_dict():
    """build a dict with keys being image modality and values being corresponding file fullname"""
    image_fullname_dict = dict()
    image_fullname_dict['KFA_DKI'] = 'kurtosis_fractional_anisotropy_KFA.nii'
    image_fullname_dict['ICVF_NODDI'] = 'FIT_ICVF.nii'
    image_fullname_dict['FA_CHARMED'] = 'CHARMED_denoisedMPPCA_driftCo_TED_gibbsCorrSubVoxShift_dtFit_nonlinear_FA.nii'
    image_fullname_dict['RD_CHARMED'] = 'CHARMED_denoisedMPPCA_driftCo_TED_gibbsCorrSubVoxShift_dtFit_nonlinear_RD.nii'
    image_fullname_dict['MD_CHARMED'] = 'CHARMED_denoisedMPPCA_driftCo_TED_gibbsCorrSubVoxShift_dtFit_nonlinear_MD.nii'
    image_fullname_dict['AD_CHARMED'] = 'CHARMED_denoisedMPPCA_driftCo_TED_gibbsCorrSubVoxShift_AD_MRtrix.nii'
    image_fullname_dict['FRtot_CHARMED'] = 'CHARMED_denoisedMPPCA_driftCo_TED_gibbsCorrSubVoxShift_charmed_standard_FRtot.nii'
    image_fullname_dict['MWF_mcDESPOT'] = 'mcDESPOT_3C_f_m.nii'

    return image_fullname_dict


def build_wand_tract_modality_fullname_dict():
    """build a dict with keys being tract modality and values being corresponding file fullname"""
    tract_fullname_dict = dict()
    tract_fullname_dict['KFA_DKI'] = 'kurtosis_fractional_anisotropy_KFA_cm.txt'
    tract_fullname_dict['ICVF_NODDI'] = 'w_ic.w_cm.txt'
    tract_fullname_dict['FA_CHARMED'] = 'CHARMED_denoisedMPPCA_driftCo_TED_gibbsCorrSubVoxShift_dtFit_nonlinear_FA_cm.txt'
    tract_fullname_dict['RD_CHARMED'] = 'CHARMED_denoisedMPPCA_driftCo_TED_gibbsCorrSubVoxShift_dtFit_nonlinear_RD_cm.txt'
    tract_fullname_dict['MD_CHARMED'] = 'CHARMED_denoisedMPPCA_driftCo_TED_gibbsCorrSubVoxShift_dtFit_nonlinear_MD_cm.txt'
    tract_fullname_dict['AD_CHARMED'] = 'CHARMED_denoisedMPPCA_driftCo_TED_gibbsCorrSubVoxShift_AD_MRtrix_cm.txt'
    tract_fullname_dict['FRtot_CHARMED'] = 'CHARMED_denoisedMPPCA_driftCo_TED_gibbsCorrSubVoxShift_charmed_standard_FRtot_cm.txt'
    tract_fullname_dict['MWF_mcDESPOT'] = 'mcDESPOT_3C_f_m_mcf_cm.txt'

    return tract_fullname_dict


def prep_all_wand_images(wand_age_dir, option='compact'):
    """main function to prepare images"""
    # prepare dict for reference
    # modality -> dir
    image_modality_dir_dict = build_wand_image_modality_dir_dict()
    # modality -> fullname
    image_modality_fullname_dict = build_wand_image_modality_fullname_dict()
    # subject_id -> age
    with open(wand_age_dir) as f:
        wand_id_age_dict = json.load(f)
    # option -> output_path
    output_image_dir_dict = {'full': WAND_FULL_IMAGE_DIR, 'compact': WAND_COMPACT_IMAGE_DIR}

    # prepare matched id list
    # there are some mismatches on the subject ids between age csv file and cluster data
    id_list_with_age = set(wand_id_age_dict.keys())
    logger.info(f'number of ids with age: {len(id_list_with_age)}')

    if option == 'compact':
        # iterate all modalities and get all ids in their folders
        image_modality_ids_dict = dict()
        for modality in image_modality_dir_dict.keys():
            full_image_dir = os.path.join(WAND_IMAGE_DIR, image_modality_dir_dict[modality])
            id_list_with_image = set([i.split('-')[-1] for i in os.listdir(path=full_image_dir)])
            image_modality_ids_dict[modality] = id_list_with_image
            logger.info(f'modality: {modality}; number of ids: {len(id_list_with_image)}')
        id_list_with_image = set.intersection(*image_modality_ids_dict.values())
        logger.info(f'number of common ids for all modalities: {len(id_list_with_image)}')
        id_list_available = id_list_with_age.intersection(id_list_with_image)
        logger.info(f'number of common ids for all modalities and age: {len(id_list_available)}')

    # prepare each image modality
    for modality in image_modality_dir_dict.keys():
        if option == 'full':
            # in full version, id_list_available is based on age id and current modality id
            full_image_dir = os.path.join(WAND_IMAGE_DIR, image_modality_dir_dict[modality])
            id_list_with_image = set([i.split('-')[-1] for i in os.listdir(path=full_image_dir)])
            logger.info(f'modality: {modality}; number of ids: {len(id_list_with_image)}')
            id_list_available = id_list_with_age.intersection(id_list_with_image)
            logger.info(f'number of common ids for {modality} and age: {len(id_list_available)}')

        id_list_available = unzip_wand_image(image_modality_dir_dict=image_modality_dir_dict, 
            image_modality_fullname_dict=image_modality_fullname_dict, 
            image_modality=modality, output_image_dir=output_image_dir_dict[option], 
            id_list_available=id_list_available)

        if option == 'full':
            save_wand_image_to_npy(image_modality=modality, wand_id_age_dict=wand_id_age_dict, 
                output_image_dir=output_image_dir_dict[option], id_list_available=id_list_available)

    if option == 'compact':
        for modality in image_modality_dir_dict.keys():
            save_wand_image_to_npy(image_modality=modality, wand_id_age_dict=wand_id_age_dict, 
                output_image_dir=output_image_dir_dict[option], id_list_available=id_list_available)


def unzip_wand_image(image_modality_dir_dict, image_modality_fullname_dict, 
        image_modality, output_image_dir, id_list_available):
    # unzip matched images to .nii format
    full_image_dir = os.path.join(WAND_IMAGE_DIR, image_modality_dir_dict[image_modality])
    os.makedirs(os.path.join(output_image_dir, image_modality), exist_ok=True)
    
    potential_missing_files = []
    for i in id_list_available:
        subject_folder = f'sub-{i}'
        try:
            with gzip.open(
                os.path.join(full_image_dir, subject_folder, f'{subject_folder}_{image_modality_fullname_dict[image_modality]}.gz'), 'rb') as f_in:
                with open(os.path.join(output_image_dir, image_modality, f"{subject_folder}_{image_modality_fullname_dict[image_modality]}"), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except:
            potential_missing_files.append(i)
    logger.info(f'{image_modality} has {len(potential_missing_files)} missing files: {potential_missing_files}')
    for each in potential_missing_files:
        id_list_available.remove(each)
    
    return id_list_available


def save_wand_image_to_npy(image_modality, wand_id_age_dict, output_image_dir, id_list_available):
    """base function to prepare each image modality"""
    # a resultant dict to store age and corresponding images
    res_dict = dict((k, [int(wand_id_age_dict[k])]) for k in id_list_available)
    for j in os.listdir(path=os.path.join(output_image_dir, image_modality)):
        if j.endswith('.nii'):
            subject_id = j.split('-')[-1].split('_')[0]
            if subject_id not in id_list_available:
                continue
            img = nib.load(os.path.join(output_image_dir, image_modality, j))
            img = np.array(img.dataobj)

            res_dict[subject_id].append(np.float32(img))
    
    age_array = np.array([i[0] for i in res_dict.values()])
    img_array = np.array([i[1] for i in res_dict.values()])
    logger.info(f'{image_modality} images has shape {img_array.shape}; age has shape {age_array.shape}')
    np.save(os.path.join(output_image_dir, image_modality, f'subject_age_{image_modality}.npy'), age_array)
    np.save(os.path.join(output_image_dir, image_modality, f'subject_images_{image_modality}.npy'), img_array)


def prep_tract_metrics(wand_id_dir):
    full_metrics_dir = os.path.join(WAND_IMAGE_DIR, 'derivatives/tract_corr/analysis')
    with open(wand_id_dir) as f:
        wand_id_dict = json.load(f)
    id_list_of_age = set(wand_id_dict.keys())

    id_list_of_metrics = list()
    for i in os.listdir(path=full_metrics_dir):
        id_list_of_metrics.append(i.split('_')[0].split('-')[-1])
    id_list_of_metrics = set(id_list_of_metrics)
    id_list_available = id_list_of_age.intersection(id_list_of_metrics)

    tract_fullname_dict = build_wand_tract_modality_fullname_dict()
    dict1 = dict()
    for j in id_list_available:
        dict1[j] = []
        for each in sorted(tract_fullname_dict.keys()):
            df = pd.read_csv(f'sub-{j}_{tract_fullname_dict[each]}', sep=" ", 
                header=None, names=['sub_id', 'tract', 'metrics', 'values'])
            dict[j].append(df['values'].tolist())
        assert len(dict1[j]) == 8
    pass


if __name__ == '__main__':
    # Image data preparation
    # prep_wand_age(file_dir='~/wand_age.csv')
    prep_all_wand_images(wand_age_dir='../../wand_age.json', option='full')
