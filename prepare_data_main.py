import logging, argparse
from utils.wand_dataset_utils.wand_prep import prep_wand_age, prep_all_wand_images, \
    prep_tract_data

"""This script is used to prepare the dataset for training"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


# ------------------- WAND Image Prep ---------------------
def build_parser_wand_image_prep():
    parser = argparse.ArgumentParser(description='WAND Image preparation')
    parser.add_argument('--option', type=str, default='compact', choices=['compact', 'full'],
                        help='two options for preparing dataset')
    parser.add_argument('--wand-age-dir', type=str, default='Age_labels/wand_age.json',
                        help='dir of wand age json file')
    return parser


def wand_image_prep_main():
    args = build_parser_wand_image_prep().parse_args()
    prep_all_wand_images(args=args)


# ------------------- WAND Tract Prep ---------------------
def build_parser_wand_tract_prep():
    parser = argparse.ArgumentParser(description='WAND Tract preparation')
    parser.add_argument('--wand-tract-dir-prefix', type=str, 
                        default='/Users/hanzhiwang/Datasets/tract_metrics/314_wand_tract_corr/analysis', 
                        help='dir of wand tract data file')
    parser.add_argument('--wand-age-dir', type=str, default='Age_labels/wand_age.json',
                        help='dir of wand age json file')
    parser.add_argument('--wand-tract-data-dir', type=str, default='tract_data/temp',
                        help='dir of tract data for training')
    return parser


def wand_tract_prep_main():
    args = build_parser_wand_tract_prep().parse_args()
    prep_tract_data(args=args)


if __name__ == '__main__':
    # WAND data preparation
    # original wand_age file clean up
    # prep_wand_age(save_dir_name='Age_labels', file_name='wand_age.csv')
    # prepare image data
    # wand_image_prep_main()
    # # prepare tract data
    wand_tract_prep_main()
