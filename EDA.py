import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def all_sub_age_dist():
    df = pd.read_csv('wand_age_clean.csv')
    plt.hist(df['subject_age'], bins=25, edgecolor='black')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Histogram of Age for all participants')
    plt.savefig('EDA_plots/all_sub_age_hist.jpg', bbox_inches='tight')

def compact_sub_age_dist():
    age_list = np.load('EDA_plots/subject_age_KFA_DKI.npy')
    print(np.median(age_list))
    max_error = (np.sum(age_list) - np.median(age_list) * len(age_list)) / len(age_list)
    print(max_error)

    exit(0)
    plt.hist(age_list, bins=25, edgecolor='black')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.title('Histogram of Age for selected participants')
    plt.savefig('EDA_plots/compact_sub_age_hist.jpg', bbox_inches='tight')


if __name__ == '__main__':
    compact_sub_age_dist()