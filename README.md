# Microstructure_Age_Prediction

Considering the size of image, we decide to drop "MWF_mcDESPOT" modality as it has different image shape compared with other modalities.
"KFA_DKI" and "ICVF_NODDI" are NOT going to be dropped as marginal benefit is low.

Experiments:
27th June: we are now testing model performance for each image modality using DenseNet121


TODO: 
1. We need to find better models for this dataset
2. We might want to combine all image modalities (use subject-id to do data split), to increase the size of the dataset.
3. If we have a model ready for final runs, run each model 5 times.
4. We might want to use different spite to run more times.