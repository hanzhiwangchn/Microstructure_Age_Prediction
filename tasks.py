# Image preprocessing pipeline:
#    1. skull-stripping using deep brain (Github)
#    2. image registration to MNI152. FSL FLIRT
#    3. intensity standardization
#    4. blank background removal  bounding box [:, 13:166, 14:206, 0:161]

# model performance check and ensemble with microstructure models (TODO)

# tract code need refactor, result is finished

# Performance check before Sat