# Image preprocessing pipeline:
#    1. skull-stripping using deep brain (Github)
#    2. image registration to MNI152. FSL FLIRT
#    3. intensity standardization
#    4. blank background removal  bounding box [:, 13:166, 14:206, 0:161]
