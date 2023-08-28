# ThreePoint4DFlow
Code for Machine Learning prediction of velocities from complex data. From the paper: 
  
  Kim D, Jen M-L,Eisenmenger LB, Johnson KM. Accelerated 4D-flow MRI with 3-point encoding enabledby machine learning.Magn Reson Med.2023;89:800-811. doi: 10.1002/mrm.29469  (
  https://onlinelibrary.wiley.com/doi/epdf/10.1002/mrm.29469 )

# Data Preparation
Data is preared in convert_datasets.py where the data is in HDF5 format with the structure:

1. '/IMAGE' 
    - Size:  6x320x320x320
    - Datatype:   H5T_IEEE_F32LE (single)
    - Description: Raw 3 point flow data from complex images stores as 6 channels (real encode 0, imag coil 0, real encode 1, ...) 
    
1. '/VELOCITY' 
    - Size:  3x320x320x320
    - Datatype:   H5T_IEEE_F32LE (single)
    - Description:  Three channel velocity groundtruth

1. '/WEIGHT' 
    - Size:  320x320x320
    - Datatype:   H5T_IEEE_F32LE (single)
    - Description:  Magnitude image, weighting is handled in load_4Dflow  [threepoint_io.py](threepoint_io.py) 

#  Data Training
Data is trained in blocks with [threepoint_train.py](threepoint_train.py) 

#  Inference
Inference is in [threepoint_check.py](threepoint_check.py). It expects data in the same format as the training data. 

  


