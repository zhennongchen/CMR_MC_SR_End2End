# Motion correction and super-resolution for multi-slice CMR 
**Author: Zhennong Chen, PhD**<br />

This is the GitHub repo for a published paper: <br />
*Motion correction and super-resolution for multi-slice cardiac magnetic resonance imaging via an end-to-end deep learning approach*<br />
[paper link](https://www.sciencedirect.com/science/article/abs/pii/S0895611124000661)<br />
Authors: Zhennong Chen, Hui Ren, Quanzheng Li, Xiang Li<br />

**Citation**: Chen, Zhennong, et al. "Motion correction and super-resolution for multi-slice cardiac magnetic resonance imaging via an end-to-end deep learning approach." Computerized Medical Imaging and Graphics 115 (2024): 102389.

## Description
Accurate reconstruction of a high-resolution 3D volume of the heart is critical for comprehensive cardiac assessments. However, cardiac magnetic resonance (CMR) data is usually acquired as a stack of 2D short-axis (SAX) slices, which suffers from the inter-slice misalignment due to cardiac motion and data sparsity from large gaps between SAX slices. Therefore, we aim to propose an end-to-end deep learning (DL) model to address these two challenges simultaneously, employing specific model components for each challenge. The objective is to reconstruct a high-resolution 3D volume of the heart (VHR) from acquired CMR SAX slices (VLR). We define the transformation from VLR to VHR as a sequential process of **motion correction** and **super-resolution**. Accordingly, our DL model incorporates two distinct components. The first component conducts **motion correction** by predicting displacement vectors to re-position each SAX slice accurately. The second component takes the motion-corrected SAX slices from the first component and performs the **super-resolution** to fill the data gaps. These two components operate in a sequential way, and the entire model is trained ***end-to-end***.<br />

- our model only works on segmented data (e.g., image with LV, myo and RV segmented) rather than original CMR image.<br />
- our goal is to build 3D cardiac volumes from segmented low-resolution contours.<br />


## User Guideline
### Environment Setup
The entire code is [containerized](https://www.docker.com/resources/what-container). This makes setting up environment swift and easy. Make sure you have nvidia-docker and Docker CE [installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) on your machine before going further. <br />
- You can build your own docker from the folder ```docker```. The code is based on tensorflow. <br />
- Make sure you have ```voxelmorph``` installed.

### Data Preparation (we have examples available)
- **High resolutional CMR contours** for simulation and training<br />
    - download the dataset from [Cardiac super-resolution label maps](https://data.mendeley.com/datasets/pw87p286yx/1)
    - we prepare two examples ```example_data/processed_HR_data/1081``` and ```/1256```.  <br />

- **CMR data you want to correct** (in prediction)<br />
    - it has to be segmentations (LV, myo, RV) instead of original image. <br />

- **Patient list** that enumerates all your cases <br />
    - it lists the paired data with ground truth high-resolutional CMR and simulated motion-corrupted low-resolutional CMR. <br />
    - please refer ```example_data/Patient_list/patient_list.xlsx```.<br />


### Experiments
we have design our study into 3 steps.<br /> 
- **step1: data simulation**: use ```step1_data_simulation.ipynb```<br />
    - originally we have CMR with high resolution in z-axis ```example_data/processed_HR_data/```.<br />
    - to do the supervised training, we need to simulate some motion-corrupted low resolution CMR.<br />
    - it will generate two types of data saved in ```example_data/simulated_data```. First is downsampled data (downsample a factor of 5 in z-axis), saved in folder ```ds```. Second is applying inter-slice motion to downsampled data, mimicing the motion artifacts, saved in folder ```normal_motion_X```.<br />

- **step2: model training**: use ```step2_train.py```  <br /> 

- **step3: prediction**: use ```step3_predict.py``` <br /> 
    - it uses trained model to do CMR motion correction and super-resolution. 
    - it saves ```pred_img_LR.nii.gz``` as motion-corrected image (still in low z-resolution) and ```pred_img_HR.nii.gz``` as super-resolutioned image (in high resolution) --> sequential correction of CMR data


### Additional guidelines 
Please contact chenzhennong@gmail.com for any further questions.



