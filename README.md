# An end-to-end deep learning solution to perform motion correction and super-resolution concurrently in CMR SAX slices
**Author: Zhennong Chen, PhD**<br />

This is the GitHub repo for the published paper: <br />
*Motion Correction and Super-Resolution for Multi-slice Cardiac Magnetic Resonance Imaging via an End-to-End Deep Learning Approach.*<br />
Authors: Zhennong Chen, Hui Ren, Quanzheng Li, Xiang Li<br />

**Citation**: TBD

## Description
Accurate reconstruction of a high-resolution 3D volume of the heart is critical for comprehensive cardiac assessments. However, cardiac magnetic resonance (CMR) data is usually acquired as a stack of 2D short-axis (SAX) slices, which suffers from the inter-slice misalignment due to cardiac motion and data sparsity from large gaps between SAX slices. Therefore, we aim to propose an end-to-end deep learning (DL) model to address these two challenges simultaneously, employing specific model components for each challenge. <br />

The objective is to reconstruct a high-resolution 3D volume of the heart (V_HR) from acquired CMR SAX slices (V_LR).<br />


## User Guideline
### Deep Learning Model
The deep learning model can be found in ```end2end/model.py```, which is the core innovation of our study. To be more specific, this model generates three outpus: combined_results, final_LR_img, final_HR_img, which correspond to the slice-wise displacement vector, motion-corrected low-resolution volume and final high-resolution volume respectively. (refer to section 2.2 in the paper)<br />

The user can easily create their own data generator as well as model training/testing scripts so we decide not to provide.


### Environment Setup
The entire code is [containerized](https://www.docker.com/resources/what-container). This makes setting up environment swift and easy. Make sure you have nvidia-docker and Docker CE [installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) on your machine before going further. <br />
    - You can build your own docker from provided dockerfile ```docker/dockerfile```. <br />
    

### Data 
We utilized[Cardiac Super-resolution Label Maps](https://data.mendeley.com/datasets/pw87p28). This dataset contains high-spatial resolution 3D balanced steady-state free precession cine CMR sequences. We simulated the inter-slice motion misalignment and z-axis downsampling and thus prepared paris of high-resolution and low-resolution data for supervised training. More details can be found in section 2.5.1 in the paper.

Please contact zchen36@mgh.harvard.edu or chenzhennong@gmail.com for any further questions.




