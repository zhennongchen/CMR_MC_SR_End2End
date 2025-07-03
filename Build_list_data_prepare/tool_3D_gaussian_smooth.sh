#!/usr/bin/env bash

main_path="/mnt/mount_zc_NAS/motion_correction/data/raw_data"
patients=(${main_path}/nii-images/thin_slice/*/*)


for p in ${patients[*]};
do
    echo ${p}
    patient_smooth_folder=${p}/img-nii-smooth-1mm/
    echo ${patient_smooth_folder}
    mkdir -p ${patient_smooth_folder}

    outputfile=${patient_smooth_folder}img.nii.gz
    echo ${outputfile}


    if [ -f ${outputfile} ];then
      echo "already done this file"
      continue
    else
      c3d ${p}/img-nii/img.nii.gz -smooth 1.06mm ${outputfile}
    fi   


    # phases=(${p}/*)

    # for phase in ${phases[*]};
    # do
    #     phase_smooth_folder=${patient_smooth_folder}$(basename ${phase})/
    #     mkdir -p ${phase_smooth_folder}

    #     images=(${phase}/*.nii.gz)
    #     for i in ${images[*]};
    #     do
    #         echo ${i}
    #         outputfile=${phase_smooth_folder}$(basename ${i})
            
    #         c3d ${i} -smooth 0.638mm ${outputfile} #change the sigma here FWHM = 2.35sigma
            
    #         #c3d ${i} -smooth 0.425mm ${phase_smooth_folder}0.nii.gz
    #         #c3d ${i} -smooth 0.638mm ${phase_smooth_folder}0_0.6.nii.gz
    #         #c3d ${i} -smooth 0.851mm ${phase_smooth_folder}0_0.8.nii.gz
            

    #         done


        
    #     done

    done