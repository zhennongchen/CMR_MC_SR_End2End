#!/usr/bin/env bash

# run in terminal of your own laptop

##############
## Settings ##
##############

set -o nounset
set -o errexit
set -o pipefail

#shopt -s globstar nullglob

###########
## Logic ##
###########

# define the folder where dcm2niix function is saved, save it in your local laptop
dcm2niix_fld="/Users/zhennongchen/Documents/GitHub/AI_reslice_orthogonal_view/dcm2niix_11-Apr-2019/"

# main_path="/mnt/mount_zc_NAS/motion_correction/data/20221017_head_ct"
main_path='/Volumes/TOSHIBA_4TB/MGH/HFpEF_zhennong'
# define patient lists (the directory where you save all the patient data)
# PATIENTS=(${main_path}/dicoms_manual_seg/ID_1106)
PATIENTS=(${main_path}/dicoms_LAX/ID_0962)

echo ${#PATIENTS[@]}


for p in ${PATIENTS[*]};
do

  echo ${p}
  
  if [ -d ${p} ];
  then

  # patient_id=$(basename $(dirname ${p}))
  patient_id=$(basename ${p})
  echo ${patient_id}
  

  # output_folder=${main_path}/nii_manual_seg
  output_folder=${main_path}/nii_img_LAX
  mkdir -p ${output_folder}/${patient_id}/
  # mkdir -p ${output_folder}/${patient_id}/SAX_ED/
  # nii_folder=${output_folder}/${patient_id}/SAX_ED/ # same as above

  nii_folder=${output_folder}/${patient_id}/
  # IMGS=(${p}/SAX*)  # Find all the images under this patient ID
  IMGS=(${p}/*) 

  for i in $(seq 0 $(( ${#IMGS[*]} - 1 )));
      do

      echo ${IMGS[${i}]}
      
      if [ "$(ls -A ${IMGS[${i}]})" ]; then  # check whether the image folder is empty
        
        filename=$(basename ${IMGS[${i}]})
        # filename='Org3D_frame1'
        o_file=${nii_folder}${filename}.nii.gz # define the name of output nii files, the name will be "timeframe.nii.gz"
        echo ${o_file}

        if [ -f ${o_file} ];then
          echo "already done this file"
          continue

        else
        # if dcm2niix doesn't work (error = ignore image), remove -i y
        ${dcm2niix_fld}dcm2niix -i y -m y -b n -o "${nii_folder}" -f "${filename}" -9 -z y "${IMGS[${i}]}"
        fi

      else
        echo "${IMGS[${i}]} is emtpy; Skipping"
        continue
      fi
      
    done

  else
    echo "${p} missing dicom image folder"
    continue
    
  fi

done
