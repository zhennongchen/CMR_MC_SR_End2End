#!/usr/bin/env bash
# run in docker 

main_path="/mnt/mount_zc_NAS/HFpEF/data/Sunny_Brooks/LAX/nii-images"
# Get a list of patients.
patients=(${main_path}/SC-HYP-38)

img_folder="img-nii"

for p in ${patients[*]};
do

# Print the current patient.
 
  patient_id=$(basename ${p})
  
  echo ${p}

  # assert whether nii image exists
  if ! [ -d ${p}/${img_folder} ] || ! [ "$(ls -A  ${p}/${img_folder})" ];then
    echo "no image"
    continue
  fi

  # set output folder
  o_dir=${p}/img-nii-resampled
  # echo ${o_dir}
  mkdir -p ${o_dir}

  IMGS=(${p}/${img_folder}/4CH_2.nii.gz)

  for i in $(seq 0 $(( ${#IMGS[*]} - 1 )));
  do
  #echo ${IMGS[${i}]}
    i_file=${IMGS[${i}]}
    echo ${i_file}
    o_file=${o_dir}/$(basename ${i_file})

    if [ -f ${o_file} ];then
      echo "already done this file"
      continue
    else
      c3d ${i_file} -interpolation Cubic -resample-mm 1.3672x1.3672x7.0mm -o ${o_file}
    fi   
  done
done


