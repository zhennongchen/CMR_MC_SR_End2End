#/bin/bash
# 
# change the following variables to set up code, input and output dirs
#HOST_WORK_DIR="/home/local/PARTNERS/dw640/docker_example"  # change this to your code directory
#HOST_DATA_DIR="/home/local/PARTNERS/dw640/mnt/women_health_internal/DBT"  # change this to your data directory
HOST_WORK_DIR="/home/local/PARTNERS/zc13/Documents"
HOST_DATA_DIR="/home/local/PARTNERS/zc13/mnt"


# get host user id to populate to the container
HOST_USER_ID="$(id -u)"
HOST_GROUP_ID="$(id -g)"
HOST_USER_NAME=${USER}
CONTAINER_WORKDIR="/workspace/Documents"   #/workspace/CTProjector" 
CONTAINER_DATADIR="/mnt" #should be /mnt/data when using NAS drive   

sudo docker run -it --gpus=all --name=docker_ex --network="bridge" --shm-size=256m --ipc=host \
-v ${HOST_WORK_DIR}:${CONTAINER_WORKDIR} \
-v ${HOST_DATA_DIR}:${CONTAINER_DATADIR} \
-p 8110:8110 \
-e CONTAINER_UID=${HOST_USER_ID} \
-e CONTAINER_GID=${HOST_GROUP_ID} \
-e CONTAINER_UNAME=${HOST_USER_NAME} \
-e CONTAINER_WORKDIR=${CONTAINER_WORKDIR} \
zc_sam:1.0