#!/bin/bash

cd $HOME/brassica/ || exit

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export OMP_NUM_THREADS=4

##### INFERENCE #####

# export IMAGE_DIR=${HOME}/brassica/br9_organised/images
# export TEST_ANNOT=${HOME}/brassica/br9_organised/br9_organised_fixed.json

# export IMAGE_DIR=${HOME}/brassica/BR9_disorganised/images
# export TEST_ANNOT=${HOME}/brassica/BR9_disorganised/br9.json

# export IMAGE_DIR=$HOME/br017/images
# export TEST_ANNOT=$HOME/br017/br017.json

# export IMAGE_DIR=${HOME}/br017/images
# export TEST_ANNOT=${HOME}/br017/br017_annotated.json

# export IMAGE_DIR=${HOME}/brassica/Images_avg_score/BR11_disorganised/images
# export TEST_ANNOT=${HOME}/brassica/Images_avg_score/BR11_disorganised/ds_out.json

export OUT_DIR="${HOME}/brassica_out/"
export WEIGHTS="${HOME}/brassica/model 4/out/model.pth"

python -m torch.distributed.launch --use-env test.py "${IMAGE_DIR}" "${TEST_ANNOT}" "${OUT_DIR}" "${WEIGHTS}"