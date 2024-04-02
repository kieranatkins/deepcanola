#!/bin/bash

cd $HOME/brassica/ || exit

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
export OMP_NUM_THREADS=4

##### Model 1 #####

# export TRAIN_IMAGE_DIR="${HOME}/brassica/model 1/images/"
# export TRAIN_ANNOT="${HOME}/brassica/model 1/model_1_train.json"
# export TEST_IMAGE_DIR="${HOME}/brassica/model 1/images/"
# export TEST_ANNOT="${HOME}/brassica/model 1/model_1_test.json"
# export OUT_DIR="${HOME}/brassica/model 1/out/"
# export WEIGHTS="${OUT_DIR}/model.pth"

# python -m torch.distributed.launch --use-env train.py "${TRAIN_IMAGE_DIR}" "${TRAIN_ANNOT}" "${TEST_IMAGE_DIR}" "${TEST_ANNOT}" "${OUT_DIR}" --batch-size=2 --schedule=1 || exit

##### Model 2 #####

# export TRAIN_IMAGE_DIR="${HOME}/brassica/model 2/images/"
# export TRAIN_ANNOT="${HOME}/brassica/model 2/model_2_train.json"
# export TEST_IMAGE_DIR="${HOME}/brassica/model 2/images/"
# export TEST_ANNOT="${HOME}/brassica/model 2/model_2_test.json"
# export OUT_DIR="${HOME}/brassica/model 2/out/"
# export WEIGHTS="${OUT_DIR}/model.pth"
# export LOAD_FILE="/home/kieran/brassica/model 1/out/model.pth"

# python -m torch.distributed.launch --use-env train.py "${TRAIN_IMAGE_DIR}" "${TRAIN_ANNOT}" "${TEST_IMAGE_DIR}" "${TEST_ANNOT}" "${OUT_DIR}" --load-file "${LOAD_FILE}" --batch-size=2 --schedule=1 || exit

##### Model 3 #####

# export TRAIN_IMAGE_DIR="${HOME}/brassica/model 3/images/"
# export TRAIN_ANNOT="${HOME}/brassica/model 3/model_3_train.json"
# export TEST_IMAGE_DIR="${HOME}/brassica/model 3/images/"
# export TEST_ANNOT="${HOME}/brassica/model 3/model_3_test.json"
# export OUT_DIR="${HOME}/brassica/model 3/out/"
# export WEIGHTS="${OUT_DIR}/model.pth"

# python -m torch.distributed.launch --use-env train.py "${TRAIN_IMAGE_DIR}" "${TRAIN_ANNOT}" "${TEST_IMAGE_DIR}" "${TEST_ANNOT}" "${OUT_DIR}" --batch-size=2 --schedule=1 || exit

##### Model 4 #####

# export TRAIN_IMAGE_DIR="${HOME}/brassica/model 4/images/"
# export TRAIN_ANNOT="${HOME}/brassica/model 4/model_4_train.json"
# export TEST_IMAGE_DIR="${HOME}/brassica/model 4/images/"
# export TEST_ANNOT="${HOME}/brassica/model 4/model_4_test.json"
# export OUT_DIR="${HOME}/brassica/model 4/out/"
# export WEIGHTS="${OUT_DIR}/model.pth"

# python -m torch.distributed.launch --use-env train.py "${TRAIN_IMAGE_DIR}" "${TRAIN_ANNOT}" "${TEST_IMAGE_DIR}" "${TEST_ANNOT}" "${OUT_DIR}" --batch-size=2 --schedule=1 || exit
