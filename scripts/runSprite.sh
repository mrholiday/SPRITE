#!/bin/sh
#$ -cwd
#$ -l mem_free=8g
#$ -l h_rt=72:00:00
#$ -l h_vmem=16g
#$ -l num_proc=11
#$ -N sprite_factored
#$ -S /bin/bash

# For training a SPRITE model on the grid.  May have to play around with
# the arguments a bit.

MODEL_NAME=$1
IN_PATH=$2
OUT_DIR=$3
Z=$4
STEP=$5
SIGMA_ALPHA=$6

NUM_THREADS=10

cd ../sprite/src/

echo "java -cp ../lib/jcommander-1.48-SNAPSHOT.jar:. -Xmx6144M -XX:+UseSerialGC models/factored/impl/${MODEL_NAME} -input ${IN_PATH} -Z ${Z} -omegaB -4.0 -step ${STEP} -iters 5000 -nthreads ${NUM_THREADS} -sigmaAlpha ${SIGMA_ALPHA} -outDir ${OUT_DIR} -logPath ${OUT_DIR}/${MODEL_NAME}_${Z}_${STEP}_${SIGMA_ALPHA}.log"
java -cp ../lib/jcommander-1.48-SNAPSHOT.jar:. -Xmx6144M -XX:+UseSerialGC models/factored/impl/${MODEL_NAME} -input ${IN_PATH} -Z ${Z} -omegaB -4.0 -step ${STEP} -iters 5000 -nthreads ${NUM_THREADS} -sigmaAlpha ${SIGMA_ALPHA} -outDir ${OUT_DIR} -logPath ${OUT_DIR}/${MODEL_NAME}_${Z}_${STEP}_${SIGMA_ALPHA}.log
