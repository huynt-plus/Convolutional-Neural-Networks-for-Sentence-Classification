#!/usr/bin/env bash

#PBS -q SINGLE
#PBS -o /home/s1620007/cnn-sentence/error.out
#PBS -e /home/s1620007/cnn-sentence/error.in
#PBS -N train-cnn
#PBS -j oe

cd /home/s1620007/cnn-sentence

setenv PATH ${PBS_O_PATH}

root="$PWD"
logdir="$root/cnn_result"
mkdir -pv $logdir

THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python conv_net_sentence.py -static -word2vec \
--log-path $logdir