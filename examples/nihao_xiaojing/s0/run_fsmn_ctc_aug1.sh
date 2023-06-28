#!/bin/bash
# Copyright 2021  Binbin Zhang(binbzha@qq.com)
#           2023  Jing Du(thuduj12@163.com)

. ./path.sh

stage=$1
stop_stage=$2
num_keywords=2599
keyword="你 好 小 镜"

config=conf/fsmn_ctc_aug1.yaml
norm_mean=true
norm_var=true
gpus="5"
data=data_ctc
checkpoint=
dir=exp/fsmn_ctc_aug1
average_model=true
num_average=30
if $average_model ;then
  score_checkpoint=$dir/avg_${num_average}.pt
else
  score_checkpoint=$dir/final.pt
fi

#noise and reverb set
noise_scp=$data/noise.scp
noise_lmdb=$data/noise.lmdb
if [ ! -d $noise_lmdb ]; then
  python tools/make_lmdb.py $noise_scp $noise_lmdb
fi
rir_scp=$data/rir.scp
rir_lmdb=$data/rir.lmdb
if [ ! -d $rir_lmdb ]; then
  python tools/make_lmdb.py $rir_scp $rir_lmdb
fi


. tools/parse_options.sh || exit 1;
window_shift=50


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then

  echo "Use the base model from modelscope"
  if [ ! -d speech_charctc_kws_phone-xiaoyun ] ;then
      git lfs install
      git clone https://www.modelscope.cn/damo/speech_charctc_kws_phone-xiaoyun.git
  fi
  checkpoint=speech_charctc_kws_phone-xiaoyun/train/base.pt
  cp speech_charctc_kws_phone-xiaoyun/train/feature_transform.txt.80dim-l2r2 $data/global_cmvn.kaldi

  echo "Start training ..."
  mkdir -p $dir
  cmvn_opts=
  $norm_mean && cmvn_opts="--cmvn_file $data/global_cmvn.kaldi"
  $norm_var && cmvn_opts="$cmvn_opts --norm_var"
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  #checkpoint=$dir/13.pt
  #torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
  LOCAL_RANK=0  WORLD_SIZE=1 python   wekws/bin/train.py --gpus $gpus \
      --config $config \
      --train_data $data/train/data.list \
      --cv_data $data/dev/data.list \
      --reverb_lmdb  $rir_lmdb  \
      --noise_lmdb  $noise_lmdb  \
      --model_dir $dir \
      --num_workers 2 \
      --num_keywords $num_keywords \
      --min_duration 50 \
      --seed 666 \
      $cmvn_opts \
      ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Do model average, Compute FRR/FAR ..."
  if $average_model; then
    python wekws/bin/average_model.py \
      --dst_model $score_checkpoint \
      --src_path $dir  \
      --num ${num_average} \
      --val_best
  fi
  result_dir=$dir/test_$(basename $score_checkpoint)
  mkdir -p $result_dir
  stream=true   # we detect keyword online with ctc_prefix_beam_search
  score_prefix=""
  if $stream ; then
    score_prefix=stream_
  fi
  python wekws/bin/${score_prefix}score_ctc.py \
    --config $dir/config.yaml \
    --test_data $data/test/data.list \
    --gpu 0  \
    --batch_size 256 \
    --checkpoint $score_checkpoint \
    --score_file $result_dir/score.txt  \
    --num_workers 8  \
    --keywords 你好小镜 \
    --token_file $data/tokens.txt \
    --lexicon_file $data/lexicon.txt

  python wekws/bin/compute_det_ctc.py \
      --keywords 你好小镜 \
      --test_data $data/test/data.list \
      --window_shift $window_shift \
      --step 0.001  \
      --score_file $result_dir/score.txt \
      --token_file $data/tokens.txt \
      --lexicon_file $data/lexicon.txt
fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  jit_model=$(basename $score_checkpoint | sed -e 's:.pt$:.zip:g')
  onnx_model=$(basename $score_checkpoint | sed -e 's:.pt$:.onnx:g')
  python wekws/bin/export_jit.py \
    --config $dir/config.yaml \
    --checkpoint $score_checkpoint \
    --jit_model $dir/$jit_model
  python wekws/bin/export_onnx.py \
    --config $dir/config.yaml \
    --checkpoint $score_checkpoint \
    --onnx_model $dir/$onnx_model
fi
