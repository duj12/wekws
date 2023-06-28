#!/bin/bash
# Copyright 2021  Binbin Zhang(binbzha@qq.com)
#           2023  Jing Du(thuduj12@163.com)

. ./path.sh

stage=$1
stop_stage=$2
num_keywords=2599
keyword="你 好 小 镜"

config=conf/ds_tcn_ctc_aug.yaml
norm_mean=true
norm_var=true
gpus="4"
data=data_ctc
checkpoint=
dir=exp/ds_tcn_ctc_aug
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

#Whether to train base model. If set true, must put train+dev data in trainbase_dir
trainbase=false
trainbase_dir=$data/base
trainbase_config=conf/ds_tcn_ctc_base.yaml
trainbase_exp=exp/base

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
# Here we Use Paraformer Large(https://www.modelscope.cn/models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary)
# to transcribe the negative wavs, and upload the transcription to modelscope.
  git clone https://www.modelscope.cn/datasets/thuduj12/mobvoi_kws_transcription.git
  for folder in train dev test; do
    if [ -f $data/$folder/text ];then
      mv $data/$folder/text $data/$folder/text.label
      cat  $data/$folder/text.label | sed "s/ 0/ $keyword/g" | sed "s/ -1/ /g" >  $data/$folder/text
    fi
  done

  # and we also copy the tokens and lexicon that used in
  # https://modelscope.cn/models/damo/speech_charctc_kws_phone-xiaoyun/summary
  cp mobvoi_kws_transcription/tokens.txt $data/tokens.txt
  cp mobvoi_kws_transcription/lexicon.txt $data/lexicon.txt

fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Compute CMVN and Format datasets"
  tools/compute_cmvn_stats.py --num_workers 16 --train_config $config \
    --in_scp $data/train/wav.scp \
    --out_cmvn $data/train/global_cmvn

  for x in train dev test; do
    tools/wav_to_duration.sh --nj 8 $data/$x/wav.scp $data/$x/wav.dur

    # Here we use tokens.txt and lexicon.txt to convert txt into index
    tools/make_list.py $data/$x/wav.scp $data/$x/text \
      $data/$x/wav.dur $data/$x/data.list  \
      --token_file $data/tokens.txt \
      --lexicon_file $data/lexicon.txt
  done
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ $trainbase == true ]; then
  for x in train dev ; do
    if [ ! -f $trainbase_dir/$x/wav.scp ] || [ ! -f $trainbase_dir/$x/text ]; then
      echo "If You Want to Train Base KWS-CTC Model, You Should Prepare ASR Data by Yourself."
      echo "The wav.scp and text in KALDI-format is Needed, You Should Put Them in $trainbase_dir/$x"
      exit
    fi
    if [ ! -f $trainbase_dir/$x/wav.dur ]; then
      tools/wav_to_duration.sh --nj 128 $trainbase_dir/$x/wav.scp $trainbase_dir/$x/wav.dur
    fi

    # Here we use tokens.txt and lexicon.txt to convert txt into index
    if [ ! -f $trainbase_dir/$x/data.list ]; then
      tools/make_list.py $trainbase_dir/$x/wav.scp $trainbase_dir/$x/text \
          $trainbase_dir/$x/wav.dur $trainbase_dir/$x/data.list  \
          --token_file $data/tokens.txt \
          --lexicon_file $data/lexicon.txt
    fi
  done

  echo "Start base training ..."
  mkdir -p $trainbase_exp
  cmvn_opts=
  $norm_mean && cmvn_opts="--cmvn_file $data/train/global_cmvn"
  $norm_var && cmvn_opts="$cmvn_opts --norm_var"
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wekws/bin/train.py --gpus $gpus \
      --config $trainbase_config \
      --train_data $trainbase_dir/train/data.list \
      --cv_data $trainbase_dir/dev/data.list \
      --model_dir $trainbase_exp \
      --num_workers 2 \
      --ddp.dist_backend nccl \
      --num_keywords $num_keywords \
      --min_duration 50 \
      --seed 666 \
      $cmvn_opts # \
      #--checkpoint $trainbase_exp/23.pt
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Start training ..."
  mkdir -p $dir
  cmvn_opts=
  $norm_mean && cmvn_opts="--cmvn_file $data/train/global_cmvn"
  $norm_var && cmvn_opts="$cmvn_opts --norm_var"
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')

  if $trainbase; then
    echo "Use the base model you trained as checkpoint: $trainbase_exp/final.pt"
    checkpoint=$trainbase_exp/final.pt
  else
    echo "Use the base model trained with WenetSpeech as checkpoint: mobvoi_kws_transcription/23.pt"
    if [ ! -d mobvoi_kws_transcription ] ;then
      git clone https://www.modelscope.cn/datasets/thuduj12/mobvoi_kws_transcription.git
    fi
    checkpoint=mobvoi_kws_transcription/23.pt    # this ckpt may not converge well.
  fi
  #checkpoint=$dir/43.pt
  #torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
  LOCAL_RANK=0  WORLD_SIZE=1 python  wekws/bin/train.py --gpus $gpus \
      --config $config \
      --train_data $data/train/data.list \
      --cv_data $data/dev/data.list \
      --reverb_lmdb  $rir_lmdb  \
      --noise_lmdb  $noise_lmdb  \
      --model_dir $dir \
      --num_workers 8 \
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
  stream=true  # we detect keyword online with ctc_prefix_beam_search
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