METHOD=$1
MODEL_NAME_OR_PATH=$2
TASK=$3
TRAIN_EXAMPLES=$4
PATTERN_ID=$5
DEVICE=$6

DATA_ROOT='data/'
TRAIN_BATCH_SIZE=8
EVAL_BATCH_SIZE=128
UNLABELED_BATCH_SIZE=8
ACCU=1
MAX_STEPS=2500
LOGGING_STEPS=50

if [ $TASK = "cola" ]; then
  DATA_DIR=${DATA_ROOT}cola
  SEQ_LENGTH=64
  PATTERN_ID=0
elif [ $TASK = "mnli" ] || [ $TASK = "mnli-mm" ]; then
  DATA_DIR=${DATA_ROOT}mnli
  SEQ_LENGTH=256
  LOGGING_STEPS=100
  PATTERN_ID=3
elif [ $TASK = "mrpc" ]; then
  DATA_DIR=${DATA_ROOT}mrpc
  SEQ_LENGTH=128
  PATTERN_ID=2
elif [ $TASK = "sst-2" ]; then
  DATA_DIR=${DATA_ROOT}sst-2
  SEQ_LENGTH=64
  PATTERN_ID=1
elif [ $TASK = "sts-b" ]; then
  DATA_DIR=${DATA_ROOT}sts-b
  SEQ_LENGTH=64
elif [ $TASK = "qqp" ]; then
  DATA_DIR=${DATA_ROOT}qqp
  SEQ_LENGTH=256
  LOGGING_STEPS=100
  PATTERN_ID=1
elif [ $TASK = "qnli" ]; then
  DATA_DIR=${DATA_ROOT}qnli
  SEQ_LENGTH=128
  PATTERN_ID=2
elif [ $TASK = "rte" ]; then
  DATA_DIR=${DATA_ROOT}rte
  SEQ_LENGTH=128
  PATTERN_ID=3
elif [ $TASK = "wnli" ]; then
  DATA_DIR=${DATA_ROOT}wnli
  SEQ_LENGTH=64
fi


CUDA_VISIBLE_DEVICES=$DEVICE python3 fire.py \
--method $METHOD \
--data_dir $DATA_DIR \
--model_name_or_path $MODEL_NAME_OR_PATH \
--task_name $TASK \
--max_length $SEQ_LENGTH \
--per_gpu_train_batch_size $TRAIN_BATCH_SIZE \
--per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
--per_gpu_unlabeled_batch_size $UNLABELED_BATCH_SIZE \
--gradient_accumulation_steps $ACCU \
--max_steps $MAX_STEPS \
--train_examples $TRAIN_EXAMPLES \
--pattern_id $PATTERN_ID \
--logging_steps $LOGGING_STEPS