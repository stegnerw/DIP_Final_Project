#!/bin/bash

# Directory definitions
BASE_DIR=$(realpath $(dirname "$0"))
MODELS_DIR="$BASE_DIR/Models"
TB_DIR_NAME="TensorBoard"
CSV_OUT_DIR="$BASE_DIR/Training_Logs"

# Misc constants
HOST="localhost"
PORT="8080"

# Create necessary directories
if [ -d $CSV_OUT_DIR ]; then
  echo "WARNING: Directory already exists:"
  echo $CSV_OUT_DIR
else
  mkdir $CSV_OUT_DIR
fi

# Iterate through models directory
#for model_dir in $(ls -d $MODELS_DIR/*); do
  #tb_dir="$model_dir/$TB_DIR_NAME"
  #if [ ! -d $tb_dir ]; then
    #echo "ERROR: No TensorBoard directory found:"
    #echo $tb_dir
    #continue
  #fi
  #echo "Found TensorBoard directory:"
  #echo $tb_dir
#done

# Testing with just one directory
echo
echo "* * *Testing stuff * * *"
echo
model_dir="$MODELS_DIR/Dense_0x000"
tb_dir="$model_dir/$TB_DIR_NAME"
echo $tb_dir
if [ -d $tb_dir ]; then
  echo "Found directory"
else
  echo "!!!!!Could not find directory"
fi

tensorboard --logdir $tb_dir --port 8080 > /dev/null &
tb_pid=$!
echo "TensorBoard PID: $tb_pid"

# Wait for server to fire up
# Probably only needs to be about 1.5 seconds but I'm paranoid
sleep 2

echo "Pulling CSV from server"
model_name="$(basename $model_dir)"
train_csv="$CSV_OUT_DIR/${model_name}_train.csv"
test_csv="$CSV_OUT_DIR/${model_name}_test.csv"

#curl --silent --output $train_csv "\"${HOST}:${PORT}/data/plugin/scalars/scalars?tag=epoch_accuracy&run=TensorBoard%2Ftrain&format=csv"
curl 'localhost:8080/data/plugin/scalars/scalars?tag=epoch_accuracy&run=TensorBoard%2Ftrain&format=csv'

echo "Closing TensorBoard server"
kill $tb_pid

exit 0

