export PYTHONPATH=$PWD:$PYTHONPATH
set -aux

GPU_NUMS=4
PORT=12346
MODEL_ARCH="fan_tiny_12_p16_224"
BATCH_SIZE=32
LEARNING_RATE=1e-4

export CUDA_VISIBLE_DEVICES=0,5,6,7
python3 -m oneflow.distributed.launch \
        --nproc_per_node $GPU_NUMS \
        --master_addr 127.0.0.1 \
        --master_port $PORT \
        main.py \
        --cfg configs/default_settings.yaml \
        --model_arch $MODEL_ARCH \
        --batch-size $BATCH_SIZE \
        --lr $LEARNING_RATE  

