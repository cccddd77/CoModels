export PYTHONPATH=$PWD:$PYTHONPATH
set -aux

GPU_NUMS=4
NODE_NUMS=2
PORT=12377
MODEL_ARCH="resnest50"
IMAGE_SIZE=224

python3 -m oneflow.distributed.launch \
        --nproc_per_node $GPU_NUMS \
        --nnodes $NODE_NUMS \
        --master_addr 192.168.1.27 \
        --master_port $PORT \
        main.py \
        --cfg configs/resnest_default_settings.yaml \
        --model_arch $MODEL_ARCH \
        --image-size $IMAGE_SIZE
