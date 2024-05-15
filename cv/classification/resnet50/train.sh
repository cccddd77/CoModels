export PYTHONPATH=$PWD:$PYTHONPATH
set -aux

GPU_NUMS=8
NODE_NUMS=2
PORT=12346
MODEL_ARCH="resnet50"

python3 -m oneflow.distributed.launch \
        --nproc_per_node $GPU_NUMS \
        --nnodes $NODE_NUMS \
        --master_addr 192.168.1.27 \
        --master_port $PORT \
        main.py \
        --cfg configs/default_settings.yaml \
        --model_arch $MODEL_ARCH 

