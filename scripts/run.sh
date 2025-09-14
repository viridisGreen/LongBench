clear

# export CUDA_VISIBLE_DEVICES=0
# python pred.py \
#     --model llama2-7b-chat-4k \
#     --cuda_visible_devices 0

export CUDA_VISIBLE_DEVICES=1
python pred.py \
    --model qwen3-8b \
    --cuda_visible_devices 1

# export CUDA_VISIBLE_DEVICES=2
# python pred.py \
#     --model llama3-8b-instruct \
#     --cuda_visible_devices 2

# export CUDA_VISIBLE_DEVICES=3
# python pred.py \
#     --model llama2-7b \
#     --cuda_visible_devices 3

