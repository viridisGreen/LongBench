clear


# export CUDA_VISIBLE_DEVICES=3
# python pred.py \
#     --model llama2-7b \
#     --cuda_visible_devices 3 \
#     --skip_layer_id 17 19 21 23 \
#     --quantize_type none


# export CUDA_VISIBLE_DEVICES=5
# python pred.py \
#     --model llama2-7b-chat-4k \
#     --cuda_visible_devices 5 \
#     --skip_layer_id 21 23 \
#     --quantize_type none


# export CUDA_VISIBLE_DEVICES=3
# python pred.py \
#     --model llama3-8b-instruct \
#     --cuda_visible_devices 3 \
#     --skip_layer_id 21 22 23 24 \
#     --quantize_type none


export CUDA_VISIBLE_DEVICES=4
python pred.py \
    --model qwen3-8b \
    --cuda_visible_devices 4 \
    --sparse_rate 0.01 \
    --sparse_ratio 0.7
