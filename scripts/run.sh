clear


# export CUDA_VISIBLE_DEVICES=3
# python pred.py \
#     --model llama2-7b \
#     --cuda_visible_devices 3
# python eval.py --model llama2-7b


# export CUDA_VISIBLE_DEVICES=0
# python pred.py \
#     --model llama2-7b-chat-4k \
#     --cuda_visible_devices 0
# python eval.py --model llama2-7b-chat-4k


export CUDA_VISIBLE_DEVICES=2
python pred.py \
    --model llama3-8b-instruct \
    --cuda_visible_devices 2 \
    --offload_layer_id 16 17 18 19 20 21 22 23 \
    --skip_layer_id 17 19 21 23 \
    --quantize_type none
# python eval.py --model llama3-8b-instruct


# export CUDA_VISIBLE_DEVICES=1
# python pred.py \
#     --model qwen3-8b \
#     --cuda_visible_devices 1
# python eval.py --model qwen3-8b

