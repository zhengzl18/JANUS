# Refer to https://github.com/bigcode-project/bigcode-evaluation-harness.git
cd YOUR/PATH/TO/bigcode-evaluation-harness
FT_MODEL="path_to_your_finetuned_model"
OUTPUT_PATH="path_to_save_generations"


accelerate launch --main_process_port=29500 main.py \
    --model $FT_MODEL \
    --tasks mbpp \
    --temperature 0.1 \
    --n_samples 15 \
    --batch_size 15 \
    --max_memory_per_gpu auto \
    --allow_code_execution \
    --trust_remote_code \
    --precision bf16 \
    --metric_output_path "$OUTPUT_PATH/MBPP_results.json" \
    --save_generations_path "$OUTPUT_PATH/generations_path.json" \
    --save_generations \
    --generation_only

accelerate launch main.py \
  --model $FT_MODEL \
  --tasks mbpp \
  --temperature 0.1 \
  --n_samples 15 \
  --batch_size 15 \
  --max_memory_per_gpu auto \
  --allow_code_execution \
  --trust_remote_code \
  --precision bf16 \
  --metric_output_path "$OUTPUT_PATH/mbpp_results.json" \
  --load_generations_path "$OUTPUT_PATH/generations_path_mbpp.json"