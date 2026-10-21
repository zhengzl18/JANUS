# Refer to https://github.com/EleutherAI/lm-evaluation-harness
FT_MODEL="path_to_your_finetuned_model"
OUTPUT_PATH="path_to_save_evaluation_results"

accelerate launch -m lm_eval --model hf \
    --model_args "pretrained=$FT_MODEL" \
    --tasks triviaqa,nq_open,webqs \
    --batch_size 128 \
    --output_path $OUTPUT_PATH \