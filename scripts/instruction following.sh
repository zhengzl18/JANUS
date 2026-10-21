# Refer to https://github.com/lm-sys/FastChat.git
cd YOUR/PATH/TO/FastChat/fastchat/llm_judge
MODEL_ID="your_model_id"
FT_MODEL="path/to/$MODEL_ID"

python gen_model_answer.py --model-path $FT_MODEL --model-id $MODEL_ID
python gen_judgment.py --model-list $MODEL_ID --parallel 2 --judge-model gpt-4
python show_result.py --model-list $MODEL_ID --judge-model gpt-4