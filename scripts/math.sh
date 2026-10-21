# Refer to https://github.com/GraphPKU/PiSSA.git and https://huggingface.co/datasets/fxmeng/pissa-dataset
cd YOUR/PATH/TO/PiSSA
FT_MODEL="path_to_your_finetuned_model"
OUTPUT_PATH="path_to_save_responses"

python utils/gen_vllm.py --model $FT_MODEL --sub_task metamath --output_file $OUTPUT_PATH/metamath_response.jsonl --tokenizer_mode auto
python utils/test_acc.py --input_file $OUTPUT_PATH/metamath_response.jsonl
