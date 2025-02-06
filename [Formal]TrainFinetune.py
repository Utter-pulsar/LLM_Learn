# Experimental environment: A10, 3090, V100, ...
# 20GB GPU memory
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch

from swift.llm import (
    DatasetName, InferArguments, ModelType, SftArguments,
    infer_main, sft_main, app_ui_main
)
#dataset=[f'{DatasetName.self_cognition}#500'],
if __name__ == '__main__':
    model_type = ModelType.qwen1half_0_5b_chat
    sft_args = SftArguments(
        model_type=model_type,
        dataset=[f'{"./FineTuneDataset/finetunedata.jsonl"}'],
        logging_steps=5,
        max_length=2048,
        max_steps = 200,
        learning_rate=1e-4,
        output_dir='output',
        lora_target_modules=['ALL'],
        model_name=['小电', 'Xiao Dian'],
        model_author=['上海交通大学', 'Shanghai-Jiaotong University'],
        dtype="AUTO",
      )
    # print("-"*100)
    result = sft_main(sft_args)
    best_model_checkpoint = result['best_model_checkpoint']
    print(f'best_model_checkpoint: {best_model_checkpoint}')
    torch.cuda.empty_cache()