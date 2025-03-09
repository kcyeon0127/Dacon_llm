import sys
sys.path.append('/home/elicer/DaconAcc/')

from utils import (
    get_lora_config,
    get_train_config,
    get_dataset,
    get_valid_input_dataset,
    get_dataloader_kwargs,
    train
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    PeftModel,
    prepare_model_for_kbit_training,
    get_peft_model,
)
import fire
import torch
import os
from torch.optim.lr_scheduler import StepLR
from peft import get_peft_model
import torch.optim as optim


def main(
    **kwargs,
    ) -> None:

    train_config = get_train_config(**kwargs)
    folder_path = train_config.output_dir
    print(f"save path: {folder_path}")
    os.makedirs(folder_path, exist_ok=True)
    setattr(train_config, "output_dir", folder_path)
    

    model_id = train_config.model_name
    print(model_id)
    torch_dtype = torch.float16

    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True, ###
        bnb_4bit_compute_dtype=torch_dtype, ###
        bnb_4bit_use_double_quant=False,
    )
    
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant_config, device_map="auto")
    
    # model = AutoModelForCausalLM.from_pretrained(lora_path, quantization_config=quant_config, device_map="auto")
    
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)    
    
    lora_path = "/home/elicer/DaconAcc/finetuned_juungwon_WO_Quant"
    model = PeftModel.from_pretrained(model, lora_path) ###

    for name, param in model.named_parameters():
        if "lora" in name:
            param.requires_grad = True

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # lora_config = get_lora_config(**kwargs)
    # model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    train_dataset = get_dataset(
        tokenizer=tokenizer,
        split="train"
    )
    
    valid_dataset = get_dataset(
        tokenizer=tokenizer,
        split="valid"
    )

    train_dl_kwargs = get_dataloader_kwargs(train_config, train_dataset, tokenizer, "train")
    train_dataloader = torch.utils.data.DataLoader(train_dataset,pin_memory=True,**train_dl_kwargs,)

    val_dl_kwargs = get_dataloader_kwargs(train_config, valid_dataset, tokenizer, "val")
    eval_dataloader = torch.utils.data.DataLoader(valid_dataset,pin_memory=True,**val_dl_kwargs,)


    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)


    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
    )
    [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

fire.Fire(main)