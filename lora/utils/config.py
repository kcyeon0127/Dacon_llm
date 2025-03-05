from dataclasses import(
    dataclass,
    field,
    asdict,
)
from typing import List

from peft import LoraConfig


@dataclass
class Lora_config:
    r:int = 32
    lora_alpha:int = 32
    target_modules:List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "out_proj"]) # "q_proj", "k_proj", "v_proj", "out_proj"
    bias = "none"
    task_type:str = "CAUSAL_LM"
    lora_dropout:float = 0.05
    inference_mode:bool = False


@dataclass
class Train_config:
    model_name:str = "juungwon/Llama-3-instruction-constructionsafety"
    # model_name:str = "DBCMLAB/Llama-3-instruction-constructionsafety-layertuning"
    batch_size_training:int = 8
    gradient_accumulation_steps:int = 4
    gradient_clipping:bool = True
    gradient_clipping_threshold:float = 1.0
    num_epochs:int = 50
    lr:float = 2e-4
    weight_decay:float = 0.0
    gamma:float = 0.85
    seed:int = 42
    val_batch_size:int = 8
    output_dir:str = "finetuned_juungwon_WO_Quant"

def update_config(config,**kwargs) -> None:
    """
    This function changes the values of config according to kwargs.
    """
    for k,v in kwargs.items():
        if hasattr(config,k):
            setattr(config, k, v)
    return

def get_lora_config(**kwargs) -> Lora_config:
    """
    This function returns LoraConfig of peft library which has default values of Lora_config dataclass.
    You can change the value, by inputting kwargs like this: lora_dropout=0.1
    """
    config = Lora_config()
    update_config(config, **kwargs)
    lora_config_params = asdict(config)
    return LoraConfig(**lora_config_params)

def get_train_config(**kwargs):
    """
    This function returns object of Train_config class.
    You can change Train_config's attributes by inputting kwargs.
    """
    config = Train_config()
    update_config(config,**kwargs)
    return config
