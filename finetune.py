import sys
import site
import os
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from transformers import TrainingArguments
warnings.filterwarnings("ignore")

!{sys.executable} -m pip install --upgrade "transformers>=4.38.*"
!{sys.executable} -m pip install --upgrade "datasets>=2.18.*"
!{sys.executable} -m pip install --upgrade "wandb>=0.16.*"
!{sys.executable} -m pip install --upgrade "trl>=0.7.11"
!{sys.executable} -m pip install --upgrade "peft>=0.9.0"
!{sys.executable} -m pip install --upgrade "accelerate>=0.28.*"

site_packages_dir = site.getsitepackages()[0]

# add the site pkg directory where these pkgs are insalled to the top of sys.path
if not os.access(site_packages_dir, os.W_OK):
    user_site_packages_dir = site.getusersitepackages()
    if user_site_packages_dir in sys.path:
        sys.path.remove(user_site_packages_dir)
    sys.path.insert(0, user_site_packages_dir)
else:
    if site_packages_dir in sys.path:
        sys.path.remove(site_packages_dir)
    sys.path.insert(0, site_packages_dir)

from peft import LoraConfig

lora_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "o_proj",
                    "v_proj", "k_proj",
                    "gate_proj", "up_proj",
                    "down_proj"],
    task_type="CAUSAL_LM")

from huggingface_hub import notebook_login
notebook_login()

USE_CPU = True
device = "cpu" if USE_CPU else "cuda"
print(f"using device: {device}")

model_id = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
model.config.use_cache = False
model.config.pretraining_tp = 1

from datasets import load_dataset

resume_dataset = load_dataset("InferencePrince555/Resume-Dataset")
resume_dataset = resume_dataset.map(lambda x: {"text": x["Resume_test"]})

text_field = "text"
max_seq_length = 1024

num_train_samples = len(resume_dataset)
batch_size = 2
gradient_accumulation_steps = 8
steps_per_epoch = num_train_samples // (batch_size * gradient_accumulation_steps)
num_epochs = 5
max_steps = steps_per_epoch * num_epochs
print(f"Finetuning for max number of steps: {max_steps}")

finetuned_model_id = "my-gemma-2b-resume-finetuned"
tuned_lora_model = "my-gemma-2b-resume-lora"
PUSH_TO_HUB = True
USE_WANDB = True

training_args = transformers.TrainingArguments(
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_ratio=0.05,
    max_steps=max_steps,
    learning_rate=1e-5,
    evaluation_strategy="steps",
    save_steps=500,
    bf16=True,
    logging_steps=100,
    output_dir=finetuned_model_id,
    hub_model_id=finetuned_model_id if PUSH_TO_HUB else None,
    report_to="wandb" if USE_WANDB else None,
    push_to_hub=PUSH_TO_HUB,
    max_grad_norm=0.6,
    weight_decay=0.01,
    group_by_length=True
)

trainer = SFTTrainer(
    model=model,
    train_dataset=resume_dataset,
    eval_dataset=None,
    tokenizer=tokenizer,
    args=training_args,
    peft_config=lora_config,
    dataset_text_field=text_field,
    max_seq_length=max_seq_length,
    packing=True
)

results = trainer.train()

def print_training_summary(results):
    print(f"Time: {results.metrics['train_runtime']: .2f}")
    print(f"Samples/second: {results.metrics['train_samples_per_second']: .2f}")

print_training_summary(results)

wandb.finish()

trainer.model.save_pretrained(tuned_lora_model)

from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.bfloat16,
)

model = PeftModel.from_pretrained(base_model, tuned_lora_model)
model = model.merge_and_unload()

model.save_pretrained(finetuned_model_id)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = model.to(device)

sample_input = "This is a sample resume text."
inputs = tokenizer(sample_input, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=200,
                         do_sample=False, top_k=100, temperature=0.1,
                         eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if PUSH_TO_HUB:
    trainer.push_to_hub()