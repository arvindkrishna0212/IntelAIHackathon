# IntelAIHackathon

# Resume Ranking System

This repository contains code for a website (made using streamlit) that allows users to upload their resumes. The resumes are ranked automatically using a fine tuned model of gemma.

## Google Colab Notebook
[Click here](https://colab.research.google.com/drive/16PhoBeTMnbbOzkBd8JkHxTTog-h0-2D_?usp=sharing)

Make sure to upload app.py to the colab file.

## Prerequisites
Before running the application, ensure you have the following packages installed:

- [Streamlit](https://streamlit.io/)
- [PDFMiner.six](https://github.com/pdfminer/pdfminer.six)
- [SpaCy](https://spacy.io/)
- [Groq](https://groq.io/)
- [Urllib](https://docs.python.org/3/library/urllib.html)
- [LocalTunnel](https://localtunnel.github.io/www/)

Additionally, if running the Streamlit app on Google Colab, it will prompt for a password provided by the LocalTunnel.

## Fine-tuning the Model
To fine-tune a model for ranking resumes, execute the following code snippet:

```python
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
```


## Usage
1) Upload resumes in PDF or DOCX format using the file uploader.
2) Click on the "Submit" button to analyze the resumes.
3) View extracted information such as name, education, and skills.
4) Receive ranked results based on predefined criteria.

# Explanation

## Code Structure
The code is organized into several functions, each serving a specific purpose:

### 1. `extract_skills_from_resume(text, skills_list)`
This function takes the text of a resume and a list of skills as input. It extracts skills from the resume text based on the provided list of skills.

### 2. `extract_education_from_resume(text)`
This function extracts education information from the resume text by searching for keywords related to education, such as "Bsc", "Msc", etc.

### 3. `extract_name(resume_text)`
This function extracts the name of the candidate from the resume text using SpaCy's matcher.

### 4. `extract_skills(text)`
This function extracts technical skills from the resume text using SpaCy's named entity recognition (NER) capabilities.

### 5. `LLM(query)`
This function interacts with the Groq API to perform language model-based completions. It sends a query and receives a response from the API.

### 6. `check_resume_keywords(resume_text, keywords)`
This function checks if the skills extracted from the resume match the required keywords using the LLM function and Groq API.

### 7. `get_recent_file_path(directory)`
This function retrieves the path of the most recently uploaded resume from a specified directory.

### 8. `main()`
The main function sets up the Streamlit application interface. It allows users to upload resumes, analyze them, and view extracted information.

