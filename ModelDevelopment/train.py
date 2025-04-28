from datasets import load_dataset
import torch,os
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments, EarlyStoppingCallback
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from utils import find_all_linear_names, print_trainable_parameters
from accelerate import Accelerator

accelerator = Accelerator()
device_index = Accelerator().process_index

device_map = {"": device_index}



token = '' # your token
output_dir = "/path/to/output/directory"
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

os.environ["WANDB_PROJECT"] = output_dir.split('/')[-1]


train_dataset = load_dataset("csv", data_files=["/path/to/training_datz.csv"], split="train")


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                  torch_dtype=torch.bfloat16, 
                                                  quantization_config=bnb_config,
                                                  device_map=device_map,
                                                  cache_dir="/path/to/cache/huggingface_models/",
                                                  token = token)



base_model.config.use_cache = False
base_model = prepare_model_for_kbit_training(base_model)

tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=find_all_linear_names(base_model),
    #target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

base_model = get_peft_model(base_model, peft_config)
print_trainable_parameters(base_model)
#base_model = accelerator.prepare(base_model)


template = '''### Task:
Your task is to identify and annotate specific medical entities in the given text. For each entity found, you should provide an output in JSON format that includes:
- The start index of the entity in the text.
- The end index of the entity in the text.
- The extracted entity text.
- The entity type.

### Entity Markup Guides:
use "visual" to denote a mention of visual elements.
use "visual_color" to denote mention of color of an object.
use "visual_brightness" to denote mention of brightness.
use "visual_shape" to denote mention of shape of an object.
use "visual_size" to denote mention of size of an object.
use "visual_num_quant" to denote mention of number of an physical object.
use "visual_texture" to denote mention of texture of an physical object.
use "visual_pattern" to denote mention of pattern of an physical object.
use "visual_motion" to denote mention of motions observed.
use "visual_spatial" to denote mention of position of an physical object.
use "somatosensory" to denote the mention of physical sensation.
use "somato_touch" to denote the mention of physical tactile sensation.
use "somato_pain" to denote the mention of pain/discomfort from external sources.
use "somato_temp" to denote the mention of temperature sensation from external sources.
use "somato_pressure" to denote the mention of sensation of pressure.
use "somato_vibration" to denote the mention of sensations of rapid oscillations on skin.
use "somato_proprio" to denote the mention of awareness of body position and movement.
use "somato_kinesthesia" to denote the mention of sensations related to body movements.
use "smell" to denote the mention of olfactory sensations.
use "sound" to denote the mention of perception of hearing.
use "taste" to denote the mention of sensations of gustation.
use "introception" to denote the mention of internal bodily sensations.
use "introception_pulse_hb" to denote the mention of awareness of pulse or heartbeats.
use "introception_breathing" to denote the mention of act/awareness of breathing.
use "introception_hunger" to denote the mention of sensations of hunger.
use "introception_satiety" to denote the mention of sensations of being full after a meal.
use "introception_thirst" to denote the mention of feeling dehydrated.
use "introception_temp" to denote the mention of internal sensations of being hot or cold.
use "introception_pain" to denote the mention of internal pain.
use "introception_energy_level" to denote the mention of sensations related to energy levels or fatigue.
use "introception_tension" to denote the mention of bodily tension.
use "introception_urine_bowels" to denote the mention of sensations indicating the need to relieve oneself.
use "time" to denote the mention of time.
use "time_of_day" to denote the mention the time of day.
use "time_season" to denote the mention of the season.
use "time_month" to denote the mention of the month.
use "time_holiday" to denote the mention of the holiday/ special day.
use "time_day_of_week" to denote the mention of the specific day of the week.
use "location" to denote the mention of the location.
use "action" to denote the mention of an action done.
use "social_interaction" to denote the mention of the interactions between people where the subject is interacting with others.
use "emotion" to denote the mention of the subjects emotions.
use "thought" to denote the mention of the subject's thoughts.
use "thought_memory" to denote the mention of recall of events.
use "thought_judgment" to denote the judgement made by the subject's decision.
use "thought_self_reflection" to denote the self-reflection mentions of the subject.
use "thought_mentalizing" to denote the subject understanding/awareness of others emotions.
use "thought_thinking" to denote the mention of the subject's general cognitive processing activities.

### Input Text: {}

### Output Text: {}
'''

def formatting_prompts_func(example):
    """
    Format each example so the LLM learns from full instructions + input text and produces the expected output.
    """
    output_texts = []
    for i in range(len(example["unprocessed"])):
        input_text = example["unprocessed"][i].split("### Input Text:")[1].split("### Output Text:")[0].strip()
        expected_output = example["processed"][i].strip()

        # Format the example with the full template
        formatted_text = template.format(input_text, expected_output)
        output_texts.append(formatted_text)

    return output_texts


# Parameters for training arguments details => https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L158
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing = True,
    max_grad_norm= 0.3,
    num_train_epochs=10, 
    learning_rate=2e-4,
    bf16=True,
    save_strategy="epoch",
    save_total_limit=10,
    logging_steps=10,
    output_dir=output_dir,
    optim="paged_adamw_32bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    ddp_find_unused_parameters=False,
    evaluation_strategy="no",
    #load_best_model_at_end=True,
    #metric_for_best_model='eval_loss'
)

trainer = SFTTrainer(
    base_model,
    train_dataset=train_dataset,
    #eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_seq_length=2048,
    formatting_func=formatting_prompts_func,
    args=training_args,
    #callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
)


#trainer.train(resume_from_checkpoint=True) 
trainer.train() 
trainer.save_model(output_dir)

output_dir = os.path.join(output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)



print("Training Complete! Model saved at:", output_dir)













