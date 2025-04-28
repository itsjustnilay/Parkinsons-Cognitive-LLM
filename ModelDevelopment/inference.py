import os
import re
import spacy
from glob import glob
from bs4 import BeautifulSoup
import csv
import pandas as pd
import json
from vllm import LLM, SamplingParams
from tqdm import tqdm

gpus = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = gpus
device_map = [f"cuda:{i}" for i in gpus.split(",")]

model_dir = "path/to/merged/model/"

stop_token = "### STOP_HERE"

#sampling_params = SamplingParams(max_tokens=1024, stop='<EOS>', temperature=0)
sampling_params = SamplingParams(
        max_tokens=2048, 
        stop=['### Input Text:', '<EOS>', stop_token], 
        temperature=0
    )
llm = LLM(model=model_dir, tensor_parallel_size=len(device_map))
    

template = '''"### Task:
Your task is to identify and annotate specific medical entities in the given text. For each entity found, you should provide an output in JSON format that includes the following information: The start index of the entity in the text,The end index of the entity in the text, The extracted entity text, and The entity type

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
use "somato_kinesthesia" to denote the mention of sensations realted to body moments.
use "smell" to denote the mention of olfactory sensations.
use "sound" to denote the mention of perception of hearing.
use "taste" to denote the mention of senstation of gustation.
use "introception" to denote the mention of internal bodily sensations.
use "introception_pulse_hb" to denote the mention of awareness of pulse or heartbeats.
use "introception_breathing" to denote the mention of act/awareness of breathing.
use "introception_hunger" to denote the mention of sensations hunger.
use "introception_satiety" to denote the mention of sensations of being full after a meal/after food.
use "introception_thirst" to denote the mention of feeling dehydrated.
use "introception_temp" to denote the mention of internal sensations of being hot or cold.
use "introception_pain" to denote the mention of the internal pain.
use "introception_energy_level" to denote the mention of the sensations related to the amount of energy or fatigue.
use "introception_tension" to denote the mention of sensations of bodily tension.
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

### Entity Definitions:
visual: Perceptions related to sight, including attributes like color, size, shape, patterns, movement, and spatial relationships between objects.
visual_color: Description of hue, tint, or shade of an object.
visual_brightness: Description of light intensity or glare.
visual_shape: Description of the form or outline of objects.
visual_size: Description of the dimension or magnitude of objects.
visual_num_quant: Description of the count or amount of physical objects.
visual_texture: Description of the surface characteristics or feel.
visual_pattern: Description of repeated designs or regular arrangements.
visual_motion: Description of movement of objects, seen visually, includes the actions done by others.
visual_spatial: Description of the positioning or arrangement of objects in space.
somatosensory: Physical sensations felt through the skin or body, such as touch, temperature, pain, pressure, vibration, and body movement or position.
somato_touch: Sensations of physical contact.
somato_pain: Sensations of physical discomfort originating from the skin or muscles, from external reasons.
somato_temp: Sensations of heat or cold perceived on the skin, from external reasons.
somato_pressure: Sensations of force applied to the skin
somato_vibration: Sensations of rapid oscillations felt on the skin.
somato_proprio: Awareness of body position and movement.
somato_kinesthesia: Sensations related to body movements and the feel of muscles being moved.
smell: Sensory experiences related to smell, including pleasant or unpleasant odors.
sound: Perceptions related to hearing, including sounds like voices, machinery, or environmental noises but not limited to those only.
taste: Sensations related to the sense of taste, such as flavors experienced by the subject.
introception: Internal bodily sensations and physiological states, including awareness of heartbeat, breathing, hunger, thirst, internal temperature, pain (non-external), energy levels, and muscle tension.
introception_pulse_hb: Awareness of heartbeats or pulse.
introception_breathing: Sensations related to the act of breathing.
introception_hunger: Sensations of needing food.
introception_satiety: Sensations of being full after a meal/after food.
introception_thirst: Sensations of needing liquid, mention of feeling dehydrated.
introception_temp: Internal sensations of being hot or cold.
introception_pain: Internal physical discomfort that is not externally induced
introception_energy_level: Sensations related to the amount of energy or fatigue.
introception_tension: Sensations of muscle contraction or mental strain.
introception_urine_bowels: Sensations indicating the need to relieve oneself.
time: This category captures temporality, as well as references to time, including specific times of day, days of the week, months, seasons, and holidays or special days.
time_of_day: Specific mentions of time points during the day.
time_day_of_week: Mentions of a specific day in the week.
time_season: Mentions of periods within the year, including references to seasons, like Spring, Summer, Fall, Winter, or Rainy.
time_month: Mentions of specific months.
time_holiday: Mentions of holidays or important dates or special days.
location: References to places or settings where actions or events occur. It includes specific locations (e.g., a room or city) or general descriptions of spatial positioning.
acion: Descriptions of physical or mental actions performed by the subject or others, this does not entail mental action or others actions.
social_interaction: This category includes references to interactions or exchanges between people. It covers both verbal communication and non-verbal social interactions, such as gestures or expressions.
emotion: The subject’s emotional experiences or feelings. It includes both positive and negative emotions.
thought: Captures internal mental processes and reflections that do not necessarily lead to concrete actions or plans. It includes memories, judgments, self-reflections, decision-making, and attributions of thoughts to others, and mentalizing.
thought_memory: Recall of past events, emotions, or facts.
thought_judgment: Cognitive evaluations or decisions made.
thought_self_reflection: Reflections on one's thoughts or actions.
thought_mentalizing: Attributing thoughts or feelings to oneself or others.
thought_thinking: General cognitive processing activities.

### Input Text: {}

### Output Text:
'''

    
def batch_list(input_list, batch_size):
    """Splits the list into batches of size `batch_size`."""
    return [input_list[i:i + batch_size] for i in range(0, len(input_list), batch_size)]

    
def extract_input_text(unprocessed_text):
    """Extracts text inside the first and last quotes after '### Input Text:'."""
    match = re.search(r'### Input Text:\s*"(.*)"', str(unprocessed_text), re.DOTALL)
    return match.group(1).strip() if match else None

def preprocess_text(input_text):
    """Fix parentheses and ensure consistent punctuation before passing to the LLM."""
    if not input_text:
        return None  # Skip if empty

    # Fix unclosed parentheses at the end
    input_text = re.sub(r"\(\s*([^\)]*)\s*$", r"(\1)", input_text)

    # Ensure a period exists after a closing parenthesis **only when it's at the end of the sentence**
    input_text = re.sub(r"\)$", ").", input_text)  # Adds period only if `)` is the last character

    # Add stop token to ensure generation stops
    return input_text.strip() + f" {stop_token}"


def process_batch(batch, start_index):
    """Processes a batch of input texts with the LLM and saves outputs."""
    extracted_inputs = [extract_input_text(text) for text in batch]
    extracted_inputs = [text for text in extracted_inputs if text]  # Remove None values

    if not extracted_inputs:
        return start_index  # Skip batch if no valid inputs

    prompts = [template.format(preprocess_text(text)) for text in extracted_inputs]
    batch_outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(batch_outputs):
        ner_output = output.outputs[0].text.strip()

        # Truncate at stop token if present
        if stop_token in ner_output:
            ner_output = ner_output.split(stop_token)[0].strip()

        # Save output immediately with correct index
        output_path = os.path.join(output_dir, f"output_{start_index + i}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ner_output, f, indent=4)

    return start_index + len(extracted_inputs)  # Return updated index


test_input = "Adds 80 grams of water to his coffee pot, lets it soak up for 30 seconds, and then adds another 80 grams of water, letting it soak for another 30 seconds."
test_input = preprocess_text(test_input)
test_prompt = template.format(test_input)

print("\nRunning test inference for:", test_input)

# Run LLM inference
output = llm.generate([test_prompt], sampling_params)
generated_text = output[0].outputs[0].text.strip()

import json; json.dump(generated_text, open("generated_text.json", "w"), indent=4)



# Load test dataset
test_csv_path = "path/to/test/data"
test_df = pd.read_csv(test_csv_path)

output_dir = "path/to/output/NER/"
os.makedirs(output_dir, exist_ok=True)

batch_size = 50


print(f"\nRunning inference on {len(test_df)} samples...")



global_index = 0  # Track index across batches

# Prepare batched inputs
batched_inputs = [test_df["unprocessed"][i:i + batch_size].tolist() for i in range(0, len(test_df), batch_size)]

for batch in tqdm(batched_inputs, total=len(batched_inputs), desc="Processing Batches"):
    global_index = process_batch(batch, global_index)  # Pass global_index as start_index

print("\nNER Inference Complete! Outputs saved in ./output/NER/")




