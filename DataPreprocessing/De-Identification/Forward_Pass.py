# %% [markdown]
# ## STEPS

# %% [markdown]
# * We go through the 4 steps that are required to de-identify a dataset (i.e run the forward pass on this dataset using a trained model)

# %% [markdown]
# ## STEP 0: LIBRARIES

# %%
import json

# %% [markdown]
# # ***Modified***

# %%
from transformers import HfArgumentParser, TrainingArguments

# %% [markdown]
# *   regex~=2021.11.2
# *   typing~=3.10.0.0
# *   setuptools~=58.5.2
# *   configparser~=5.1.0
# *   transformers==4.11.3
# *   tensorflow==2.7.0
# *   torch==1.10.0
# *   numpy==1.21.2
# *   scipy==1.7.2
# *   pandas==1.3.3
# *   scikit-learn==1.0.1
# *   jupyterlab==3.2.9
# *   ipywidgets==7.6.5
# *   matplotlib==3.4.3
# *   beautifulsoup4==4.10.0
# *   spacy==3.0.6
# *   nodejs==0.1.1
# *   stanza==1.3.0
# *   tensorflow-addons==0.15.0
# *   wget==3.2
# *   seqeval
# *   scispacy==0.4.0
# *   datasets==1.18.3
# *   allennlp==2.9.0
# *   pycorenlp==0.3.0
# *   https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz
# *   https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_lg-0.4.0.tar.gz
#    
# 
# 

# %% [markdown]
# # ***Modified***

# %%
!pip install pycorenlp
!pip install datasets
!pip install robust-deid
!pip install load_metric
!pip uninstall datasets
!pip install datasets==1.18.3
!pip install scispacy==0.4.0
!pip install seqeval
# !pip install spacy==3.0.6
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz

# %%
from robust_deid.ner_datasets import DatasetCreator
from robust_deid.sequence_tagging import SequenceTagger
from robust_deid.sequence_tagging.arguments import (
    ModelArguments,
    DataTrainingArguments,
    EvaluationArguments,
)
from robust_deid.deid import TextDeid
from google.colab import files

# %% [markdown]
# ## STEP 1: INITIALIZE

# %%
# # Initialize the path where the dataset is located (input_file).
# # Input dataset
# input_file = '../../data/notes/notes.jsonl'
# # Initialize the location where we will store the sentencized and tokenized dataset (ner_dataset_file)
# ner_dataset_file = '../../data/ner_datasets/test.jsonl'
# # Initialize the location where we will store the model predictions (predictions_file)
# # Verify this file location - Ensure it's the same location that you will pass in the json file
# # to the sequence tagger model. i.e. output_predictions_file in the json file should have the same
# # value as below
# predictions_file = '../../data/predictions/predictions.jsonl'
# # Initialize the file that will contain the original note text and the de-identified note text
# deid_file = '../../data/notes/deid.jsonl'
# # Initialize the model config. This config file contains the various parameters of the model.
# model_config = './run/i2b2/predict_i2b2.json'

# %% [markdown]
# # ***Modified***

# %%
uploaded = files.upload()
input_filename = list(uploaded.keys())[0]

input_file = input_filename
ner_dataset_file = 'test.jsonl'
predictions_file = 'predictions.jsonl'
deid_file = 'deid.jsonl'

# %% [markdown]
# ## STEP 2: NER DATASET
# * Sentencize and tokenize the raw text. We used sentences of length 128, which includes an additional 32 context tokens on either side of the sentence. These 32 tokens serve (from the previous & next sentence) serve as additional context to the current sentence.
# * We used the en_core_sci_lg sentencizer and a custom tokenizer (can be found in the preprocessing module)
# * The dataset stored in the ner_dataset_file will be used as input to the sequence tagger model

# %%
# Create the dataset creator object
dataset_creator = DatasetCreator(
    sentencizer='en_core_sci_sm',
    tokenizer='clinical',
    max_tokens=128,
    max_prev_sentence_token=32,
    max_next_sentence_token=32,
    default_chunk_size=32,
    ignore_label='NA'
)

# %%
# This function call sentencizes and tokenizes the dataset
# It returns a generator that iterates through the sequences.
# We write the output to the ner_dataset_file (in json format)
ner_notes = dataset_creator.create(
    input_file=input_file,
    mode='predict',
    notation='BILOU',
    token_text_key='text',
    metadata_key='meta',
    note_id_key='note_id',
    label_key='label',
    span_text_key='spans'
)
# Write to file
with open(ner_dataset_file, 'w') as file:
    for ner_sentence in ner_notes:
        file.write(json.dumps(ner_sentence) + '\n')

# %% [markdown]
# ## STEP 3: SEQUENCE TAGGING
# * Run the sequence model - specify parameters to the sequence model in the config file (model_config). The model will be run with the specified parameters. For more information of these parameters, please refer to huggingface (or use the docs provided).
# * This file uses the argmax output. To use the recall threshold models (running the forward pass with a recall biased threshold for aggressively removing PHI) use the other config files.
# * The config files in the i2b2 direct`ory specify the model trained on only the i2b2 dataset. The config files in the mgb_i2b2 directory is for the model trained on both MGB and I2B2 datasets.
# * You can manually pass in the parameters instead of using the config file. The config file option is recommended. In our example we are passing the parameters through a config file. If you do not want to use the config file, skip the next code block and manually enter the values in the following code blocks. You will still need to read in the training args using huggingface and change values in the training args according to your needs.

# %% [markdown]
# # ***Modified***

# %%
uploaded = files.upload()
model_config = list(uploaded.keys())[0]

# %%
parser = HfArgumentParser((
    ModelArguments,
    DataTrainingArguments,
    EvaluationArguments,
    TrainingArguments
))
# If we pass only one argument to the script and it's the path to a json file,
# let's parse it to get our arguments.
model_args, data_args, evaluation_args, training_args = parser.parse_json_file(json_file=model_config)

# %%
# Initialize the sequence tagger
sequence_tagger = SequenceTagger(
    task_name=data_args.task_name,
    notation=data_args.notation,
    ner_types=data_args.ner_types,
    model_name_or_path=model_args.model_name_or_path,
    config_name=model_args.config_name,
    tokenizer_name=model_args.tokenizer_name,
    post_process=model_args.post_process,
    cache_dir=model_args.cache_dir,
    model_revision=model_args.model_revision,
    use_auth_token=model_args.use_auth_token,
    threshold=model_args.threshold,
    do_lower_case=data_args.do_lower_case,
    fp16=training_args.fp16,
    seed=training_args.seed,
    local_rank=training_args.local_rank
)

# %%
# Load the required functions of the sequence tagger
sequence_tagger.load()

# %%
# Set the required data and predictions of the sequence tagger
# Can also use data_args.test_file instead of ner_dataset_file (make sure it matches ner_dataset_file)
sequence_tagger.set_predict(
    test_file=ner_dataset_file,
    max_test_samples=data_args.max_predict_samples,
    preprocessing_num_workers=data_args.preprocessing_num_workers,
    overwrite_cache=data_args.overwrite_cache
)

# %%
# Initialize the huggingface trainer
sequence_tagger.setup_trainer(training_args=training_args)

# %%
# Store predictions in the specified file
predictions = sequence_tagger.predict()
# Write predictions to a file
with open(predictions_file, 'w') as file:
    for prediction in predictions:
        file.write(json.dumps(prediction) + '\n')

# %% [markdown]
# ## STEP 4: DE-IDENTIFY TEXT
# 
# * This step uses the predictions from the previous step to de-id the text. We pass the original input file where the original text is present. We look at this text and the predictions and use both of these to de-id the text.

# %%
# Initialize the text deid object
text_deid = TextDeid(notation='BILOU', span_constraint='super_strict')

# %% [markdown]
# # ***Modified***

# %%
# De-identify the text - using deid_strategy=replace_informative doesn't drop the PHI from the text, but instead
# labels the PHI - which you can use to drop the PHI or do any other processing.
# If you want to drop the PHI automatically, you can use deid_strategy=remove
deid_notes = text_deid.run_deid(
    input_file=input_file,
    predictions_file=predictions_file,
    # deid_strategy='replace_informative',
    deid_strategy='remove',
    keep_age=False,
    metadata_key='meta',
    note_id_key='note_id',
    tokens_key='tokens',
    predictions_key='predictions',
    text_key='text',

)

# %%
# Write the deidentified output to a file
with open(deid_file, 'w') as file:
    for deid_note in deid_notes:
        file.write(json.dumps(deid_note) + '\n')


