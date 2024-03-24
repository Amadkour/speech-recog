import re

import numpy as np
import pandas as pd
import os

import torchaudio as torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2CTCTokenizer, Wav2Vec2Model, Wav2Vec2Config, Wav2Vec2ForCTC

# #
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base-100h", unk_token="[UNK]", pad_token="[PAD]",
                                                 word_delimiter_token="|")
from transformers import Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                                             return_attention_mask=True)
# #========== processor ================#
from transformers import Wav2Vec2Processor

# processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
asr_model = "muzamil47/wav2vec2-large-xlsr-53-arabic-demo"
asr_model = "anas/wav2vec2-large-xlsr-arabic"

processor = Wav2Vec2Processor.from_pretrained(asr_model)

os.environ["HUGGINGFACE_TOKEN"] = "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"

# Now you can interact with the Hugging Face Hub
# from huggingface_hub import notebook_login

# Authenticate with the Hugging Face Hub
# notebook_login()
from datasets import load_dataset, load_metric, Audio, Dataset

# common_voice_train = load_dataset("common_voice", "tr", split="validation")
from utiles.my_class_utiles import DataCollatorCTCWithPadding

tsv_file_path = "/Users/ahmedmadkour/Documents/speech-recog/my_datasets/cv-corpus-5.1-2020-06-22/ar/train.tsv"
df = pd.read_csv(tsv_file_path, sep="\t")
#
# # Assume the TSV file has two columns: 'audio_path' and 'transcript'
transcripts = df['sentence'].tolist()
audio_paths = ('/Users/ahmedmadkour/Documents/speech-recog/my_datasets/cv-corpus-5.1-2020-06-22/ar/clips/' + df[
    'path'].to_numpy()).tolist()
# #
resamplers = {  # all three sampling rates exist in test split
    48000: torchaudio.transforms.Resample(48000, 16000),
    44100: torchaudio.transforms.Resample(44100, 16000),
    32000: torchaudio.transforms.Resample(32000, 16000),
}
import librosa

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\“]'


def prepare_example(example):
    speech, sampling_rate = torchaudio.load(example["path"])
    example["audio"] = resamplers[sampling_rate](speech).squeeze().numpy()
    example["sampling_rate"] = 16000
    example["sentence"] = re.sub(chars_to_ignore_regex, '', example["sentence"]).lower()

    return example


def prepare_dataset(batch):
    audio = batch["audio"]
    # batched output is "un-batched"
    batch["input_values"] = processor(audio, sampling_rate=batch["sampling_rate"],return_tensors="pt",padding=True,truncation=True,max_length=16000).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    with processor.as_target_processor():
        # Flatten the nested lists in batch["sentence"]
        flattened_sentences = [sentence for sublist in batch["sentence"] for sentence in sublist]
        labels = processor(flattened_sentences, return_tensors="pt", padding=True, truncation=True).input_ids
        # Flatten the nested list structure of the labels
        batch["labels"] = [label for sublist in labels.tolist() for label in sublist]

    # with processor.as_target_processor():
    #     batch["labels"] = processor(batch["sentence"],return_tensors="pt",padding=True,truncation=True,max_length=16000).input_ids
    return batch


data = {
    'sentence': transcripts[:1],
    'path': audio_paths[:1],
}
# dataset = Dataset.from_pandas(pd.DataFrame(data))
# dataset = dataset.map(prepare_example)
import torchaudio

def process_example(audio_path, sentence):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_path)
    audio = waveform.squeeze().numpy()

    # Prepare the example
    example = {
        'audio': audio,
        'sampling_rate': 16000,
        'sentence': sentence
    }
    return example
examples = []
for audio_path, sentence in zip(data['path'], data['sentence']):
    example = process_example(audio_path, sentence)
    examples.append(example)
dataset = Dataset.from_list(examples)
dataset = dataset.map(prepare_dataset)

# inputs = processor(dataset["speech"], sampling_rate=16000, return_tensors="pt", padding=True)

#
# # data = {'sentence': transcripts, 'path': audio_paths}
# # # print(transcripts)
# # # print(audio_paths)
# from transformers import Wav2Vec2CTCTokenizer
#
# #
# tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base-100h", unk_token="[UNK]", pad_token="[PAD]",
#                                                  word_delimiter_token="|")
# from transformers import Wav2Vec2FeatureExtractor
#
# feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
#                                              return_attention_mask=True)
# # #========== processor ================#
# from transformers import Wav2Vec2Processor
#
# processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
# data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
# from evaluate import load
#
from datasets import load_dataset, load_metric, Audio


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

from  evaluate import  load
wer_metric = load("wer", trust_remote_code=True)
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-large-xlsr-53",
    attention_dropout=0.1,
    hidden_dropout=0.05,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    gradient_checkpointing=True,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer)
)
# model = Wav2Vec2Model(configuration)
model.freeze_feature_encoder()


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


from transformers import TrainingArguments
from accelerate import Accelerator, DataLoaderConfiguration

dataloader_config = DataLoaderConfiguration(
    split_batches=False,
    even_batches=True,
    use_seedable_sampler=True
)

accelerator = Accelerator(dataloader_config=dataloader_config)
language_code = 'ar'
language_name = 'arabic'
new_output_models_dir = f"/workspace/output_models/{language_code}/wav2vec2-large-xlsr-{language_name}"

training_args = TrainingArguments(
  output_dir='model_checkpoints2/',
  group_by_length=True,
  per_device_train_batch_size=16,
  gradient_accumulation_steps=2,
  dataloader_num_workers = 10,
  evaluation_strategy="steps",
  num_train_epochs=1,
  fp16=False,
  save_steps=400,
  eval_steps=400,
  logging_steps=400,
  learning_rate=1e-4,
  warmup_steps=500,
  save_total_limit=2,
)
from transformers import Trainer

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
print(dataset.data.shape)
from transformers import Trainer
if __name__ == '__main__':
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=processor.feature_extractor,
    )
    trainer.train()
