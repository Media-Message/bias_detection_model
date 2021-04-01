# Bias Detection Model
Repository to hold code for subjective bias identification experiments. 

## Run NER Experiment

We used AWS Sagemaker notebooks with a single GPU instance to run experiments. A typical experiment command might look like this:

``` 
python run_ner.py \
    --model_name_or_path GroNLP/bert-base-dutch-cased \
    --dataset_name full-cased \
    --output_dir output-full \
    --overwrite_output_dir \
    --evaluation_strategy epoch \
    --save_steps 20000 \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs 4 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --fp16
```

Multiple pre-trained models exist for this task including `GroNLP/bert-base-dutch-cased` , `pdelobelle/robbert-v2-dutch-base` and `bert-base-multilingual-cased` . We observed `GroNLP/bert-base-dutch-cased` to work best for our domain. 

## Run NER + POS tags experiment

A typical experiment command might look like this:

``` 
python main.py \
    --model_name_or_path GroNLP/bert-base-dutch-cased \
    --dataset_name sample-cased \
    --output_dir output-sample \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_predict \
    --num_train_epochs 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy epoch \
    --fp16
```

This involves concatenating POS embeddings or one-hot encodings to the sequence output of the Bert model. 

## Experiment Logging

We make use of Weights & Biases to log experiment results. Make sure to add the `WANDB_API_KEY` to the environment, prior to running experiments. 

``` 
export WANDB_API_KEY=<wandb-api-key>
```
