from datasets import load_dataset
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)
from transformers import Trainer, TrainingArguments


MODEL_NAME = "flax-community/gpt2-base-thai"
n_labels = 5


print('Loading configuraiton...')
model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=MODEL_NAME, num_labels=n_labels)
print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=MODEL_NAME)
#tokenizer.padding_side = "left"
#tokenizer.pad_token = tokenizer.eos_token
print('Loading model...')
model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=MODEL_NAME, config=model_config)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id


dataset = load_dataset('wongnai_reviews') 
train_data = dataset['train'].select(range(2000))
test_data = dataset['test'].select(range(200))

#train_data = train_data.rename_column("review_body", "text")
#train_data = train_data.rename_column("star_rating", "label")
#test_data = test_data.rename_column("review_body", "text")
#test_data = test_data.rename_column("star_rating", "label")

train_data = train_data.map(lambda x: tokenizer(x["review_body"], padding="max_length", truncation=True), batched = True)
test_data = test_data.map(lambda x: tokenizer(x["review_body"], padding="max_length", truncation=True), batched = True)

# print(train_data[0])
# print(len(train_data[0]['input_ids']))
# exit()

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir='./thaigpt_results',          # output directory
    num_train_epochs=1,              # total number of training epochs
    per_device_train_batch_size=4,  # batch size per device during training
    per_device_eval_batch_size=4,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',          # directory for storing logs
    logging_steps=10,
    #evaluation_strategy='epoch'
)

# instantiate the trainer class and check for available devices
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=test_data
)

trainer.train()


