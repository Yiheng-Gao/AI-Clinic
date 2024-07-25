from transformers import T5Tokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import load_dataset
from create_dataset import get_csv
import nltk
import evaluate
import numpy as np

def fine_tuning():
    finetuned_model = T5ForConditionalGeneration.from_pretrained("./fine-tuned-model").to("cuda")
    finetuned_tokenizer = T5Tokenizer.from_pretrained("./fine-tuned-model")
    data_collator = DataCollatorForSeq2Seq(tokenizer=finetuned_tokenizer, model=finetuned_model)
    get_csv()
    ds = load_dataset("csv", data_files="./datasets/dataset.csv", split="train")
    ds=ds.train_test_split(test_size=0.3)
    print(ds)
    # We prefix our tasks with "answer the question"
    prefix = "If you are a doctor, please answer the medical questions based on the patient's description: "

    # Define the preprocessing function

    def preprocess_function(examples):
        # The "inputs" are the tokenized answer:
        inputs = [prefix + doc for doc in examples["question"]]
        model_inputs = finetuned_tokenizer(inputs, max_length=512, truncation=True)
            
        # The "labels" are the tokenized outputs:
        labels = finetuned_tokenizer(text_target=examples["answer"], 
                                max_length=512,         
                                truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
        
    tokenized_dataset = ds.map(preprocess_function, batched=True)

    nltk.download("punkt", quiet=True)
    metric = evaluate.load("rouge")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        # decode preds and labels
        labels = np.where(labels != -100, labels, finetuned_tokenizer.pad_token_id)
        decoded_preds = finetuned_tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = finetuned_tokenizer.batch_decode(labels, skip_special_tokens=True)

        # rougeLSum expects newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        
        return result



    training_args = Seq2SeqTrainingArguments(
        output_dir="./personized_model",
        evaluation_strategy="epoch",
        learning_rate=1e-4,  
        per_device_train_batch_size=4,  
        per_device_eval_batch_size=2, 
        num_train_epochs=10,
        weight_decay=0.01,
        logging_steps=10,
        save_total_limit=1,
        predict_with_generate=True,
        save_steps=5,
        push_to_hub=False
    
    )

    # Initialize the Trainer
    trainer = Seq2SeqTrainer(
        model=finetuned_model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=finetuned_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()







