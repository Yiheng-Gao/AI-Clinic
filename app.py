from flask import Flask, render_template, request, redirect, url_for, flash
from transformers import T5ForConditionalGeneration, T5Tokenizer, DataCollatorForSeq2Seq
from RecordDAL import save_case, get_records
import training
import torch
import re
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'

path = "./personized_model"
model_dir = "./fine-tuned-model"
dir_list = os.listdir(path)
for file in dir_list:
    if file.startswith("checkpoint"):
        model_dir="./personized_model/"+file


print(model_dir)
finetuned_model = T5ForConditionalGeneration.from_pretrained(model_dir).to("cuda")
finetuned_tokenizer = T5Tokenizer.from_pretrained(model_dir)





@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    input_text = "If you are a doctor, please answer the medical questions based on the patient's description: " + question
    input_ids = finetuned_tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

    outputs = finetuned_model.generate(
        input_ids,
        max_length=300,
        min_length=100,
        repetition_penalty=2.0
    )
    answer = finetuned_tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer=answer.replace("Chat Doctor","AI Clinic")
    answer = re.sub(r'([.?])\s', r'\1\n\n', answer)
    records = get_records()
    return render_template('index.html', question=question, answer=answer,records=records)

@app.route('/')
def home():
    records = get_records()
    return render_template('index.html', records=records)

@app.route('/rate', methods=['POST'])
def rate():
    question = request.form['question']
    answer = request.form['answer']
    rating = int(request.form['rating'])
    save_case(question, answer, rating)
    return redirect(url_for('home'))


@app.route('/fine-tune', methods=['POST'])
def fine_tune():
    try:
        training.fine_tuning()  # Call the fine_tuning function from the training module
        flash("Model fine-tuned successfully!", "success")
    except Exception as e:
        flash(f"Error during fine-tuning: {e}", "danger")
    return redirect(url_for('home'))
   

if __name__ == '__main__':
    app.run(debug=True)
