# AI Clinic Project

## Introduction

The AI Clinic project aims to answer medical questions based on user input. The system utilizes a fine-tuned T5 model to generate responses to medical queries, enabling users to receive accurate and contextually relevant answers. Additionally, the system allows for continuous improvement by fine-tuning the model with new user-provided data. Users can fine-tune the model with their own high-rate answers, so that each user can have their personalized model. <br/>
**Demo Video**: https://youtu.be/WCmDtncoVNE


## Technologies and Tools
- **NLP**: Hugging Face Transformers library
  - The 🤗 Transformers library is used for fine-tuning and running the model.
- **Model**: Fine-tuned Flan-T5-Small
  - The model used to answer medical questions is based on [Flan-T5-Small](https://huggingface.co/google/flan-t5-small), and fine-tuned by following datasets: [MedQuad-MedicalQnADataset](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset), [ChatDoctor-HealthCareMagic-100k](https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k)
- **Frontend**: HTML, CSS, JavaScript
  - The user interface is designed with HTML and styled using CSS. JavaScript is used for interactivity and dynamic content updates.
- **Backend**: Flask (Python)
  - Flask is used to handle HTTP requests, manage sessions, and serve the web pages. It integrates seamlessly with the machine learning model and database operations.
- **Database**: MySQL
  - MySQL is used to store user questions, answers, and ratings. The database also tracks which records have been used for fine-tuning to avoid redundancy.


## User Menu

1. **Home Page**
   - Provides a sidebar with previous questions
   - A form to ask new questions.
   - Display questions and answers.
   - Includes a button to trigger model fine-tuning with collected data.
2. **Ask a Question**
   - Users can type their medical question in the provided text area and submit it.
   - The system generates an answer using the fine-tuned T5 model and displays it on the same page.
   - Users can view the question and the generated answer immediately.
3. **Rate the Answer**
   - After receiving an answer, users can rate it using a dropdown menu.
   - Ratings range from 1 (Poor) to 5 (Excellent).
   - The rating form includes hidden fields to store the question and answer, ensuring accurate data submission.
4. **Fine-Tune the Model**
   - A button in the sidebar allows users to trigger the fine-tuning process.
   - The system retrieves high-rated records from the database, updates their status, and uses them to fine-tune the T5 model.

## Preparation of the Model
See details at https://github.com/Yiheng-Gao/AI-Clinic-flan-t5-fine-tuning-process


## File and Folder Descriptions

- **datasets/**: Contains the CSV file `dataset.csv` used for model fine-tuning.
- **fine-tuned-model/**: Contains the original fine-tuned model files.
- **personized_model/**: Contains the models fine-tuned by user records, with subdirectories like `checkpoint-50` indicating different checkpoints.
- **runs/**: Directory used for logging and storing training runs (e.g., TensorBoard logs).
- **static/**: Contains static files like JavaScript and CSS.
- **templates/**: Contains HTML templates for rendering web pages.
  - **index.html**: Main HTML template for the web application.
- **app.py**: Main backend logic including endpoints for the Flask application.
  - Handles HTTP requests, processes user inputs, and serves web pages.
  - Integrates with the machine learning model to generate answers and manage records.
- **create_dataset.py**: Helps get records from the database and prepare the CSV file for fine-tuning the model.
  - Fetches data from the database and writes it to `dataset.csv`.
- **RecordDAL.py**: Data Access Layer for interacting with the MySQL database.
  - Contains functions to get and save records and manage data.
- **training.py**: Contains the fine-tuning method for the machine learning model.
  - Defines the `fine_tuning` function to fine-tune the `flan-t5-small` model using the dataset.

## Reference
[FLAN-T5 Tutorial: Guide and Fine-Tuning](https://www.datacamp.com/tutorial/flan-t5-tutorial) - A complete guide to fine-tuning a FLAN-T5 model for a question-answering task using transformers library, and running optmized inference on a real-world scenario.
