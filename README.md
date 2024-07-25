# AI Clinic Project

## Introduction

The AI Clinic project aims to answer medical questions based on user input. The system utilizes a fine-tuned T5 model to generate responses to medical queries, enabling users to receive accurate and contextually relevant answers. Additionally, the system allows for continuous improvement by fine-tuning the model with new user-provided data. The web application features a user-friendly interface where users can ask questions, view previous interactions, rate responses, and trigger model fine-tuning based on collected data. Users can rate each answer, and users can fine-tune the model with high-rate answers, so that each user can have their personalized model. 


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
   - Displays a list of previous questions and answers.
   - Provides a sidebar with previous questions and a form to ask new questions.
   - Includes a button to trigger model fine-tuning with collected data.
2. **Ask a Question**
   - Users can type their medical question in the provided textarea and submit it.
   - The system generates an answer using the fine-tuned T5 model and displays it on the same page.
   - Users can view the question and the generated answer immediately.
3. **Rate the Answer**
   - After receiving an answer, users can rate it using a dropdown menu.
   - Ratings range from 1 (Poor) to 5 (Excellent).
   - The rating form includes hidden fields to store the question and answer, ensuring accurate data submission.
4. **Fine-Tune the Model**
   - A button in the sidebar allows users to trigger the fine-tuning process.
   - The system retrieves high-rated records from the database, updates their status, and uses them to fine-tune the T5 model.
   - Successful fine-tuning is acknowledged with a flash message.
