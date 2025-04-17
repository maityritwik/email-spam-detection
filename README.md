# 📱 SMS Spam Detection with Machine Learning 🤖

Welcome to the **SMS Spam Detection** project! This repository contains a complete machine learning pipeline to classify SMS messages as **Spam** or **Not Spam** using Natural Language Processing (NLP) and traditional ML algorithms.

![Spam Detection Banner](https://img.shields.io/badge/NLP-Text_Classification-blue) ![Build](https://img.shields.io/badge/Python-3.10+-yellow) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Project Overview

Spam messages are a nuisance! This project leverages data science and machine learning to automatically detect whether a given SMS message is spam or not.

🔍 **Tech Highlights:**
- Text preprocessing using `sklearn` and `NLTK`
- Feature extraction with `TfidfVectorizer`
- Classification using `Multinomial Naive Bayes`
- Model persistence using `pickle`
- Exploratory Data Analysis and visualizations

---


---

## How to do it

Some visuals from the notebook:

**Word Frequency in Spam vs Ham**

| ![Spam Word Cloud](https://user-images.githubusercontent.com/placeholder/spam.png) | ![Ham Word Cloud](https://user-images.githubusercontent.com/placeholder/ham.png) |
| :--------------------------------------------: | :------------------------------------------: |
| *Spam Messages* | *Non-Spam (Ham) Messages* |

> *Word clouds from the SMS dataset show the most frequent terms.*

---

## 🛠️ How to Use

### ✅ Requirements

Install the required libraries:

```bash
pip install -r requirements.txt
```
---

## 🧪 Run the Notebook
Open and run the Notebook:

```bash
jupyter notebook sms-spam-detection.ipynb
```
It contains everything from loading the data to training the model and evaluating results.

---

## 📦 Model Inference (Optional Script)

```python
To use the trained model for prediction:
import pickle

with open('model.pkl', 'rb') as mfile, open('vectorizer.pkl', 'rb') as vfile:
    model = pickle.load(mfile)
    vectorizer = pickle.load(vfile)

def predict_sms(text):
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

print(predict_sms("Congratulations! You've won a free iPhone. Click here!"))

```
---
## 📈 Performance Metrics

| Metric |	Score |
|--------|--------|
|Accuracy|✅ 98.2%|
|Precision|✅ 98.7%|

---

## 📌 Key Learnings
- Importance of text preprocessing
- Role of TF-IDF in vectorizing natural language
- Practical implementation of Naive Bayes
- Handling imbalanced data in classification tasks

---

## 📜 License
This project is licensed under the MIT License. Feel free to use it for learning or integrating into your own projects! 😊

---

## ✨ Contributions & Feedback
Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

📧 Feedback or suggestions? Drop a message!
mailto:maityritwik2002@gmail.com

---

## 🙌 Acknowledgements
- UCI SMS Spam Dataset
- `scikit-learn`, `nltk`, `matplotlib`, and the Python community ❤️
