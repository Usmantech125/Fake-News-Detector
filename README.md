# 📰 Fake News Detection using Decision Tree Classifier

This project was developed as part of my **Codveda Machine Learning Internship**. The goal was to build a reliable fake news detection system using **natural language processing (NLP)** techniques and a **Decision Tree Classifier**.

---

## 🎯 Objective

- Combine and preprocess real and fake news datasets.
- Clean text data: lowercase conversion, punctuation removal, and stopword removal.
- Convert textual data into numerical format using **TF-IDF Vectorization**.
- Train a **Decision Tree model** to classify news articles as real or fake.
- Evaluate performance using metrics like accuracy, F1-score, and confusion matrix.

---

## 🧰 Tools & Libraries Used

- Python 🐍  
- `pandas` – Data handling  
- `nltk` – Stopwords removal  
- `scikit-learn` – ML models, vectorization, evaluation  
- `matplotlib`, `seaborn` – For visualizations  

---

## 📊 Workflow Summary

### 1. 📂 Data Preparation
- Used two datasets: `True.csv` (real news) and `Fake.csv` (fake news)
- Added a binary label: `1` for real, `0` for fake
- Combined `title` and `text` into a single `content` column

### 2. 🧹 Text Preprocessing
- Converted all text to lowercase
- Removed special characters and punctuation using regular expressions
- Removed English stopwords using `nltk.corpus.stopwords`

### 3. 📐 Feature Extraction
- Applied **TF-IDF Vectorizer** to convert cleaned text into numerical vectors
- Split the dataset: **80% training / 20% testing**

### 4. 🤖 Model Building
- Used `DecisionTreeClassifier` with:
  - `criterion='entropy'`
  - `max_depth=10`
  - `min_samples_split=5`
  - `min_samples_leaf=2`
- Trained on the TF-IDF-transformed feature vectors

### 5. 📈 Evaluation
- **Accuracy:** 99.5%  
- **F1 Score:** 0.995  
- **Confusion Matrix:** Extremely low false positives/negatives

---

## 🔍 Key Insights

- Proper text preprocessing (like stopwords removal) significantly improves model performance.
- Decision Trees can perform exceptionally well when overfitting is controlled.
- TF-IDF is an effective technique for turning unstructured text into machine-readable features.

---

## 📁 Files Included

- `Fake-News-Detector.ipynb` – Jupyter Notebook with full code & comments  
- `True.csv`, `Fake.csv` – Raw datasets  
- `README.md` – This project documentation  

---

## 💡 Future Scope

- Apply ensemble models like **Random Forest**, **XGBoost**, or **VotingClassifier**
- Integrate **stemming** or **lemmatization** for deeper NLP preprocessing
- Deploy the model as a **web app** using Streamlit or Flask
- Perform hyperparameter tuning via GridSearchCV

---

## 🧠 Internship Note

This project was completed as part of my #Codveda internship under the **Machine Learning domain**, where I applied theoretical concepts to real-world, unstructured textual data. It strengthened my skills in NLP, data preprocessing, and model evaluation.

---

### 🔖 Tags  
`#CodvedaJourney` `#FakeNewsDetection` `#MachineLearning` `#NLP` `#TextClassification` `#DecisionTree` `#MLInternship` `#TFIDF` `#Python`

