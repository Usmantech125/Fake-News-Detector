# 📰 Fake News Detection using Decision Tree Classifier

This project is part of my **Codveda Machine Learning Internship**. The objective was to build a model that can accurately classify news as **real or fake** using natural language processing and a decision tree algorithm.

---

## 🎯 Objective

- Preprocess and combine real and fake news datasets.
- Convert textual data into numerical form using TF-IDF vectorization.
- Train a **Decision Tree Classifier** to predict fake or real news.
- Evaluate the model using accuracy, F1-score, and a confusion matrix.

---

## 🧰 Tools & Libraries Used

- Python
- pandas
- scikit-learn
- matplotlib & seaborn
- TfidfVectorizer (for feature extraction)

---

## 📊 Workflow Summary

### 1. Data Preparation
- Merged two datasets: `True.csv` and `Fake.csv`
- Labeled real news as `1` and fake news as `0`
- Combined title and body into a single `content` column

### 2. Feature Engineering
- Applied **TF-IDF Vectorization** to convert text into numerical vectors
- Split the data into training and testing sets (80/20 split)

### 3. Model Building
- Used `DecisionTreeClassifier` with:
  - `criterion='entropy'`
  - `max_depth=10`
  - `min_samples_split=5`
  - `min_samples_leaf=2`
- Trained the model on the TF-IDF features

### 4. Evaluation
- Accuracy Score: ✅ High classification performance
- F1 Score: ✅ Balanced measure of precision and recall
- Confusion Matrix: ✅ Clear visualization of results

---

## 🔍 Key Insights

- Decision Trees can effectively classify fake vs real news when well-regularized
- TF-IDF is useful for converting unstructured text into ML-ready vectors

---

## 📁 Files Included

- `Fake-News-Detector.ipynb` – Complete implementation
- `True.csv`, `Fake.csv` – News datasets
- `README.md` – Project documentation

---

## 💡 Future Scope

- Try ensemble methods like Random Forest or XGBoost
- Add more preprocessing (lemmatization, stemming, etc.)
- Deploy the model as a web app

---

## 🧠 Internship Note

This project was developed as part of my #Codveda internship under the **Machine Learning domain**, helping me apply theoretical ML concepts to practical, real-world datasets.

#CodvedaJourney #FakeNewsDetection #DecisionTree #TextClassification #MachineLearning #NLP #MLInternship #CodvedaExperience
