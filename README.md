# 🌸 Iris Flower Classification

A simple yet powerful **Machine Learning** project that predicts the species of Iris flowers — *Setosa, Versicolor,* or *Virginica* — based on their petal and sepal measurements.  
Built using **Python**, **Scikit-Learn**, and **Jupyter Notebook**.

## Output(as it would apper in Terminal);
First 5 rows of dataset:

sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)  target
0                5.1               3.5                1.4               0.2       0
1                4.9               3.0                1.4               0.2       0
2                4.7               3.2                1.3               0.2       0
3                4.6               3.1                1.5               0.2       0
4                5.0               3.6                1.4               0.2       0


Confusion Matrix:
[[10  0  0]
 [ 0  7  0]
 [ 0  0  3]]

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00         7
           2       1.00      1.00      1.00         3

    accuracy                           1.00        20
   macro avg       1.00      1.00      1.00        20
weighted avg       1.00      1.00      1.00        20


Model Accuracy: 100.00%

Predicted Iris Class: setosa


## 📘 Project Overview

The **Iris Flower Classification** is one of the most famous beginner-friendly datasets in machine learning.  
This project aims to classify Iris flowers into three species using various ML algorithms and evaluate their performance.


## 🧠 Algorithms Used
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- K-Nearest Neighbors (KNN)  
- Support Vector Machine (SVM)

## 📊 Dataset Information

The **Iris dataset** contains:
- **150 samples**
- **4 features**:
  - Sepal Length  
  - Sepal Width  
  - Petal Length  
  - Petal Width  
- **3 classes**:
  - *Iris-setosa*  
  - *Iris-versicolor*  
  - *Iris-virginica*

## ⚙️ Technologies Used
- **Python**
- **NumPy**
- **Pandas**
- **Matplotlib / Seaborn**
- **Scikit-learn**
- **Jupyter Notebook**
  

## 🚀 Steps to Run the Project

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Iris-Flower-Classification.git
Navigate to the project directory

bash
Copy code
cd Iris-Flower-Classification
Install dependencies

bash
Copy code
pip install -r requirements.txt
Run the notebook

bash
Copy code
jupyter notebook Iris_Classification.ipynb
📈 Model Accuracy Example
Algorithm	Accuracy
Logistic Regression	96%
Decision Tree	94%
Random Forest	97%
KNN	95%
SVM	98%

📷 Visualizations
📊 Pair Plot of Iris Features
🌼 Confusion Matrix
📈 Feature Importance Chart

(Plots are generated using Matplotlib and Seaborn for better understanding of the data.)

🏁 Output Example
vbnet
Copy code
Predicted Class: Iris-setosa 🌸
🧩 Future Enhancements
Add a web interface using Flask or Streamlit.

Deploy the model using Heroku or Render.

Experiment with deep learning models for better accuracy.

💡 Learning Outcome
This project helps you understand:

Data preprocessing and visualization

Model selection and evaluation

Building and tuning ML classifiers

🌟 Acknowledgement
Dataset Source: UCI Machine Learning Repository

