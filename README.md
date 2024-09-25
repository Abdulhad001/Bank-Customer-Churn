# bank-customer-churn
This project aims to predict whether a bank customer will churn or not based on several customer attributes. Churn prediction helps banks identify customers who are likely to close their accounts, enabling them to take preemptive action to retain these customers.

The project utilizes machine learning techniques to analyze customer data and predict churn. It includes data preprocessing, exploratory data analysis (EDA), model training, and evaluation.

Table of Contents
Project Overview
Dataset
Requirements
Installation
Usage
Model Training
Evaluation
Results
Contributors
License
Dataset
The dataset used in this project contains customer data with the following features:

Customer ID
Credit Score
Geography
Gender
Age
Tenure
Balance
Number of Products
Has Credit Card
Is Active Member
Estimated Salary
Exited (Target variable indicating whether the customer churned or not)
Note: The dataset is either publicly available or proprietary, and you should provide information about where it can be accessed or if it needs to be uploaded.

Requirements
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter

You can install the dependencies using the following command:
pip install -r requirements.txt
Installation
Clone the repository:
bash
Copy code
git clone https://github.com/yourusername/bank-customer-churn.git
Navigate to the project directory:
bash
Copy code
cd bank-customer-churn
Install the required dependencies:
bash
Copy code
pip install -r requirements.txt
Usage
To run the project, follow these steps:

Open the Jupyter Notebook:
bash
Copy code
jupyter notebook Bank_Customer_Churn.ipynb
Execute each cell in the notebook sequentially. The notebook includes steps for:
Data loading
Data preprocessing (handling missing values, feature encoding, scaling, etc.)
Exploratory Data Analysis (EDA)
Model training (various machine learning models)
Model evaluation
Model Training
The notebook contains several machine learning models such as:

Logistic Regression
Random Forest
Support Vector Machine (SVM)
Each model is trained on the processed dataset, and evaluation metrics such as accuracy, precision, recall, and F1-score are computed.

python
Copy code
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
rf_model = RandomForestClassifier()
try:
    rf_model.fit(X_train, y_train)
except Exception as e:
    print(f"Error during model training: {e}")

# Predictions and evaluation
try:
    predictions = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy}")
except Exception as e:
    print(f"Error during prediction: {e}")
Evaluation
The models are evaluated using various performance metrics, including:

Accuracy
Precision
Recall
F1 Score
These metrics help determine the effectiveness of each model in predicting customer churn.


