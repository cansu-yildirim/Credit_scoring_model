# Credit Scoring Model for Home Equity Loan Approval

## Overview

This project aims to automate the home equity loan approval process by developing a credit scoring model using machine learning techniques. The model is trained on historical data from applicants, ensuring both accuracy and compliance with the Equal Credit Opportunity Act. By leveraging statistical methods and data preprocessing, this model predicts the likelihood of loan default, helping streamline the decision-making process for financial institutions.

## Project Description

The goal of this project is to create a robust, data-driven solution for determining home equity loan approvals based on various factors such as loan amount, applicant's financial history, and other demographic data. The model predicts whether a borrower will default on their loan (binary classification: default/no default).

Key objectives include:
- Build a credit scoring model using supervised machine learning techniques.
- Evaluate the model using performance metrics such as accuracy, precision, recall, and F1-score.
- Ensure compliance with the Equal Credit Opportunity Act guidelines during the model development.

## Data Description

The dataset contains 5,960 records of home equity loan applicants. Each record includes 12 features, such as loan amount, income, and credit history, along with a binary target variable indicating whether the applicant eventually defaulted on the loan.

### Dataset Columns:
- `BAD`: Target variable (1 = Default, 0 = No Default)
- `LOAN`: Loan amount
- `MORTDUE`: Mortgage due
- `VALUE`: Home value
- `REASON`: Loan reason
- `JOB`: Job type
- `YOJ`: Years on the job
- `DEROG`: Number of derogatory reports
- `DELINQ`: Number of delinquent credit lines
- `CLAGE`: Age of credit line
- `NINQ`: Number of inquiries
- `CLNO`: Number of credit lines
- `DEBTINC`: Debt-to-income ratio

## Technologies Used

- Python
- Pandas (Data Manipulation)
- NumPy (Numerical Computation)
- Scikit-learn (Machine Learning Algorithms)
- Matplotlib & Seaborn (Data Visualization)

## Installation

To run this project locally, follow these steps:

* Clone the repository:
   ```bash
   git clone https://github.com/cansu-yildirim/credit-scoring-model.git

* Navigate to the project directory:
   ```bash
   cd credit-scoring-model

