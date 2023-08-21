# MLOps Churn Project
![](https://github.com/zabull1/mlops_churn_project/blob/main/assets/images/customer-churn.jpeg)

# 1. Problem description

A leading telecom company, **Zabull Telecoms**, has been experiencing a high churn rate among its customers. The churn rate refers to the percentage of customers who cancel their subscription services within a given period. This high churn rate is causing a significant loss in revenue and customer base, and the company is keen on reducing it to remain competitive in the market.

The telecom company collects customer data, including call records, customer demographics, and service usage history. The management believes that by implementing a machine learning solution to predict customer churn in advance, they can take proactive measures to retain at-risk customers and reduce churn.

We aim to develop an end-to-end MLOps (Machine Learning Operations) solution for churn prediction.

* Problem type: Supervised/Classification

## Dataset

The data can be found on [kaggle](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets)

- **State**: the US state in which the customer resides, indicated by a two-letter abbreviation; for example, OH or NJ
- **Account Length**: the number of days that this account has been active
- **Area Code**: the three-digit area code of the corresponding customer’s phone number
- **Int’l Plan**: whether the customer has an international calling plan: yes/no
- **VMail Plan**: whether the customer has a voice mail feature: yes/no
- **VMail Message**: the average number of voice mail messages per month
- **Day Mins**: the total number of calling minutes used during the day
- **Day Calls**: the total number of calls placed during the day
- **Day Charge**: the billed cost of daytime calls
- **Eve Mins**, **Eve Calls**, **Eve Charge**: the billed cost for calls placed during the evening
- **Night Mins**, **Night Calls**, **Night Charge**: the billed cost for calls placed during nighttime
- **Intl Mins**, **Intl Calls**, **Intl Charge**: the billed cost for international calls
- **CustServ Calls**: the number of calls placed to Customer Service
- **Churn?**: whether the customer left the service: true/false

## Proposed Approach

To address the Zabull Telecoms churn problem above, we aim to develop a predictive model that reliably detects customers inclined to churn. This model will provide the company with valuable insights, enabling them to adopt proactive measures such as tailored incentives, personalized services, or early intervention tactics. Since the model is for churn prediction, we are deploying the model in batches.

# 2. Technologies Used

Below are the technologies used in this project

## Cloud

- **AWS (Amazon Web Services):** AWS is used for cloud-based infrastructure, including hosting and scaling deployed models.
Amazon EC2: Amazon Elastic Compute Cloud (EC2) instances are utilized for virtual machines (VMs), supporting various project tasks.
Amazon S3: Amazon Simple Storage Service (S3) is employed for secure and scalable data storage.
Terraform: Terraform is used for Infrastructure as Code (IAC) to define, provision, and manage cloud resources in a declarative manner.


