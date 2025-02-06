# **XGBoost Model Training and Deployment Using AWS SageMaker**

## **📌 Project Overview**
### **Problem Statement**
Predicting customer responses to a bank marketing campaign using **Amazon SageMaker's built-in XGBoost algorithm**. The dataset contains customer information and whether they subscribed to a bank product after a marketing campaign.

### **Solution Approach**
- This project was **implemented as a Notebook Instance in AWS SageMaker**.
- We **train an XGBoost model** on AWS SageMaker.
- Data is **uploaded to Amazon S3** for storage.
- **Hyperparameters are optimized** for better model accuracy.
- The trained model is **deployed as an endpoint** for predictions.
- The model predicts **whether a customer will subscribe (`yes` or `no`)**.

---

## **🛠️ Project Workflow**
1. **Set Up AWS SageMaker and S3**
2. **Prepare and Upload Data**
3. **Train XGBoost Model in SageMaker**
4. **Deploy the Model as an Endpoint**
5. **Make Predictions**

---

## **📂 Project Directory Structure**
```
AWS_SM_Project/
│── README.md                   # Project Documentation
│── AWS_SM.ipynb                 # Jupyter Notebook with Code
│── bank_clean.csv               # Dataset
│── train.csv                    # Processed Training Data
│── test.csv                     # Processed Test Data
```

---

## **1️⃣ Set Up AWS SageMaker and S3**
### **📌 Prerequisites**
- AWS Account
- IAM Role with permissions (`s3:ListBucket`, `s3:GetObject`, `s3:PutObject`, `sagemaker:*`)

### **🔹 Create an S3 Bucket**
```python
import boto3

bucket_name = 'bankapplication-eu'  # Must be globally unique
region = 'eu-central-1'

s3 = boto3.client('s3', region_name=region)
existing_buckets = [bucket['Name'] for bucket in s3.list_buckets()['Buckets']]
if bucket_name not in existing_buckets:
    s3.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={'LocationConstraint': region}
    )
```

### **🔹 Upload Data to S3**
```python
import pandas as pd
import urllib

# Download dataset
urllib.request.urlretrieve("https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.csv", "bank_clean.csv")

# Upload to S3
s3.upload_file("bank_clean.csv", bucket_name, "bank_clean.csv")
```

---

## **2️⃣ Prepare Data for XGBoost**
### **🔹 Split Data into Train & Test**
```python
import numpy as np

model_data = pd.read_csv("bank_clean.csv", index_col=0)
train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
```

### **🔹 Transform Data for SageMaker XGBoost**
XGBoost requires the target variable (`y_yes`) to be **the first column**:
```python
train_data = pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)], axis=1)
train_data.to_csv("train.csv", index=False, header=False)
```

### **🔹 Upload Processed Data to S3**
```python
import os
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join('xgboost-model', 'train/train.csv')).upload_file('train.csv')
```

---

## **3️⃣ Train XGBoost Model in SageMaker**
### **🔹 Create a SageMaker Estimator**
```python
import sagemaker
from sagemaker.estimator import Estimator

sagemaker_session = sagemaker.Session()
container = sagemaker.image_uris.retrieve("xgboost", region, "1.0-1")

estimator = Estimator(
    image_uri=container,
    hyperparameters={"max_depth": "5", "eta": "0.2", "gamma": "4", "subsample": "0.7", "objective": "binary:logistic", "num_round": 50},
    role=sagemaker.get_execution_role(),
    instance_count=1,
    instance_type='ml.m5.2xlarge',
    output_path=f's3://{bucket_name}/xgboost-model/output',
    sagemaker_session=sagemaker_session
)
```

### **🔹 Train the Model**
```python
s3_input_train = sagemaker.inputs.TrainingInput(s3_data=f's3://{bucket_name}/xgboost-model/train/', content_type='csv')
estimator.fit({'train': s3_input_train})
```

---

## **4️⃣ Deploy the Model as an Endpoint**
### **🔹 Deploy the Model**
```python
xgb_predictor = estimator.deploy(instance_type='ml.m5.large', initial_instance_count=1)
```

---

## **5️⃣ Make Predictions**
### **🔹 Send Data for Prediction**
```python
from sagemaker.serializers import CSVSerializer

test_data_array = test_data.drop(['y_no', 'y_yes'], axis=1).values
xgb_predictor.serializer = CSVSerializer()
xgb_predictor.content_type = 'text/csv'
predictions = xgb_predictor.predict(test_data_array).decode('utf-8')
```

---

## **🚀 SageMaker Libraries Used**
| Library | Description |
|---------|------------|
| **sagemaker.image_uris** | Retrieves built-in SageMaker container images. |
| **sagemaker.Session** | Manages SageMaker interactions and training jobs. |
| **sagemaker.inputs.TrainingInput** | Maps S3 training data to SageMaker training jobs. |
| **sagemaker.estimator.Estimator** | Defines and configures ML training jobs. |
| **sagemaker.serializers.CSVSerializer** | Converts input data into CSV format for inference. |

---

## **📌 Summary**
- **AWS SageMaker Notebook Instance** was used to train and deploy an XGBoost model.
- **S3 is used for data storage**, and model outputs are stored there.
- **Trained model is deployed as an endpoint** for real-time inference.
- **The model predicts customer responses** to a bank marketing campaign.

---

## **📜 License**
This project is licensed under the **MIT License**.

Let me know if you need further modifications! 🚀
