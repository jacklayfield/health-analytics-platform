# HeartHealthPredictor
A project deisgned to predict the likelihood and risk of heart disease

### Objective
Create an application that ingests and pre-processes data to be used to train machine learning models. There predictions should be displayed in a simple web applicatioon.

### Basic project outline: 

```
├── data_preprocessing/ 
│   ├── data/ 
│   ├── scripts/ 
│   ├── notebooks/ 
│
├── model_training/ 
│   ├── models/ 
│   ├── training/ 
│   ├── notebooks/
│ 
├── web_app/
│   ├── backend/
│   ├── frontend/
│   ├── static/ 
```

What we will use: 

- Terraform for provisioning cloud resources
- Amazon SageMaker or Airflow for ML pipeline orchestration (Haven't decided yet)
- Apache Spark for data processing