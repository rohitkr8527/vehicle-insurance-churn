# Vehicle Insurance Response Prediction

A comprehensive end-to-end machine learning system for predicting customer interest in vehicle insurance products. This project demonstrates advanced ML & MLOps practices with complete CI/CD pipeline, cloud deployment, and production-ready architecture.

---

##  Project Overview

This system predicts whether customers will be interested in purchasing vehicle insurance based on demographic and policy features. Built with modern MLOps practices, it includes data ingestion from MongoDB, model training with CatBoost, AWS S3 model storage, and FastAPI deployment.

### Key Features

* **End-to-end MLOps Pipeline:** Complete workflow from data ingestion to model deployment
* **Production-Ready Architecture:** Modular design with proper separation of concerns
* **Cloud Integration:** AWS S3 for model storage and MongoDB for data management
* **Advanced ML Techniques:** Class imbalance handling with SMOTE-ENN, hyperparameter tuning, CatBoost for model training
* **Web Interface:** FastAPI application with HTML frontend for real-time predictions
* **CI/CD Pipeline:** Automated testing and deployment with GitHub Actions
* **Containerization:** Docker support for consistent deployment environments

---

## Dataset Information

* **Size:** 381,109 records with 12 features
* **Target:** Binary classification (`Response`: 1 = Interested, 0 = Not Interested)
* **Class Distribution:** Highly imbalanced (87.7% negative, 12.3% positive)

### Features

* **Demographics:** Gender, Age, Driving License status
* **Geographic:** Region Code
* **Historical:** Previously Insured, Vehicle Age, Vehicle Damage history
* **Financial:** Annual Premium amount
* **Operational:** Policy Sales Channel, Vintage (customer relationship duration)

---

##  Architecture

```
.
├── src/
│   ├── components/          # Core ML pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── model_pusher.py
│   ├── pipline/             # Training and prediction pipelines
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   ├── entity/              # Configuration and artifact entities
│   ├── cloud_storage/       # AWS S3 integration
│   ├── configuration/       # Configuration management
│   ├── constants/           # Project constants
│   ├── exception/           # Custom exception handling
│   ├── logger/              # Logging utilities
│   └── utils/               # Utility functions
├── config/                  # Configuration files
├── notebook/                # Jupyter notebooks for EDA
├── templates/               # HTML templates
├── static/                  # Static files (CSS, JS)
├── .github/workflows/       # CI/CD pipeline
├── app.py                   # FastAPI application
├── Dockerfile               # Container configuration
└── requirements.txt         # Dependencies
```

---

## Installation & Setup

### Prerequisites

* Python 3.8+
* MongoDB instance
* AWS account (for S3 storage)

### Local Development Setup

**Clone the repository**

```bash
git clone https://github.com/rohitkr8527/vehicle-insurance-churn.git
cd vehicle-insurance-churn
```

**Create virtual environment**

```bash
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

**Install dependencies**

```bash
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file with the following variables:

```bash
MONGODB_URL=your_mongodb_connection_string
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
```

### Run the application

```bash
python app.py
```

The application will be available at **[http://localhost:5000](http://localhost:5000)**

---

## Docker Deployment

**Build Docker image**

```bash
docker build -t vehicle-insurance-predictor .
```

**Run container**

```bash
docker run -p 5000:5000 --env-file .env vehicle-insurance-predictor
```

---

## Model Performance

**Final Model:** CatBoost Classifier

| Metric            | Value |
| ----------------- | ----: |
| ROC-AUC           | 0.854 |
| Precision         | 0.306 |
| Recall            | 0.844 |
| F1-Score          | 0.449 |
| Optimal Threshold |  0.65 |

### Model Configuration

* **Algorithm:** CatBoost with class weight balancing
* **Hyperparameters:** 400 iterations, depth = 8, learning\_rate = 0.1
* **Class Imbalance Handling:** SMOTE-ENN for balanced training
* **Validation:** 5-fold stratified cross-validation

### Performance Analysis

* **High Recall (84.4%):** Captures most interested customers, crucial for business
* **Balanced F1-Score:** Optimized threshold balances precision and recall
* **ROC-AUC (85.4%):** Excellent discrimination between classes
* **Business Impact:** Reduces marketing costs while maximizing customer acquisition

---

## Usage

### Web Interface

1. Navigate to **[http://localhost:5000](http://localhost:5000)**
2. Fill in customer information form
3. Click **Predict** to get insurance interest probability
4. View result as **Response-Yes** or **Response-No**

### API Endpoints

**Health Check**

```bash
GET /
```

**Make Prediction**

```bash
POST /
# Content-Type: application/x-www-form-urlencoded
gender=1&age=35&driving_license=1&region_code=28&previously_insured=0&annual_premium=30000&policy_sales_channel=152&vintage=100&vehicle_age_lt_1_year=0&vehicle_age_gt_2_years=1&vehicle_damage_yes=1
```

### Programmatic Usage

```python
from src.pipline.prediction_pipeline import VehicleData, VehicleDataClassifier

# Create prediction data
vehicle_data = VehicleData(
    Gender=1, Age=35, Driving_License=1, Region_Code=28.0,
    Previously_Insured=0, Annual_Premium=30000, Policy_Sales_Channel=152.0,
    Vintage=100, Vehicle_Age_lt_1_Year=0, Vehicle_Age_gt_2_Years=1,
    Vehicle_Damage_Yes=1
)

# Make prediction
classifier = VehicleDataClassifier()
df = vehicle_data.get_vehicle_input_data_frame()
prediction = classifier.predict(df)
print(f"Prediction: {prediction}")  # 1: Interested, 0: Not Interested
```

---

## MLOps Pipeline

### Training Pipeline

1. **Data Ingestion:** Fetch data from MongoDB
2. **Data Validation:** Schema validation and data quality checks
3. **Data Transformation:** Feature engineering and preprocessing
4. **Model Training:** CatBoost training with hyperparameter tuning
5. **Model Evaluation:** Performance assessment against baseline
6. **Model Deployment:** Push to AWS S3 if performance threshold met

### Prediction Pipeline

1. **Data Preprocessing:** Apply same transformations as training
2. **Model Loading:** Fetch latest model from S3
3. **Prediction:** Generate probability scores
4. **Post-processing:** Apply optimal threshold for binary classification

### Monitoring & Maintenance

* Model versioning in AWS S3
* Performance tracking across deployments
* Automated retraining triggers
* Data drift detection capabilities

---

## Business Impact

### Key Business Metrics

* **Customer Acquisition Cost:** Reduced by targeting high-probability prospects
* **Marketing Efficiency:** 84% recall ensures minimal missed opportunities
* **Resource Optimization:** Focus sales efforts on interested customers
* **Revenue Impact:** Improved conversion rates through better targeting

### Use Cases

* **Marketing Campaign Optimization:** Target customers likely to purchase
* **Sales Resource Allocation:** Prioritize high-probability leads
* **Customer Segmentation:** Identify distinct customer behavior patterns
* **Product Development:** Understand factors influencing insurance interest

---

## Technology Stack

### Core Technologies

* **ML Framework:** CatBoost, scikit-learn, imbalanced-learn
* **Web Framework:** FastAPI, Jinja2 templates
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn, Plotly
* **Cloud Services:** AWS S3, MongoDB

### MLOps Tools

* **Containerization:** Docker
* **CI/CD:** GitHub Actions
* **Model Storage:** AWS S3 with versioning
* **Configuration:** YAML-based config management
* **Logging:** Python logging with custom handlers

### Development Tools

* **Version Control:** Git with conventional commits
* **Environment Management:** Virtual environments
* **Dependency Management:** `pip` with `requirements.txt`
* **Code Quality:** Type hints, docstrings, error handling

---

## Future Enhancements

### Technical Improvements

* Model interpretability with SHAP values
* Real-time model monitoring and alerting
* A/B testing framework for model comparison
* Multi-model ensemble predictions

### Business Features

* Customer lifetime value prediction
* Churn risk scoring integration
* Dynamic pricing recommendations
* Marketing campaign ROI tracking
* Regulatory compliance reporting

### Infrastructure

* Kubernetes deployment for scalability
* Stream processing for real-time predictions
* Model serving with TensorFlow Serving
* Automated data pipeline with Apache Airflow
* Multi-cloud deployment strategy

---

## Contact & Support

For questions, suggestions, or collaboration opportunities:

* **GitHub:** `rohitkr8527`
* **LinkedIn:** `https://www.linkedin.com/in/rohitkmr8527/`
* **Email:** `rohitkr8527@gmail.com`

---

> ⭐ **Star this repository** if you found it helpful! Contributions and feedback are always welcome.
