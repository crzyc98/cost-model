# Predictive Models

## Turnover Prediction

### TurnoverPredictor
- **Location**: `cost_model.ml.turnover.TurnoverPredictor`
- **Description**: Predicts employee turnover risk using machine learning
- **Key Methods**:
  - `predict()`: Generate turnover probabilities
  - `train()`: Train model on historical data
  - `evaluate()`: Evaluate model performance
- **Search Tags**: `class:TurnoverPredictor`, `ml:turnover`

### Feature Engineering
- **Location**: `cost_model.ml.features`
- **Description**: Creates predictive features from employee data
- **Key Features**:
  - Tenure-based features
  - Compensation ratios
  - Career progression metrics
  - Department/role statistics
- **Search Tags**: `module:features`, `ml:features`

## Model Training

### Training Pipeline
- **Location**: `cost_model.ml.pipeline`
- **Description**: End-to-end model training workflow
- **Components**:
  - Data loading and preprocessing
  - Feature engineering
  - Model training
  - Evaluation
  - Model serialization
- **Search Tags**: `module:pipeline`, `ml:training`

### Model Evaluation
- **Location**: `cost_model.ml.evaluation`
- **Description**: Model performance assessment
- **Metrics**:
  - ROC-AUC
  - Precision-Recall
  - Feature importance
  - SHAP values
- **Search Tags**: `module:evaluation`, `ml:metrics`

## Model Deployment

### Model Registry
- **Location**: `cost_model.ml.registry`
- **Description**: Manages model versions and metadata
- **Features**:
  - Version control
  - Model lineage
  - Performance tracking
  - Rollback capability
- **Search Tags**: `module:registry`, `ml:deployment`

### Batch Prediction
- **Location**: `cost_model.ml.batch`
- **Description**: Runs predictions on employee cohorts
- **Features**:
  - Scheduled runs
  - Incremental updates
  - Result caching
  - Error handling
- **Search Tags**: `module:batch`, `ml:predictions`

## Example Usage

### Training a New Model
```python
from cost_model.ml.pipeline import TrainingPipeline
from cost_model.ml.registry import ModelRegistry

# Initialize pipeline
pipeline = TrainingPipeline()

# Train model
model, metrics = pipeline.run(
    training_data="data/employee_history.csv",
    config="config/ml_config.yaml"
)

# Register model
registry = ModelRegistry()
version = registry.register_model(
    model=model,
    metrics=metrics,
    description="New turnover model v2.0"
)
```

### Making Predictions
```python
from cost_model.ml.registry import ModelRegistry
from cost_model.ml.batch import BatchPredictor

# Load production model
registry = ModelRegistry()
model = registry.load_production_model()

# Make batch predictions
predictor = BatchPredictor(model=model)
predictions = predictor.predict(
    employee_data="data/current_employees.csv",
    output_path="output/turnover_risk.csv"
)
```

## Model Monitoring

### Drift Detection
- **Location**: `cost_model.ml.monitoring`
- **Description**: Detects data and concept drift
- **Metrics**:
  - Feature distribution shifts
  - Prediction distribution changes
  - Performance degradation
- **Search Tags**: `module:monitoring`, `ml:drift`

### Alerting
- **Location**: `cost_model.ml.alerts`
- **Description**: Sends alerts for model issues
- **Channels**:
  - Email
  - Slack
  - PagerDuty
  - Custom webhooks
- **Search Tags**: `module:alerts`, `ml:monitoring`
