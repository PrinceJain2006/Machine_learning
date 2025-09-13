# Machine_learning






## Lvl 1: clean data, pandas for data manipulation, mathplotlib/seaborn/plotly for visual, scikit-learn for basic model (ex: linear, logistic)
- load csv to pandas
- simple data exploratory analysis
- simple visualization
- handle missing values by drop/fill with means
- encode categorical features using one-hot encoding
- train models with default hyperparameters
- evaluate with basic metrics like accuracy

### Lvl 2: messy data, project structure: separate modules, git for version control, proper train/test split (ex: walk forward), class imbalance, models like GBM/NN/AI, hyperparameter tuning, bayesian search, simple pipeline (ex: prefect)
- build a customer churn prediction pipeline
- multiple data sources (ex: transaction records, support interactions, usage logs...)
- handle imbalance classes
- feature selection to identify the most predictive variables
- evaluating using precision recall curves, rock curves, business specific metrics

### Lvl 3: go from pure data science to ML engineering, models work in prod, containerize models using Docker, create APIs (ex: fastAPI, flask) to serve prediction, framework to simplify the process (ex: BentoML), load testing to ensure real user traffics, monitoring/logging/dashboard (ex: Grafana), versioning data and model (ex: DBC, ML flow, model registries)
- content recommendation engine for media platform
- model packaged in a docker container, deploy as a micro-service
- support both batch predictions (nightly) and real-time API (on-demand prediction)
- monitor metrics: click-through rates, latency percentiles, features distribution shifts
- employ shadow deployment, circuit breaker (if ML service  fail, system revert to simpler strategy)
- focus not only on accuracy, but also on inference latency, throughput, reliability

### Lvl 4: build robust, scalable ML system adapt to complex, everchanging environment, industrial scale, implement sophisticated solution, infra: cloud platform (AWS SageMaker, Google Vertex AI, Azure ML) for deployment/scaling, orchestration with Kurbernetes, workflow with Airflow/Prefect, deep ML framework like Pytorch/Tensorflow for custom models, optimization with quantization/knowledge distillation, fine tune massive model using loRA, experiments tracking, hyperparameter optimization (ex: Weights&Biases, MLflow), mixture of experts models, distributed training across GPU cluster (pipeline parallelism), feature stores, automated training pipeline triggered by data drift, monitring and A/B tesing frameworks
- real-time fraud detection for global financial institution
- advanced algo and ensemble methods to identify fraudulence with high recall and precision
- deploy on a cloud (ex: AWS)
- autoscaling groups to handle loads/latency
- monitor and alerts to detect system anomalies, model drift, ensure regulatory standards
- balance innovative research techniques with practical prod constraints => cutting-edge system and reliable
- push to research lvl innovation

### Lvl 5: ML system actively defining the future of AI, invent new approaches, custom NN and self-supervised learning systems, vast amount of unlabel data, novel applications of reinforcement learning, hybrid models combining symbolic reasoning with NN, design custom hardware accelerators to optimize performance
- build an autonomous scientific discovery system
- retrieval-augmented generation with neurosyolic reasoning to form and test hypothese in molecular biology
- use LLM fine-tuned to biomedical literature for hypothese generation
- employ symbolic logic modules to design experiments and test causual relationships
- reinforcement learning to simulate outcomes and optimize strategy
- proposing new compounds to test for open-ended tasks
