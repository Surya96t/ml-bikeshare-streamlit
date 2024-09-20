# ml-bikeshare-streamlit

The project aims to develop a predictive model for bike rental demand in Seoul, addressing operational inefficiencies due to fluctuating demand. The objective is to create a robust model to help the rental company optimize fleet distribution. Expected models include Linear Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, XGBoost Regressor, and Neural Networks. Success criteria focus on accuracy, generalization, feature importance, and operational feasibility.


### File Folder Setup
.
├── LICENSE
├── README.md
├── Dockerfile
├── app.yaml
├── pyproject.toml
├── setup.cfg
├── setup.py
├── requirements.txt
├── data
│   ├── exported_models
│   └── yelp.csv
├── notebooks
├── scripts
│   └── main.py
├── streamlit
│   └── app.py
├── tests
└── src
    ├── __init__.py
    ├── configs
    │   ├── __init__.py
    │   └── config.py
    ├── dataloader
    │   ├── __init__.py
    │   └── dataloader.py
    ├── executor
    │   ├── __init__.py
    │   ├── inferrer.py
    │   └── trainer.py
    ├── model
    │   ├── __init__.py
    │   ├── base_model.py
    │   └── yelp_model.py
    └── utils
        ├── __init__.py
        ├── config.py
        └── postprocessing.py