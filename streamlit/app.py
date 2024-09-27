import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from utils.supportFunc import get_cleaned_data, sidebar_input, hourly_rentals_plot, monthly_rentals, \
    rental_plot_by_week, day_hour_melted_pivot_df, plot_day_hour_rentals, weather_impact_on_rental, \
        seasonal_plot


from bikeshare.utils.config import Config
from bikeshare.configs.config import CFGLog
from bikeshare.executor.inferrer import Inferrer  

import warnings
warnings.filterwarnings('ignore')


def main():
    # set page config
    st.set_page_config(
        page_title="Bikeshare Prediction",
        page_icon=":bike:",
        layout="centered",
        initial_sidebar_state="collapsed", 
        menu_items={
            "About": "https://github.com/Surya96t/ml-bikeshare-streamlit"
        }
    )
    
    st.image("streamlit/assets/bikshare_gen_logo.png", use_column_width="always")
    
    
    config = Config.from_json(CFGLog)
    dt_model = config.output.output_path + config.output.dt_path + config.output.dt_model
    rf_model = config.output.output_path + config.output.rf_path + config.output.rf_model
    xgb_model = config.output.output_path + config.output.xgb_path + config.output.xgb_model

    data = get_cleaned_data()
    input_data = sidebar_input()

    home_tab, dt_tab, rf_tab, xgb_tab = st.tabs(["Home", "Decision Tree", "Random Forest", "XGBoost"])
    input_df = pd.DataFrame([input_data])
    model_cols = config.data.X

    # Inferrer
    inferrer = Inferrer()

    with home_tab:
        st.title("Welcome to the Bikeshare Prediction App!")
        st.write("`To generate predictions, enter your data in the sidebar and explore the model tabs.`")
        st.divider()
        
        st.header("Exploratory Data Analysis")
        st.write("### Introduction")
        st.write("""
                 Welcome to the Exploratory Data Analysis (EDA) section of this project. The goal of 
                 EDA is to gain initial insights into the dataset that will be used for building 
                 predictive models. By exploring the structure, characteristics, and relationships 
                 within the data, we can better understand key patterns and ensure the dataset is 
                 ready for modeling. In this section, we will walk through various aspects of the 
                 dataset, including its statistical properties,feature distributions, correlations,
                 and any missing data that may need attention.""")
        
        # Display the DataFrame
        st.divider()
        st.subheader("Dataset Overview:")
        st.dataframe(data.head())
        st.write(f"The dataset has the **{data.shape[0]}** rows and **{data.shape[1]}** columns.")

        # Statistical Summary
        st.divider()
        st.subheader("Statistical Summary:")
        st.write(data.describe())
        
        # Rental Comparisons 
        st.divider()
        st.subheader("Rentals Comparision:")
        start_date = datetime.date(2017, 12, 1)
        end_date = datetime.date(2018, 11, 30)
        d = st.date_input("Select a date", value=start_date, min_value=start_date, max_value=end_date)
        st.write("The date you selected is:", d)
        col1, col2 = st.columns(2)
        with col1:
            # Hourly Rentals Plot
            st.write("##### Hourly Rentals")
            hourly_rentals_plot(d.day, d.month, d.year, data, avg_rentals=True)
        
        with col2:
            st.write("##### Montly Rentals")
            merge_df = monthly_rentals(d.month, d.year, data)
        
        # Weekly Rentals Distribution
        st.divider()
        st.subheader("Rentals Distribution by Week")
        rental_plot_by_week(merge_df)
        
        # Rental trips combined plot
        st.divider()
        st.subheader("Rentals Trips by Day and Hour")
        pivot, melted = day_hour_melted_pivot_df(data)
        plot_day_hour_rentals(pivot, melted)
        
        # Weather impact on rentals
        st.divider()
        st.subheader("Weather Impact") 
        st.write("Select a weather feature to plot and analyze its impact on bike rentals.")
        weather_choice = st.selectbox("Type", data.columns[3:11])
        weather_impact_on_rental(data, weather_choice)
        
        # Seasonal Impact
        st.divider()
        st.subheader("Seasonal Impact")
        seasonal_plot(data)
        
        st.divider()
        st.write("##### More details about the EDA process can be found in the notebook folder in the \
            [GitHub repository](https://github.com/Surya96t/ml-bikeshare-streamlit) \
                or at the [Kaggle dataset page](https://www.kaggle.com/code/suryapm/bikeshare-analysis-model-comparision).")
        
    with dt_tab:
        st.title("Decision Tree Model")
        dt_metrics_path = config.output.output_path + config.output.dt_path + "DecisionTree_metrics.json"
        dt_metrics = pd.read_json(dt_metrics_path)
        dt_model_parameters = config.decision_tree
        
        
        dt_col1, dt_col2, dt_col3 = st.columns([1,1,2])
        with dt_col1:
            st.write("Number of bikes that are to be rented:")
            if st.button("DT Prediction"):
                st.write(inferrer.dt_infer(input_df))
            
            
        with dt_col2:
            st.write("Model Metrics")
            st.dataframe(dt_metrics)
        
        with dt_col3:
            st.write("Model Parameters")
            dt_params  = {
                'criterion': dt_model_parameters.criterion,
                'max_depth': dt_model_parameters.max_depth,
                'max_features': dt_model_parameters.max_features,
                'max_leaf_nodes': dt_model_parameters.max_leaf_nodes,
                'min_samples_leaf': dt_model_parameters.min_samples_leaf,
                'min_samples_split': dt_model_parameters.min_samples_split,
                'splitter': dt_model_parameters.splitter
            }
               
            st.write(dt_params)
        
        st.write("#### Feature Importance")
        dt_importance_df = pd.DataFrame(
                {
                    'Column': inferrer.get_col_names_after_transform(),
                    'DT Feature Importance': inferrer.dt_feature_importance()
                }
            )
        st.bar_chart(dt_importance_df, x='Column', y='DT Feature Importance', horizontal=True)
        
        
    with rf_tab:
        st.title("Random Forest Model")
        rf_metrics_path = config.output.output_path + config.output.rf_path + "RandomForest_metrics.json"
        rf_metrics = pd.read_json(rf_metrics_path)
        rf_model_parameters = config.random_forest
        
        rf_col1, rf_col2, rf_col3= st.columns([1,1,2])
        with rf_col1:
            st.write("Number of bikes that are to be rented:")
            if st.button("RF Prediction"):
                st.write(inferrer.rf_infer(input_df))
            
        with rf_col2:
            st.write("Model Metrics")
            st.dataframe(rf_metrics)
        
        with rf_col3:
            st.write("Model Parameters")
            rf_params = {
                'n_estimators': rf_model_parameters.n_estimators,
                'max_depth': rf_model_parameters.max_depth,
                'max_features': rf_model_parameters.max_features,
                'min_samples_leaf': rf_model_parameters.min_samples_leaf,
                'min_samples_split': rf_model_parameters.min_samples_split
            }
            
            st.write(rf_params)
            
        rf_importance_df = pd.DataFrame(
            {
                'Column': inferrer.get_col_names_after_transform(),
                'RF Feature Importance': inferrer.rf_feature_importance()
            }
        )
        st.bar_chart(rf_importance_df, x='Column', y='RF Feature Importance', horizontal=True)
            
    with xgb_tab:
        st.title("XGBoost Model")
        xgb_metrics_path = config.output.output_path + config.output.xgb_path + "XGBoost_metrics.json"
        xgb_metrics = pd.read_json(xgb_metrics_path)
        xgb_model_parameters = config.gradient_boosting
        
        
        xgb_col1, xgb_col2, xgb_col3 = st.columns([1,1,2])
        with xgb_col1:
            st.write("Number of bikes that are to be rented:")
            if st.button("XGB Prediction"):
                st.write(inferrer.xgb_infer(input_df))
        
        with xgb_col2:
            st.write("Model Metrics")
            st.dataframe(xgb_metrics)
            
        with xgb_col3:
            st.write("Model Parameters")
            xgb_params = {
                'n_estimators': xgb_model_parameters.n_estimators,
                'max_depth': xgb_model_parameters.max_depth,
                'subsample': xgb_model_parameters.subsample,
                'learning_rate': xgb_model_parameters.learning_rate
            }
            
            st.write(xgb_params)
            
        xgb_importance_df = pd.DataFrame(
            {
                'Column': inferrer.get_col_names_after_transform(),
                'XGB Feature Importance': inferrer.xgb_feature_importance()
            }
        )
        st.bar_chart(xgb_importance_df, x='Column', y='XGB Feature Importance', horizontal=True)
    
if __name__ == "__main__":
    main()
    
    
    
