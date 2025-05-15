import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(page_title="Online Shoppers Intention App", layout="wide")

# Function to load the saved models
@st.cache_resource
def load_models():
    try:
        # Load Random Forest model
        rf_model = joblib.load('random_forest_model.pkl')
        
        # Load XGBoost model
        xgb_model = joblib.load('xgboost_model.pkl')
        
        # Load scaler
        scaler = joblib.load('scaler.pkl')
        
        return rf_model, xgb_model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# Try loading the dataset for reference column structure
try:
    reference_df = pd.read_csv('online_shoppers_intention.csv', nrows=5)
    has_dataset = True
except:
    has_dataset = False
    st.warning("Reference dataset 'online_shoppers_intention.csv' not found. Some functionality may be limited.")

# Load models and scaler
try:
    rf_model, xgb_model, scaler = load_models()
    models_loaded = all(model is not None for model in [rf_model, xgb_model, scaler])
    
    # Debug information for scaler
    if models_loaded and has_dataset:
        st.sidebar.write("Models loaded successfully!")
        if hasattr(scaler, 'feature_names_in_'):
            # Store feature names for reference
            scaler_features = scaler.feature_names_in_
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.error("Please make sure 'random_forest_model.pkl', 'xgboost_model.pkl', and 'scaler.pkl' are in the same directory.")
    models_loaded = False

# Streamlit App
st.title('Online Shoppers Intention Prediction App')
st.write("This app predicts whether a visitor will generate revenue based on session attributes.")

# Define a session state to remember tab selections
if 'tab_selected' not in st.session_state:
    st.session_state.tab_selected = 0

# Create tabs for prediction and visualization
tabs = ['Single Prediction', 'Visualize Data', 'Batch Prediction']
selected_tab = st.radio('Select Tab:', tabs, index=st.session_state.tab_selected)

# Tab selection logic
if selected_tab != st.session_state.tab_selected:
    st.session_state.tab_selected = tabs.index(selected_tab)

# Month mapping (for user-friendly display)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
visitor_types = ['New_Visitor', 'Returning_Visitor', 'Other']

# Function to preprocess input data
def preprocess_input(input_df, is_single=True):
    # Create a copy to avoid modifying the original
    df_processed = input_df.copy()
    
    # Get the expected column structure from a sample of the training data
    try:
        sample_df = pd.read_csv('online_shoppers_intention.csv', nrows=1)
        sample_df = sample_df.drop('Revenue', axis=1)
        
        # Store original column names before one-hot encoding for reference
        original_cat_columns = ['Month', 'VisitorType']
        
        # Get all possible values for categorical columns from training data
        all_months = pd.read_csv('online_shoppers_intention.csv')['Month'].unique()
        all_visitor_types = pd.read_csv('online_shoppers_intention.csv')['VisitorType'].unique()
    except:
        # Fallback if file not found
        all_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'June', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        all_visitor_types = ['New_Visitor', 'Returning_Visitor', 'Other']
    
    # Process Month column - create dummy variables
    if 'Month' in df_processed.columns:
        # First, make sure all months use the exact same format as the training data
        df_processed['Month'] = df_processed['Month'].astype(str)
        # Create dummies manually to ensure all expected columns are present
        for month in all_months:
            col_name = f'Month_{month}'
            if month == all_months[0]:  # Skip first month (reference category) for drop_first=True
                continue
            df_processed[col_name] = (df_processed['Month'] == month).astype(int)
        # Remove original column
        df_processed = df_processed.drop('Month', axis=1)
    
    # Process VisitorType column - create dummy variables
    if 'VisitorType' in df_processed.columns:
        # First, make sure all visitor types use the exact same format
        df_processed['VisitorType'] = df_processed['VisitorType'].astype(str)
        # Create dummies manually to ensure all expected columns are present
        for vtype in all_visitor_types:
            col_name = f'VisitorType_{vtype}'
            if vtype == all_visitor_types[0]:  # Skip first type (reference category) for drop_first=True
                continue
            df_processed[col_name] = (df_processed['VisitorType'] == vtype).astype(int)
        # Remove original column
        df_processed = df_processed.drop('VisitorType', axis=1)
    
    # Get the exact column names from the scaler
    scaler_feature_names = scaler.feature_names_in_
    
    # Ensure all columns from the scaler are present in the processed dataframe
    for col in scaler_feature_names:
        if col not in df_processed.columns:
            df_processed[col] = 0
    
    # Make sure we only include columns that the scaler expects and in the correct order
    df_processed = df_processed[scaler_feature_names]
    
    # Scale the data
    scaled_data = scaler.transform(df_processed)
    
    return scaled_data

# Function to make predictions
def predict(input_data, model_type):
    # Choose the model based on the selection
    if model_type == 'Random Forest':
        model = rf_model
    else:  # 'XGBoost'
        model = xgb_model
    
    # Make prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    
    return prediction, probability

if models_loaded:
    # Tab 1: Single Prediction
    if st.session_state.tab_selected == 0:
        st.header('Single Session Prediction')
        
        # Create two columns for input fields
        col1, col2 = st.columns(2)
        
        with col1:
            # Numerical inputs
            administrative = st.slider('Administrative Pages', 0, 50, 0)
            administrative_duration = st.slider('Administrative Duration (sec)', 0.0, 3000.0, 0.0)
            informational = st.slider('Informational Pages', 0, 50, 0)
            informational_duration = st.slider('Informational Duration (sec)', 0.0, 3000.0, 0.0)
            product_related = st.slider('Product Related Pages', 0, 200, 0)
            product_related_duration = st.slider('Product Related Duration (sec)', 0.0, 10000.0, 0.0)
            bounce_rates = st.slider('Bounce Rate', 0.0, 1.0, 0.0)
            exit_rates = st.slider('Exit Rate', 0.0, 1.0, 0.0)
            
        with col2:
            page_values = st.slider('Page Values', 0.0, 500.0, 0.0)
            special_day = st.slider('Special Day', 0.0, 1.0, 0.0)
            month = st.selectbox('Month', months)
            operating_system = st.slider('Operating System', 1, 8, 1)
            browser = st.slider('Browser', 1, 13, 1)
            region = st.slider('Region', 1, 9, 1)
            traffic_type = st.slider('Traffic Type', 1, 20, 1)
            visitor_type = st.selectbox('Visitor Type', visitor_types)
            weekend = st.checkbox('Weekend')
            
        # Create a DataFrame for the user input
        user_input = pd.DataFrame({
            'Administrative': [administrative],
            'Administrative_Duration': [administrative_duration],
            'Informational': [informational],
            'Informational_Duration': [informational_duration],
            'ProductRelated': [product_related],
            'ProductRelated_Duration': [product_related_duration],
            'BounceRates': [bounce_rates],
            'ExitRates': [exit_rates],
            'PageValues': [page_values],
            'SpecialDay': [special_day],
            'Month': [month],
            'OperatingSystems': [operating_system],
            'Browser': [browser],
            'Region': [region],
            'TrafficType': [traffic_type],
            'VisitorType': [visitor_type],
            'Weekend': [weekend]
        })
        
        # Select model
        model_type = st.radio('Select Model:', ['Random Forest', 'XGBoost'])
        
        if st.button('Predict'):
            try:
                # Display debug info
                st.write("Preprocessing input data...")
                
                # Preprocess input data
                processed_input = preprocess_input(user_input)
                
                st.write("Making prediction...")
                # Make prediction
                prediction, probability = predict(processed_input, model_type)
                
                # Display result
                st.subheader('Prediction Result:')
                
                # Display prediction with gauge
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"### {'Revenue: Yes' if prediction[0] else 'Revenue: No'}")
                    st.markdown(f"#### Confidence: {probability[0][1]*100:.2f}%" if prediction[0] else f"#### Confidence: {probability[0][0]*100:.2f}%")
                
                with col2:
                    # Create a gauge-like visualization
                    fig, ax = plt.subplots(figsize=(4, 0.3))
                    colors = ['#ff9999', '#66b3ff'] if prediction[0] else ['#66b3ff', '#ff9999']
                    ax.barh([0], [probability[0][1]], color=colors[0], height=0.3)
                    ax.barh([0], [probability[0][0]], left=[probability[0][1]], color=colors[1], height=0.3)
                    ax.set_xlim(0, 1)
                    ax.set_ylim(-0.5, 0.5)
                    ax.axis('off')
                    st.pyplot(fig)
                
                # Show feature importance for the selected model
                st.subheader('Feature Importance:')
                
                if model_type == 'Random Forest':
                    importances = rf_model.feature_importances_
                else:  # XGBoost
                    importances = xgb_model.feature_importances_
                
                # Get feature names from the model
                if hasattr(scaler, 'feature_names_in_'):
                    feature_names = scaler.feature_names_in_
                else:
                    # Fallback if feature names not available
                    feature_names = [f"Feature_{i}" for i in range(len(importances))]
                
                # Create DataFrame with feature importance
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Plot top 10 features
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='viridis')
                plt.title(f'Top 10 Feature Importance - {model_type}')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.error("Please check that the input data is formatted correctly and the models are loaded properly.")
            
    # Tab 2: Visualize Data
    elif st.session_state.tab_selected == 1:
        st.header('Data Visualization')
        
        # Load dataset
        try:
            df = pd.read_csv('online_shoppers_intention.csv')
            
            # Select visualization type
            viz_type = st.selectbox('Select Visualization Type:', 
                                    ['Feature Distribution', 'Correlation Matrix', 'Revenue by Feature'])
            
            if viz_type == 'Feature Distribution':
                # Select feature for visualization
                numeric_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                feature_for_viz = st.selectbox('Select Feature:', numeric_features)
                
                # Create distribution plot
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df[feature_for_viz], kde=True, hue=df['Revenue'], palette='viridis')
                plt.title(f'Distribution of {feature_for_viz} by Revenue')
                st.pyplot(fig)
                
                # Display statistics
                st.subheader('Statistics:')
                stats_df = df.groupby('Revenue')[feature_for_viz].describe()
                st.write(stats_df)
                
            elif viz_type == 'Correlation Matrix':
                # Create correlation matrix for numeric features
                corr_df = df.select_dtypes(include=['float64', 'int64', 'bool']).corr()
                
                # Plot correlation matrix
                fig, ax = plt.subplots(figsize=(12, 10))
                sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
                plt.title('Correlation Matrix')
                st.pyplot(fig)
                
            elif viz_type == 'Revenue by Feature':
                # Select feature for visualization
                cat_features = ['Month', 'VisitorType', 'Weekend', 'OperatingSystems', 'Browser', 'Region', 'TrafficType']
                feature_for_viz = st.selectbox('Select Feature:', cat_features)
                
                # Create count plot
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.countplot(x=feature_for_viz, hue='Revenue', data=df, palette='viridis')
                plt.title(f'Revenue by {feature_for_viz}')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
                # Calculate conversion rates
                conversion_df = df.groupby(feature_for_viz)['Revenue'].mean().reset_index()
                conversion_df['Conversion Rate (%)'] = conversion_df['Revenue'] * 100
                conversion_df = conversion_df.rename(columns={feature_for_viz: 'Category'})
                
                # Plot conversion rates
                fig, ax = plt.subplots(figsize=(12, 6))
                sns.barplot(x='Category', y='Conversion Rate (%)', data=conversion_df, palette='viridis')
                plt.title(f'Conversion Rate by {feature_for_viz}')
                plt.xticks(rotation=45)
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error loading or visualizing data: {e}")
            st.write("Please make sure the file 'online_shoppers_intention.csv' is in the same directory.")
            
    # Tab 3: Batch Prediction
    elif st.session_state.tab_selected == 2:
        st.header('Batch Prediction from CSV')
        
        # Upload CSV file
        uploaded_file = st.file_uploader("Upload CSV file (must have same format as training data)", type=["csv"])
        
        if uploaded_file is not None:
            try:
                # Read CSV file
                test_df = pd.read_csv(uploaded_file)
                
                # Show a preview of the data
                st.subheader('Data Preview:')
                st.write(test_df.head())
                
                # Check if 'Revenue' column exists and handle accordingly
                predict_df = test_df.copy()
                has_revenue_column = 'Revenue' in predict_df.columns
                
                if has_revenue_column:
                    actual_revenue = predict_df['Revenue'].copy()
                    predict_df = predict_df.drop('Revenue', axis=1)
                
                # Select model for prediction
                model_type = st.radio('Select Model for Prediction:', ['Random Forest', 'XGBoost'])
                
                if st.button('Run Prediction'):
                    # Preprocess input data
                    processed_input = preprocess_input(predict_df, is_single=False)
                    
                    # Make prediction
                    prediction, probability = predict(processed_input, model_type)
                    
                    # Add predictions to the original dataframe
                    result_df = test_df.copy()
                    result_df['Predicted_Revenue'] = prediction
                    result_df['Revenue_Probability'] = [prob[1] for prob in probability]
                    
                    # Display results
                    st.subheader('Prediction Results:')
                    st.write(result_df)
                    
                    # If actual revenue is available, show model evaluation
                    if has_revenue_column:
                        st.subheader('Model Evaluation:')
                        
                        # Calculate accuracy
                        accuracy = (prediction == actual_revenue).mean()
                        st.write(f"Accuracy: {accuracy:.4f}")
                        
                        # Plot confusion matrix
                        from sklearn.metrics import confusion_matrix
                        cm = confusion_matrix(actual_revenue, prediction)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                    xticklabels=['No Revenue', 'Revenue'],
                                    yticklabels=['No Revenue', 'Revenue'])
                        plt.xlabel('Predicted')
                        plt.ylabel('Actual')
                        plt.title(f'Confusion Matrix - {model_type}')
                        st.pyplot(fig)
                        
                        # Plot ROC curve
                        from sklearn.metrics import roc_curve, auc
                        fpr, tpr, _ = roc_curve(actual_revenue, [prob[1] for prob in probability])
                        roc_auc = auc(fpr, tpr)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        plt.xlim([0.0, 1.0])
                        plt.ylim([0.0, 1.05])
                        plt.xlabel('False Positive Rate')
                        plt.ylabel('True Positive Rate')
                        plt.title('Receiver Operating Characteristic')
                        plt.legend(loc="lower right")
                        st.pyplot(fig)
                    
                    # Visualize results
                    st.subheader('Visualize Predictions:')
                    
                    # Select feature for visualization
                    viz_features = test_df.columns.tolist()
                    if 'Revenue' in viz_features:
                        viz_features.remove('Revenue')
                    
                    feature_for_viz = st.selectbox('Select Feature for Visualization:', viz_features)
                    
                    # Create visualization based on the feature type
                    if result_df[feature_for_viz].dtype in ['int64', 'float64'] and len(result_df[feature_for_viz].unique()) > 10:
                        # For numerical features - scatter plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.scatterplot(x=feature_for_viz, y='Revenue_Probability', 
                                       hue='Predicted_Revenue', data=result_df, palette='viridis')
                        plt.title(f'{feature_for_viz} vs Revenue Probability')
                        st.pyplot(fig)
                    else:
                        # For categorical features - count plot
                        fig, ax = plt.subplots(figsize=(12, 6))
                        sns.countplot(x=feature_for_viz, hue='Predicted_Revenue', data=result_df, palette='viridis')
                        plt.title(f'Predicted Revenue by {feature_for_viz}')
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
                    
                    # Download predictions
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.write("Make sure your CSV has the same format as the training data.")
else:
    st.write("Please ensure all model files are available before using this application.")
    
# Add footer
st.markdown("---")
st.markdown("Online Shoppers Intention Prediction App - Made with Streamlit")