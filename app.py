import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
import streamlit as st
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import pickle
import os
import plotly.express as px  # For interactive plots
from PIL import Image # Import PIL to handle images

warnings.filterwarnings('ignore')

try:  # Add
    @st.cache_data
    def load_and_preprocess_data(file_path):
        """Loads and preprocesses the hotel booking data."""
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            st.error(f"Error: The file '' was not found.")
            return None
        except Exception as e:
            st.error(f"Error loading data: ")
            return None

        # --- Data Cleaning ---
        try:
            df['children'].fillna(df['children'].median(), inplace=True)
            df['country'].fillna(df['country'].mode()[0], inplace=True)
            df['agent'].fillna(0, inplace=True)
            df.drop('company', axis=1, inplace=True)
            df['children'] = df['children'].astype('int64')
            df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
            invalid_bookings = (df['adults'] + df['children'] + df['babies']) <= 0
            df = df[~invalid_bookings]
            df['meal'] = df['meal'].replace('Undefined', 'SC')
            df = df[~df['market_segment'].isin(['Undefined'])]
            df = df[~df['distribution_channel'].isin(['Undefined'])]
            df = df[df['adr'] != 0]
        except Exception as e:
            st.error(f"Error during data cleaning: ")
            return None

        # --- Feature Engineering ---
        try:
            df['total_stay_duration'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
            df['total_guests'] = df['adults'] + df['children'] + df['babies']
            df['adr_per_person'] = df['adr'] / df['total_guests']
            df['adr_per_person'] = df['adr_per_person'].replace([float('inf'), float('-inf')], 0)
            df['adr_per_person'] = df['adr_per_person'].fillna(0)
            df['room_type_match'] = (df['reserved_room_type'] == df['assigned_room_type']).astype(int)
        except Exception as e:
            st.error(f"Error during feature engineering: ")
            return None

        # --- Encoding Categorical Features ---
        try:
            columns_to_one_hot = ['hotel', 'meal', 'market_segment', 'distribution_channel', 'deposit_type',
                                    'customer_type']
            df = pd.get_dummies(df, columns=columns_to_one_hot, drop_first=True)

            label_encoder = LabelEncoder()
            columns_to_label_encode = ['arrival_date_month', 'reserved_room_type', 'assigned_room_type']
            for column in columns_to_label_encode:
                df[column] = label_encoder.fit_transform(df[column])
        except Exception as e:
            st.error(f"Error during encoding: ")
            return None

        # --- Removing Columns ---
        try:
            columns_to_remove = ['country', 'reservation_status', 'reservation_status_date']
            df.drop(columns=columns_to_remove, axis=1, inplace=True)
        except Exception as e:
            st.error(f"Error during column removal: ")
            return None

        # --- Scaling Numerical Features ---
        try:
            numerical_cols = [col for col in df.select_dtypes(include=['number']).columns if
                                col != 'is_canceled' and df[col].dtype != 'bool']
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            st.session_state['scaler'] = scaler  # Store the scaler
        except Exception as e:
            st.error(f"Error during scaling: ")
            return None

        return df

    @st.cache_resource
    def load_model(model_path):
        """Loads a pickled model from the specified path."""
        try:
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            return model
        except FileNotFoundError:
            st.error(f"Error: The model file '' was not found.")
            return None
        except Exception as e:
            st.error(f"Error loading the model: ")
            return None

    def predict_cancellation(model, input_data, feature_names, scaler):
        """Predicts hotel booking cancellation based on user input."""
        try:
            # Create a DataFrame from the input data
            input_df = pd.DataFrame([input_data])

            # Encoding categorical features in the input data
            # Make sure you have all the columns required by the model
            for col in ['customer_type_Group', 'customer_type_Transient', 'customer_type_Transient-Party',
                        'deposit_type_Non Refund', 'deposit_type_Refundable']:
                if col not in input_df.columns:
                    input_df[col] = 0  # Setting default value

            # Ensure the input DataFrame has the same columns as the training data
            for feature in feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0  # Or some other default value

            # Select only the columns used during training
            input_df = input_df[feature_names]

            # Identify numerical columns in the input data
            numerical_cols = input_df.select_dtypes(include=['number']).columns.tolist()

            # Scale the numerical columns
            if scaler is not None and numerical_cols:
                input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

            prediction = model.predict(input_df)
            return prediction[0]
        except Exception as e:
            st.error(f"Error during prediction: ")
            return None

    # Set FILE_PATH here
    FILE_PATH = 'hotel_bookings.csv'
    MODEL_FILE_PATH = 'best_random_forest_model.pkl'

    # Add the function here
    def train_and_evaluate_model(X_train_smote, y_train_smote, X_test, y_test, X_val, y_val, MODEL_FILE_PATH):
        """Trains and evaluates the RandomForestClassifier model."""
        try:
            # Define Hyperparameter Grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            }

            # Initialize RandomizedSearchCV
            rf_model = RandomForestClassifier(random_state=42)
            grid_search = RandomizedSearchCV(
                estimator=rf_model,
                param_distributions=param_grid,
                cv=3,
                scoring='accuracy',  # Evaluate using accuracy on test set
                n_iter=10,
                n_jobs=-1,
                random_state=42
            )

            grid_search.fit(X_train_smote, y_train_smote)
            best_rf_model = grid_search.best_estimator_

            # Evaluate on the TEST set (NOT Validation)
            y_pred_test = best_rf_model.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            test_roc_auc = roc_auc_score(y_test, best_rf_model.predict_proba(X_test)[:, 1])
            st.success(f"Random Forest Accuracy on TEST Set: {test_accuracy:.4f}")  # Display accuracy
            st.success(f"Random Forest ROC AUC on TEST Set: {test_roc_auc:.4f}")

            # Feature importance analysis - interactive plot
            importances = best_rf_model.feature_importances_
            feature_importance_dict = dict(zip(X_train_smote.columns, importances))
            sorted_feature_importances = sorted(feature_importance_dict.items(), key=lambda item: item[1], reverse=True)

            feature_names = [item[0] for item in sorted_feature_importances]
            feature_importances = [item[1] for item in sorted_feature_importances]

            fig = px.bar(x=feature_names, y=feature_importances,
                        labels={'x': 'Feature', 'y': 'Importance'},
                        title='Feature Importances', color=feature_importances,
                         color_continuous_scale='Viridis')  # Visual improvements here
            st.plotly_chart(fig)

            # Save the model to a file
            with open(MODEL_FILE_PATH, 'wb') as file:
                pickle.dump(best_rf_model, file)
            st.success(f"Trained model saved to ")

            return best_rf_model, list(X_train_smote.columns)  # Return the trained model and feature names
        except Exception as e:
            st.error(f"Error during model training: ")
            return None, []

    def main():
        st.set_page_config(page_title="Hotel Cancellation Predictor", layout="wide", page_icon=":hotel:") # added a page icon

        # Custom CSS for improved aesthetics
        st.markdown(
            """
            <style>
            .stApp {
                background-color: #f0f2f6; /* Light grey background */
            }
            .stButton>button {
                color: white;
                background-color: #007bff; /* Primary blue */
                border: none;
                padding: 10px 24px;
                border-radius: 4px;
                cursor: pointer;
            }
            .stButton>button:hover {
                background-color: #0056b3; /* Darker blue on hover */
            }
            .sidebar .sidebar-content {
                background-color: #ffffff; /* White sidebar */
                padding: 20px;
                border-radius: 5px;
            }
            h1, h2, h3 {
                color: #333333; /* Dark grey headers */
            }
            .css-10trblm { /* For success message */
               background-color: #d4edda !important;
               color: #155724 !important;
               border-color: #c3e6cb !important;
            }
            .css-qri22k { /* For error message */
               background-color: #f8d7da !important;
               color: #721c24 !important;
               border-color: #f5c6cb !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Title with markdown for larger font and center alignment
        st.markdown("<h1 style='text-align: center;'>Hotel Booking Cancellation Prediction</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Developed by Amanuel Agajjie Wasihun</p>", unsafe_allow_html=True)  # Subtitle

        # --- Sidebar for App Information and Settings ---
        with st.sidebar:
            st.header("About this App",  divider='rainbow') # A bit more visually engaging
            st.write("This app predicts the likelihood of a hotel booking being canceled.")
            st.write("Enter booking details below to get a prediction.")

            # Add an image in the sidebar
            try:
                image = Image.open("hotel.jpg")  # Open the image using PIL
                st.image(image, caption="Hotel Booking", use_column_width=True) #Add a hotel image
            except FileNotFoundError:
                st.error("Error: The file 'hotel.jpg' was not found in the same directory.")
            except Exception as e:
                st.error(f"Error opening the image: {e}")


        # --- 2. Data Loading and Model Loading ---
        # Use a session state to load data and model only once
        if 'df' not in st.session_state:
            with st.spinner("Loading and preprocessing data..."): # Show a spinner while loading
                st.session_state['df'] = load_and_preprocess_data(FILE_PATH)

        if st.session_state['df'] is None:
            st.error("Failed to load and preprocess data.")
            return

        # --- Splitting the Data ---
        if 'X_train_smote' not in st.session_state:
            try:
                df = st.session_state['df'].copy()  # Avoid modifying the session state directly
                X = df.drop('is_canceled', axis=1)
                y = df['is_canceled']

                # Split data into training, validation, and testing sets
                X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
                X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

                # Handle class imbalance using SMOTE
                smote = SMOTE(random_state=42)
                X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

                st.session_state['X_train_smote'] = X_train_smote
                st.session_state['y_train_smote'] = y_train_smote
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test
                st.session_state['X_val'] = X_val
                st.session_state['y_val'] = y_val
            except Exception as e:
                st.error(f"Error during data splitting/SMOTE: ")
                return

        # Add a button to train the model
        if st.sidebar.button("Train New Model", help="Train a new model with current dataset"):
            with st.spinner("Training the model..."):
                X_train_smote = st.session_state['X_train_smote']
                y_train_smote = st.session_state['y_train_smote']
                X_test = st.session_state['X_test']
                y_test = st.session_state['y_test']
                X_val = st.session_state['X_val']
                y_val = st.session_state['y_val']

                model, feature_names = train_and_evaluate_model(X_train_smote, y_train_smote, X_test, y_test, X_val,
                                                                y_val, MODEL_FILE_PATH)
                if model is not None:
                    st.session_state['model'] = model
                    st.session_state['feature_names'] = feature_names
                    st.success("Model trained and saved successfully!")
                else:
                    st.error("Model training failed.")
                    return

        elif 'model' not in st.session_state:
            with st.spinner("Loading the model..."): # Use spinner for model loading as well
                st.session_state['model'] = load_model(MODEL_FILE_PATH)

        if st.session_state['model'] is None:
            st.error("Failed to load the model.")
            return

        # Load scaler from session state if available
        if 'scaler' in st.session_state:
            scaler = st.session_state['scaler']
        else:
            scaler = None

        # Get feature names here to avoid running train_model during prediction
        feature_names = []  # You might need a way to get the feature names without re-training if necessary

        # Check if feature names are in session state, or load them from the trained model
        if 'feature_names' in st.session_state:
            feature_names = st.session_state['feature_names']
        else:
            # Attempt to load feature names from the model (if possible)
            if hasattr(st.session_state['model'], 'feature_names_in_'):
                feature_names = list(st.session_state['model'].feature_names_in_)
            else:
                st.error("Could not load feature names: Model does not have feature_names_in_ attribute.")
                return

        if not feature_names:
            st.error("Could not load feature names")
            return

        # Input Form
        st.header("Enter Booking Details:",  divider='rainbow') #Header with divider

        # Use columns for better layout
        col1, col2 = st.columns(2)  # Adjust the number as needed

        input_data = {}

        with col1:  # First column
            input_data['lead_time'] = st.number_input("Lead Time", value=30, min_value=0,
                                                        help="Time in days between booking and arrival", step=1)
            input_data['stays_in_weekend_nights'] = st.number_input("Weekend Nights", value=1, min_value=0,
                                                                        help="Number of weekend nights", step=1)
            input_data['stays_in_week_nights'] = st.number_input("Week Nights", value=2, min_value=0,
                                                                    help="Number of week nights", step=1)
            input_data['adults'] = st.number_input("Adults", value=2, min_value=1, help="Number of adults", step=1)
            input_data['children'] = st.number_input("Children", value=0, min_value=0, help="Number of children", step=1)
            input_data['babies'] = st.number_input("Babies", value=0, min_value=0, help="Number of babies", step=1)
            input_data['previous_cancellations'] = st.number_input("Previous Cancellations", value=0, min_value=0,
                                                                        help="Number of previous cancellations", step=1)
            input_data['previous_bookings_not_canceled'] = st.number_input("Previous Non-Cancellations", value=0,
                                                                            min_value=0,
                                                                            help="Number of previous non-cancellations", step=1)

        with col2:  # Second column
            input_data['booking_changes'] = st.number_input("Booking Changes", value=0, min_value=0,
                                                            help="Number of booking changes", step=1)
            input_data['adr'] = st.number_input("ADR (Avg Daily Rate)", value=100.0, min_value=0.0,
                                                    help="Average daily rate", step=1.0)
            input_data['required_car_parking_spaces'] = st.number_input("Parking Spaces", value=0, min_value=0,
                                                                            help="Number of required parking spaces", step=1)
            input_data['total_of_special_requests'] = st.number_input("Special Requests", value=0, min_value=0,
                                                                        help="Number of special requests", step=1)
            input_data['arrival_date_week_number'] = st.number_input("Arrival Week Number", value=1, min_value=1,
                                                                        max_value=53, help="Arrival week number", step=1)
            input_data['arrival_date_day_of_month'] = st.number_input("Arrival Day of Month", value=1, min_value=1,
                                                                        max_value=31,
                                                                        help="Arrival day of month", step=1)

            # Example categorical feature, replace with actual features
            input_data['deposit_type_Non Refund'] = st.selectbox("Deposit Type (Non Refund)", [0, 1], index=0,
                                                                    help="Deposit Type (Non Refund)")

            # Customer type selector
            customer_type = st.selectbox("Customer Type", ['Transient', 'Contract', 'Transient-Party', 'Group'],
                                        index=0, help="Type of customer")

        # Map selected customer type to dummy variables
        input_data['customer_type_Group'] = 1 if customer_type == 'Group' else 0
        input_data['customer_type_Transient'] = 1 if customer_type == 'Transient' else 0
        input_data['customer_type_Transient-Party'] = 1 if customer_type == 'Transient-Party' else 0

        # Prediction Button
        if st.button("Predict Cancellation", type="primary"):  # Make primary for emphasis
            try:
                with st.spinner("Predicting Cancellation..."): #Show a spinner
                    prediction = predict_cancellation(st.session_state['model'], input_data, feature_names, scaler)

                if prediction == 1:
                    st.error("The booking will likely be canceled.")
                else:
                    st.success("The booking will likely not be canceled.")
            except Exception as e:
                st.error(f"An error occurred during prediction: ")

except Exception as e:  # Add
    st.error(f"A critical error occurred: ")  # Add

if __name__ == "__main__":
    main()