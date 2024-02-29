import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.linear_model import LinearRegression
from huggingface_hub import HfApi, login




#create hugging face repo 
api = HfApi()
#api.create_repo(repo_id="Hemg/housegrad")



#Log in to Hugging Face using the provided token.
    
def hugging_face_login(token):
    
    login(token=token, add_to_git_credential=True)


#Fill missing values in the DataFrame.
def handle_missing_values(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_features_skew = df[numerical_columns].apply(lambda x: x.skew())

    for feature in numerical_columns:
        if numerical_features_skew[feature] > 0:
            df[feature] = df[feature].fillna(df[feature].mean())
        elif numerical_features_skew[feature] < 0:
            df[feature] = df[feature].fillna(df[feature].median())
        else:
            df[feature] = df[feature].fillna(df[feature].mean())

    return df


#Encode categorical columns using LabelEncoder and OneHotEncoder.

def encode_categorical_columns(df):
    label_encoder = LabelEncoder()
    ordinal_columns = df.select_dtypes(include=['object']).columns

    for col in ordinal_columns:
        df[col] = label_encoder.fit_transform(df[col])

    nominal_columns = df.select_dtypes(include=['object']).columns.difference(ordinal_columns)
    df = pd.get_dummies(df, columns=nominal_columns, drop_first=True)

    return df

#Define a linear regression model

def linear_regression_model():
    return LinearRegression()


def preprocess_and_train(csv_file_path, dependent_var_name, independent_variables):
    
    # Load CSV file
    df = pd.read_csv(csv_file_path)

    # Handle missing values
    df = handle_missing_values(df)

    # Encode categorical columns
    df = encode_categorical_columns(df)

    # Define the X and y
    X = df[independent_variables]
    y = df[dependent_var_name]

    # Apply standard scaling on X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Define Linear Regression model
    model = linear_regression_model()

    # Fit the model
    model.fit(X_train, y_train)

    # Save the scikit-learn model locally
    joblib.dump(model,'linear.joblib')

    # Save the scikit-learn model in hugging face hub
    api.upload_file(repo_id="Hemg/housegrad",
                    path_in_repo="linear.joblib",
                    repo_type="model", 
                    path_or_fileobj="linear.joblib")

    # Save the scaler
    joblib.dump(scaler, 'scaler.joblib')

    # Save the scaler model in hugging face hub
    api.upload_file(repo_id="Hemg/housegrad", 
                    path_in_repo="scaler.joblib",
                    repo_type="model", 
                    path_or_fileobj="scaler.joblib")

    # Print Mean Squared Error on Test Set
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on Test Set: {mse}")

if __name__ == "__main__":

    hugging_face_token = "#"  # Hugging Face token
    hugging_face_login(hugging_face_token)
    csv_file_path = 'melb_data.csv'  # Replace with your CSV file path
    dependent_var_name = input('Enter the column name for the dependent variable: ') # dependent vairable represents predicated value
    independent_variables = input('Enter the column names for independent variables (comma-separated): ')
    independent_variables = [col.strip() for col in independent_variables.split(',')]
    preprocess_and_train(csv_file_path, dependent_var_name, independent_variables)
