import gradio as gr
import joblib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

# Load the trained model and scaler objects from file


REPO_ID = "Hemg/HousePricegradio" # hugging face  repo ID
MoDEL_FILENAME = "housepricegradio.joblib" # model file name
SCALER_FILENAME ="scalarpricegradio.joblib" # scaler file name

model = joblib.load(hf_hub_download(repo_id=REPO_ID, filename=MoDEL_FILENAME))

scaler = joblib.load(hf_hub_download(repo_id=REPO_ID, filename=SCALER_FILENAME))
 
# model = joblib.load('D:\gradioapp\X.joblib')
# scaler = joblib.load('D:\gradioapp\Xx.joblib')

# Define the prediction function
def predict_price(Rooms, Distance, Bedroom2, Bathroom, Car, Landsize, BuildingArea, YearBuilt, Propertycount):
    # Prepare input data represents independent variables for house prediction
    input_data = [[Rooms, Distance, Bedroom2, Bathroom, Car, Landsize, BuildingArea, YearBuilt, Propertycount]]

    # Get the feature names from the Gradio interface inputs
    feature_names = ["Rooms", "Distance", "Bedroom2", "Bathroom", "Car", "Landsize", "BuildingArea", "YearBuilt", "Propertycount"]
    # Create a Pandas DataFrame with the input data and feature names
    input_df = pd.DataFrame(input_data, columns=feature_names)

    # Scale the input data using the loaded scaler
    scaled_input = scaler.transform(input_df)

    # Make predictions using the loaded model
    prediction = model.predict(scaled_input)[0]
    
    return f"Predicted House Price: ${prediction:,.2f}" # Price is our dependent variable

# Create the Gradio app
iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Rooms"),
        gr.Number(label="Distance"),
        gr.Number(label="Bedroom2"),
        gr.Number(label="Bathroom"),
        gr.Number(label="Car"),
        gr.Number(label="Landsize"),
        gr.Number(label="BuildingArea"),
        gr.Number(label="YearBuilt"),
        gr.Number(label="Propertycount")
    ],
    outputs="text",
    title="House_PricePrediction",
    description="Predict House Price"
)

# Run the app
if __name__ == "__main__":
    iface.launch(share=True)
