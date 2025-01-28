import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Load the trained model
model = joblib.load('model/model.pkl')

# Define the FastAPI app
app = FastAPI()

# Define input schema
class InputData(BaseModel):
    age: int
    sex: int
    bmi: float
    children: int
    smoker: int
    northwest: int
    southeast: int
    southwest: int

# Define Prediction endpoint
@app.post("/api/predict")
def predict(data: InputData):
    input_features = [
        [
            data.age, data.sex, data.bmi, data.children, data.smoker,
            data.northwest, data.southeast, data.southwest
        ]
    ]
    prediction = model.predict(input_features)

    # Convert numpy.float 32 to native Python float for serialization
    predicted_charges = float(prediction[0])
    return {
        "predicted_charges": predicted_charges
    }


# Run with: uvicorn app: app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)