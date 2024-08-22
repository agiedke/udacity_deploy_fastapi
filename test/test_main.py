import pytest
from fastapi.testclient import TestClient

from application.main import app

client = TestClient(app)


# 1. Testing the GET / endpoint
def test_say_hello():
    r = client.get("/")
    # Testing successfull get call (200 OK)
    assert r.status_code == 200
    # Testing successfull get call return value
    assert r.text == '"Welcome to this project"'
    # Testing unsuccessfull get call (404 not found)
    r = client.get("/{some_param}")
    assert r.status_code == 404


# 2. Testing the POST /predict endpoint -> unsuccessfull POST request (400 bad request) due to wrong input values
def test_predict_invalid_input():
    # payload with missing required fields
    invalid_payload = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 178356,
        # Missing other required fields
    }

    # Send POST request with invalid payload
    response = client.post("/predict", json=invalid_payload)

    # Assert that the status code is 422 (Unprocessable Entity)
    assert response.status_code == 422
    assert "detail" in response.json()


# 3. Testing the POST /predict endpoint -> successful POST Request (Valid Input) with return value "<=50K"
@pytest.mark.asyncio
async def test_predict_successfull_inf1():
    # Valid payload
    valid_payload = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 178356,
        "education": "HS-grad",
        "education-num": 10,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
        "salary": ""
    }

    # Send POST request with valid payload
    response = client.post("/predict", json=valid_payload)

    # Debugging: Print response details
    print(response.status_code)
    print(response.text)

    # Assert that the status code is 200 (OK)
    assert response.status_code == 200
    # Testing successfull post call return types
    json_response = response.json()
    assert "prediction" in json_response
    predictions = json_response["prediction"]
    assert isinstance(predictions, list)
    assert all(isinstance(p, str) for p in predictions)
    # Testing successfull post call return values
    assert predictions[0] == "<=50K"

    # Assert that the response contains the prediction
    response_json = response.json()
    assert "prediction" in response_json
    assert isinstance(response_json["prediction"], list)
    assert len(response_json["prediction"]) > 0


# 4. Testing the POST /predict endpoint -> successful POST Request (Valid Input) with return value ">=50K"
@pytest.mark.asyncio
async def test_predict_successfull_inf2():
    # Valid payload
    valid_payload = {
        "age": 56,
        "workclass": "Local-gov",
        "fnlgt": 52953,
        "education": "Doctorate",
        "education-num": 16,
        "marital-status": "Divorced",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 1669,
        "hours-per-week": 38,
        "native-country": "United-States",
        "salary": ""
    }

    # Send POST request with valid payload
    response = client.post("/predict", json=valid_payload)

    # Debugging: Print response details
    print(response.status_code)
    print(response.text)

    # Assert that the status code is 200 (OK)
    assert response.status_code == 200
    # Testing successfull post call return types
    json_response = response.json()
    assert "prediction" in json_response
    predictions = json_response["prediction"]
    assert isinstance(predictions, list)
    assert all(isinstance(p, str) for p in predictions)
    # Testing successfull post call return values
    assert predictions[0] == ">50K"

    # Assert that the response contains the prediction
    response_json = response.json()
    assert "prediction" in response_json
    assert isinstance(response_json["prediction"], list)
    assert len(response_json["prediction"]) > 0
