import requests

# URL of the FastAPI endpoint
url = "http://127.0.0.1:8000/predict"

# JSON payload
payload = {
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
    "salary": "<=50K"
}

# POST request
response = requests.post(url, json=payload)

# print response code and data
if response.status_code == 200:
    print(f"Request was successful. Status code: {response.status_code}")
    print(f"Response data:\n{response.json()}")
else:
    print(f"Request failed with status code: {response.status_code}")
    print(f"Response text:\n{response.text}")