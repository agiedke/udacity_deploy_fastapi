import requests

# URL of the FastAPI endpoint
url = "https://udacity-deploy-fastapi.onrender.com/predict"

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

try:
    # POST request
    response = requests.post(url, json=payload)

    # print response code and data
    if response.status_code == 200:
        print(f"Request was successful. Status code: {response.status_code}")
        print(f"Response data:\n{response.json()}")
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(f"Response text:\n{response.text}")
except requests.exceptions.RequestException as e:
    print(f"error occured during request: {e}")

