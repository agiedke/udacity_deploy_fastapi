import requests

url = "https://udacity-deploy-fastapi.onrender.com/predict"

file_path = "data/test_inference.csv"

files = {
    'file': open(file_path, 'rb')
}

try:
    # POST request
    response = requests.post(url, files=files)

    # print response code and data
    if response.status_code == 200:
        print(f"Request was successful. Status code: {response.status_code}")
        print(f"Response data:\n{response.json()}")
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(f"Response text:\n{response.text}")
finally:
    files['file'].close()
