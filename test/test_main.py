from fastapi.testclient import TestClient

from application.main import app

client = TestClient(app)

# Testing get method say_hello()
def test_say_hello():
    r = client.get("/")
    # Testing successfull get call (200 OK)
    assert r.status_code == 200
    # Testing successfull get call return value
    assert r.text == '"Welcome to this project"'
    # Testing unsuccessfull get call (404 not found)
    r = client.get("/{some_param}")
    assert r.status_code == 404



