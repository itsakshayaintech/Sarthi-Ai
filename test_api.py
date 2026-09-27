import re
import uuid

import requests


BASE_URL = "http://127.0.0.1:5000"
URL = f"{BASE_URL}/assess"


# -------------------------------------------------
# TEST 1: AI CONSENT GIVEN
# -------------------------------------------------

data = {
    "ai_consent": True,

    "indicators": {
        "fear": 75,
        "threat": 70,
        "anxiety": 65,
        "trauma": 70,
        "social_isolation": 50,
        "displacement": 40,
        "legal_stress": 65,
        "safety_concern": 60,
    },

    "text": (
        "The complainant reports feeling unsafe, fearful and emotionally "
        "distressed because of the situation and requests support."
    ),

}


client = requests.Session()
registration_page = client.get(f"{BASE_URL}/register")
registration_token = re.search(
    r'name="csrf_token" value="([^"]+)"',
    registration_page.text,
).group(1)
client.post(
    f"{BASE_URL}/register",
    data={
        "csrf_token": registration_token,
        "name": "API Test User",
        "email": f"api-test-{uuid.uuid4().hex}@example.test",
        "password": "test-password-123",
    },
)
dashboard_page = client.get(f"{BASE_URL}/dashboard")
csrf_token = re.search(
    r'<meta name="csrf-token" content="([^"]+)"',
    dashboard_page.text,
).group(1)

response = client.post(
    URL,
    json=data,
    headers={"X-CSRF-Token": csrf_token},
)

print("\nSTATUS CODE:")
print(response.status_code)

print("\nRESPONSE:")
print(response.json())