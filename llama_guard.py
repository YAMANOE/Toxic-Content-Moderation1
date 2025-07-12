import requests

API_URL = "https://api.llamaguard.com/moderate"
API_KEY = "<YOUR_API_KEY>"

def llama_guard_check(text):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "task": "classification",
        "categories": ["safe", "unsafe"]
    }
    response = requests.post(API_URL, json=payload, headers=headers)
    result = response.json()
    
    # Return the prediction from Llama Guard API
    return result.get("prediction", "unsafe")
