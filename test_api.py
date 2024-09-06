import requests

# URL of the local server
url = "http://127.0.0.1:5000/predict"

# Data to send in the POST request
data = {
    "plot_size": [2000]
}

# Make the POST request
response = requests.post(url, json=data)

# Print the response from the server
print(response.json())
