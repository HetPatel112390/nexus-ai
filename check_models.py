import urllib.request
import json
import os

key = "AQ.Ab8RN6KzKIX2bp4CjqwSAD-wCjR9NMnOZ7eXi3uDAoDq5eCD7w"
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
req = urllib.request.Request(url)
with urllib.request.urlopen(req) as response:
    data = json.loads(response.read().decode())
    print([m["name"] for m in data.get("models", []) if "generateContent" in m.get("supportedGenerationMethods", [])])
