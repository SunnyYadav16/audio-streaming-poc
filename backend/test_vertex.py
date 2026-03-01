import google.auth
import google.auth.transport.requests
import openai
import os

project = "cloud-testing-apis-project"
location = "us-east5"
model = "meta/llama-3-1-8b-instruct-maas"  # Trying 3.1 8b first which is definitely GA

credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
auth_req = google.auth.transport.requests.Request()
credentials.refresh(auth_req)

base_url = f"https://{location}-aiplatform.googleapis.com/v1beta1/projects/{project}/locations/{location}/endpoints/openapi"
client = openai.OpenAI(base_url=base_url, api_key=credentials.token)

try:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=10,
    )
    print("Success:", response.choices[0].message.content)
except Exception as e:
    print(f"Failed: {type(e).__name__} - {e}")
