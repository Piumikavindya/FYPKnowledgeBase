from openai import OpenAI
import os
openai_api_key =  os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

response = client.responses.create(
    model="gpt-4.1",
    input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)
