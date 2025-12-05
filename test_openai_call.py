import os
import openai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("No API key found")
else:
    client = openai.OpenAI(api_key=api_key)

    try:
        print("Attempting to call client.responses.create...")
        response = client.responses.create(
            model="gpt-5",
            input=[{"role": "user", "content": "Hello"}],
            tools=[{"type": "web_search"}],
            tool_choice="auto",
            reasoning={"effort": "medium"},
        )
        print("Success!")
    except Exception as e:
        print(f"Error: {e}")
