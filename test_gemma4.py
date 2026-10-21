import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure local .env is loaded when run directly or through pytest
load_dotenv(Path(__file__).resolve().parent / ".env")

from google import genai
from google.genai import types

def test_gemma_4(prompt: str = "Introduce yourself and confirm your model name."):
    # Read API key strictly from environment
    api_key = os.environ.get("GEMINI_API_KEY", "")
    
    print(f"Connecting to Gemma 4 via Google GenAI SDK...")
    client = genai.Client(api_key=api_key)
    model = os.environ.get("GEMMA_MODEL", "gemma-4-26b-a4b-it")

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]

    print(f"\n--- [Prompt]: {prompt} ---")
    print("--- [Gemma 4 Response Streaming]: ---")
    
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
    ):
        if text := chunk.text:
            print(text, end="", flush=True)
    print("\n--- [Stream Completed] ---")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    test_gemma_4()
