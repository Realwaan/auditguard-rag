from google import genai
from google.genai import types
from app.config import GEMINI_API_KEY, get_active_gemma_model


def test_gemma_4(prompt: str = "Introduce yourself and confirm your model name."):
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY is not set in .env or environment variables.")
        return

    model = get_active_gemma_model()
    print(f"Connecting to {model} via Google GenAI SDK...")
    client = genai.Client(api_key=GEMINI_API_KEY)

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
    test_gemma_4()
