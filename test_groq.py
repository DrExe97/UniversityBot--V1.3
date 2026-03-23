import asyncio
from groq import Groq
from config import GROQ_API_KEY

async def test_groq():
    if not GROQ_API_KEY:
        print("GROQ_API_KEY is not set in .env")
        return

    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            max_tokens=100,
        )
        print(f'Response: {response.choices[0].message.content}')
        print('✓ Groq API is working correctly!')
    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    asyncio.run(test_groq())