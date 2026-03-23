import asyncio
import time
from groq import Groq
from config import GROQ_API_KEY, SYSTEM_PROMPT

async def test_groq_speed():
    if not GROQ_API_KEY:
        print("GROQ_API_KEY is not set")
        return

    # Sample prompt similar to what the bot uses
    context = "University admission requirements: Minimum GPA of 3.0, SAT scores above 1200, two letters of recommendation."
    question = "What are the admission requirements?"
    prompt = SYSTEM_PROMPT.format(context=context, question=question)

    client = Groq(api_key=GROQ_API_KEY)

    print("Testing Groq response time...")
    start_time = time.time()

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.3,
        )
        end_time = time.time()
        response_time = end_time - start_time

        print(".2f")
        print(f"Answer: {response.choices[0].message.content[:100]}...")
        print("✓ Groq test completed!")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_groq_speed())