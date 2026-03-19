import httpx
import asyncio

async def test_ollama():
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post('http://localhost:11434/api/generate', json={
                'model': 'mistral:7b-instruct-q4_K_M',
                'prompt': 'Hello, how are you?',
                'stream': False
            })
            print(f'Status Code: {response.status_code}')
            data = response.json()
            print(f'Response: {data.get("response", "No response")}')
            print('✓ Ollama API is working correctly!')
    except Exception as e:
        print(f'Error: {e}')

asyncio.run(test_ollama())
