import asyncio
from learning_engine import ask_question

async def test_full_pipeline():
    print("Testing full RAG pipeline with current LLM provider...")

    # Test with a question that should find relevant context
    question = "What are the admission requirements?"

    result = await ask_question(question, model_speed="fast")

    print(f"Question: {question}")
    print(f"Model used: {result['model_used']}")
    print(".1f")
    print(f"Confidence: {result['confidence']}")
    print(f"Sources found: {result['sources_found']}")
    print(f"Answer: {result['answer'][:200]}...")

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())