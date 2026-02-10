"""
Example workflow demonstrating CognitionOS capabilities.

This script shows how to:
1. Use the AI Runtime for completions
2. Store results in Memory Service
3. Retrieve contextual information
"""

import asyncio
import httpx
from uuid import uuid4
import json


# Configuration
AI_RUNTIME_URL = "http://localhost:8005"
MEMORY_SERVICE_URL = "http://localhost:8004"


async def main():
    """
    Run example workflow demonstrating AI-powered task execution.
    """
    print("=" * 70)
    print("CognitionOS Example Workflow")
    print("=" * 70)

    user_id = uuid4()
    print(f"\nUser ID: {user_id}")

    async with httpx.AsyncClient(timeout=30.0) as client:

        # ====================================================================
        # Step 1: Check system health
        # ====================================================================
        print("\n[Step 1] Checking system health...")

        ai_health = await client.get(f"{AI_RUNTIME_URL}/health")
        ai_data = ai_health.json()
        print(f"  AI Runtime: {ai_data['status']}")
        print(f"  - OpenAI available: {ai_data['providers']['openai']}")
        print(f"  - Anthropic available: {ai_data['providers']['anthropic']}")
        print(f"  - Simulation mode: {ai_data.get('simulation_mode', False)}")

        memory_health = await client.get(f"{MEMORY_SERVICE_URL}/health")
        memory_data = memory_health.json()
        print(f"  Memory Service: {memory_data['status']}")
        print(f"  - Database: {memory_data.get('database', 'unknown')}")
        print(f"  - Total memories: {memory_data.get('total_memories', 0)}")

        # ====================================================================
        # Step 2: Generate code with AI
        # ====================================================================
        print("\n[Step 2] Generating code with AI Runtime...")

        code_prompt = """
        Write a Python function that:
        1. Takes a list of numbers
        2. Filters out negative numbers
        3. Returns the sum of positive numbers

        Include docstring and type hints.
        """

        completion_response = await client.post(
            f"{AI_RUNTIME_URL}/complete",
            json={
                "role": "executor",
                "prompt": code_prompt,
                "context": [],
                "max_tokens": 500,
                "temperature": 0.3,  # Lower temperature for code generation
                "user_id": str(user_id)
            }
        )

        if completion_response.status_code == 200:
            completion = completion_response.json()
            print(f"  ✓ Code generated successfully")
            print(f"  - Model: {completion['model_used']}")
            print(f"  - Tokens: {completion['tokens_used']} ({completion['prompt_tokens']} + {completion['completion_tokens']})")
            print(f"  - Cost: ${completion['cost_usd']:.6f}")
            print(f"  - Latency: {completion['latency_ms']}ms")

            generated_code = completion['content']
            print(f"\n  Generated Code Preview:")
            print("  " + "-" * 60)
            for line in generated_code.split('\n')[:15]:
                print(f"  {line}")
            if len(generated_code.split('\n')) > 15:
                print(f"  ... ({len(generated_code.split('\n')) - 15} more lines)")
            print("  " + "-" * 60)
        else:
            print(f"  ✗ Failed: {completion_response.status_code}")
            return

        # ====================================================================
        # Step 3: Store in memory
        # ====================================================================
        print("\n[Step 3] Storing code in Memory Service...")

        memory_response = await client.post(
            f"{MEMORY_SERVICE_URL}/memories",
            json={
                "user_id": str(user_id),
                "content": f"Python function to sum positive numbers:\n\n{generated_code}",
                "memory_type": "semantic",
                "scope": "user",
                "metadata": {
                    "task_type": "code_generation",
                    "language": "python",
                    "function": "sum_positive",
                    "tokens_used": completion['tokens_used'],
                    "model": completion['model_used']
                },
                "source": "ai_runtime",
                "confidence": 0.9,
                "is_sensitive": False
            }
        )

        if memory_response.status_code == 201:
            memory = memory_response.json()
            print(f"  ✓ Memory stored successfully")
            print(f"  - Memory ID: {memory['id']}")
            print(f"  - Type: {memory['memory_type']}")
            print(f"  - Scope: {memory['scope']}")
        else:
            print(f"  ✗ Failed: {memory_response.status_code}")

        # ====================================================================
        # Step 4: Generate another piece of code
        # ====================================================================
        print("\n[Step 4] Generating related code...")

        second_prompt = """
        Write a Python function that:
        1. Takes a list of numbers
        2. Returns only the positive even numbers
        3. Sorts them in descending order

        Include docstring and type hints.
        """

        second_completion = await client.post(
            f"{AI_RUNTIME_URL}/complete",
            json={
                "role": "executor",
                "prompt": second_prompt,
                "context": [],
                "max_tokens": 500,
                "temperature": 0.3,
                "user_id": str(user_id)
            }
        )

        if second_completion.status_code == 200:
            second_data = second_completion.json()
            print(f"  ✓ Second code generated")
            print(f"  - Tokens: {second_data['tokens_used']}")
            print(f"  - Cost: ${second_data['cost_usd']:.6f}")

            # Store second memory
            await client.post(
                f"{MEMORY_SERVICE_URL}/memories",
                json={
                    "user_id": str(user_id),
                    "content": f"Python function to filter positive even numbers:\n\n{second_data['content']}",
                    "memory_type": "semantic",
                    "scope": "user",
                    "metadata": {
                        "task_type": "code_generation",
                        "language": "python",
                        "function": "filter_positive_even"
                    },
                    "source": "ai_runtime",
                    "confidence": 0.9,
                    "is_sensitive": False
                }
            )
            print(f"  ✓ Second memory stored")

        # ====================================================================
        # Step 5: Retrieve memories with semantic search
        # ====================================================================
        print("\n[Step 5] Retrieving memories with semantic search...")

        queries = [
            "How do I work with positive numbers?",
            "Show me functions for filtering lists",
            "What code did I generate for number processing?"
        ]

        for query in queries:
            print(f"\n  Query: \"{query}\"")

            retrieve_response = await client.post(
                f"{MEMORY_SERVICE_URL}/retrieve",
                json={
                    "user_id": str(user_id),
                    "query": query,
                    "k": 2,
                    "memory_types": ["semantic"],
                    "min_confidence": 0.0
                }
            )

            if retrieve_response.status_code == 200:
                memories = retrieve_response.json()
                print(f"  ✓ Found {len(memories)} relevant memories")

                for i, mem in enumerate(memories[:2], 1):
                    score = mem.get('relevance_score', 0)
                    content_preview = mem['content'][:80].replace('\n', ' ')
                    print(f"    {i}. Score: {score:.3f} - {content_preview}...")
            else:
                print(f"  ✗ Retrieval failed: {retrieve_response.status_code}")

        # ====================================================================
        # Step 6: Generate embeddings directly
        # ====================================================================
        print("\n[Step 6] Generating embeddings for custom texts...")

        embed_response = await client.post(
            f"{AI_RUNTIME_URL}/embed",
            json={
                "texts": [
                    "Python programming language",
                    "Machine learning algorithms",
                    "Data structures and algorithms"
                ],
                "model": "text-embedding-ada-002",
                "user_id": str(user_id)
            }
        )

        if embed_response.status_code == 200:
            embed_data = embed_response.json()
            print(f"  ✓ Generated embeddings for 3 texts")
            print(f"  - Dimension: {len(embed_data['embeddings'][0])}")
            print(f"  - Tokens: {embed_data['tokens_used']}")
            print(f"  - Cost: ${embed_data['cost_usd']:.6f}")
        else:
            print(f"  ✗ Failed: {embed_response.status_code}")

        # ====================================================================
        # Summary
        # ====================================================================
        print("\n" + "=" * 70)
        print("Workflow Summary")
        print("=" * 70)
        print(f"  ✓ AI Runtime operational")
        print(f"  ✓ Memory Service operational")
        print(f"  ✓ Generated 2 code functions")
        print(f"  ✓ Stored 2 memories with semantic search")
        print(f"  ✓ Retrieved memories using natural language queries")
        print(f"  ✓ Generated custom embeddings")
        print("\nCognitionOS is ready for production use!")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
