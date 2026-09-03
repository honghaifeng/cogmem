"""
Basic usage example for CogMem.

Prerequisites:
  pip install -e .
  export COGMEM_API_KEY=your-api-key
  export COGMEM_BASE_URL=https://api.deepseek.com/v1
  export COGMEM_MODEL=deepseek-chat
"""

from cogmem import CogMem


def main():
    mem = CogMem(db_path="example_memory.db")

    print("=== Adding conversations ===")
    mem.add(
        "I had lunch with Alice at the Italian restaurant on 5th Avenue. "
        "She mentioned she's moving to Tokyo next month for a new job at Sony.",
        user_id="user1",
    )
    mem.add(
        "Bob called me today. He said the project deadline has been moved to July 15th. "
        "He also recommended the book 'The Pragmatic Programmer'.",
        user_id="user1",
    )

    print("\n=== Searching: 'Where is Alice moving?' ===")
    results = mem.search("Where is Alice moving?", user_id="user1")
    for r in results[:5]:
        print(f"  [{r['type']}] {r['memory']} (score: {r['score']:.3f})")

    print("\n=== Searching: 'What book did Bob recommend?' ===")
    results = mem.search("What book did Bob recommend?", user_id="user1")
    for r in results[:5]:
        print(f"  [{r['type']}] {r['memory']} (score: {r['score']:.3f})")

    print("\n=== Searching: 'What is the project deadline?' ===")
    results = mem.search("What is the project deadline?", user_id="user1")
    for r in results[:5]:
        print(f"  [{r['type']}] {r['memory']} (score: {r['score']:.3f})")

    print(f"\nTotal memories: {mem.count_memories('user1')}")
    mem.close()


if __name__ == "__main__":
    main()
