"""
Chinese conversation demo for CogMem.

Shows how to use bge-small-zh-v1.5 embedding model for improved
Chinese memory retrieval.

Prerequisites:
  pip install -e .
  export COGMEM_API_KEY=your-api-key
  export COGMEM_BASE_URL=https://api.deepseek.com/v1
  export COGMEM_MODEL=deepseek-chat
"""

from cogmem import CogMem


def main():
    mem = CogMem(
        db_path="chinese_memory.db",
        embed_model="BAAI/bge-small-zh-v1.5",
    )

    print("=== 添加对话 ===")
    mem.add(
        "[对话日期: 2023年04月27日]\n"
        "今天我和小王在星巴克喝了咖啡，他说下周要去北京出差，"
        "顺便看望他在北京大学读书的妹妹。小王还推荐了一本书叫《三体》。",
        user_id="user1",
    )
    mem.add(
        "[对话日期: 2023年04月28日]\n"
        "昨天跟小李聊天，她说她正在准备考研，目标是清华大学的计算机专业。"
        "她还提到最近在学日语，打算明年去日本旅游。",
        user_id="user1",
    )

    print("\n=== 搜索: '小王要去哪里出差？' ===")
    results = mem.search("小王要去哪里出差？", user_id="user1")
    for r in results[:5]:
        print(f"  [{r['type']}] {r['memory']} (score: {r['score']:.3f})")

    print("\n=== 搜索: '小李的目标是什么？' ===")
    results = mem.search("小李的目标是什么？", user_id="user1")
    for r in results[:5]:
        print(f"  [{r['type']}] {r['memory']} (score: {r['score']:.3f})")

    print("\n=== 搜索: '小王推荐了什么书？' ===")
    results = mem.search("小王推荐了什么书？", user_id="user1")
    for r in results[:5]:
        print(f"  [{r['type']}] {r['memory']} (score: {r['score']:.3f})")

    print("\n=== 搜索: '4月27日发生了什么？' ===")
    results = mem.search("4月27日发生了什么？", user_id="user1")
    for r in results[:5]:
        print(f"  [{r['type']}] {r['memory']} (score: {r['score']:.3f})")

    print(f"\n总记忆数: {mem.count_memories('user1')}")
    mem.close()


if __name__ == "__main__":
    main()
