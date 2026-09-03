"""
Extraction prompts for entity-relation knowledge network.

Bilingual prompts that instruct the LLM to extract structured information
(entities, relations, facts) from conversation text, with automatic
relative-to-absolute date conversion.
"""

EXTRACTION_PROMPT_EN = """You are an expert information extraction system. Extract structured information from the conversation below.

IMPORTANT - Date handling:
- The conversation has a date shown at the top (e.g., "[Conversation Date: 1:56 pm on 8 May, 2023]")
- Convert ALL relative time references to ABSOLUTE dates using the conversation date
- Examples: if conversation date is "8 May 2023":
  - "yesterday" -> "7 May 2023"
  - "last year" -> "2022"
  - "next month" -> "June 2023"
  - "last week" -> "1 May 2023" (approximate is fine)
- Always store dates as properties of entities/events, not as relative references

Tasks:
1. Extract all entities (persons, locations, organizations, events, things) with their attributes
2. Extract relationships between entities
3. Extract standalone fact statements as flat memories
4. Make sure dates are absolute

Return ONLY valid JSON:
{{
  "entities": [
    {{
      "name": "entity name",
      "type": "person | location | organization | event | thing",
      "properties": {{
        "key1": "value1",
        "key2": "value2"
      }},
      "summary": "one sentence describing the core of this entity"
    }}
  ],
  "relations": [
    {{
      "source": "source entity name",
      "target": "target entity name",
      "relation": "relation type",
      "properties": {{}}
    }}
  ],
  "facts": [
    "standalone fact statements, each as a complete sentence with absolute dates"
  ]
}}

Notes:
- Only extract explicitly mentioned information
- Standardize entity names (same name for same entity)
- Ensure all dates are absolute
- If nothing worth extracting, return empty arrays

Conversation:
{conversation}
"""

EXTRACTION_PROMPT_CN = """你是一个信息提取专家系统。从以下对话中提取结构化信息。

重要 - 日期处理:
- 对话顶部的日期标记（如 "[对话日期: 2023年04月27日]"）表示对话发生的日期
- 将所有相对时间引用转换为绝对日期
- 例如：对话日期是"2023年4月27日"：
  - "昨天" -> "2023年4月26日"
  - "去年" -> "2022年"
  - "下个月" -> "2023年5月"
  - "上周" -> "2023年4月20日"（近似即可）
- 始终将日期作为实体/事件的属性存储，不要用相对引用

任务:
1. 提取所有实体（人物、地点、组织、事件、物品）及其属性
2. 提取实体之间的关系
3. 提取独立的事实陈述作为扁平记忆
4. 确保日期是绝对日期

只返回有效的 JSON:
{{
  "entities": [
    {{
      "name": "实体名称",
      "type": "person | location | organization | event | thing",
      "properties": {{
        "key1": "value1",
        "key2": "value2"
      }},
      "summary": "一句话描述这个实体的核心"
    }}
  ],
  "relations": [
    {{
      "source": "源实体名称",
      "target": "目标实体名称",
      "relation": "关系类型",
      "properties": {{}}
    }}
  ],
  "facts": [
    "独立的事实陈述，每个都是包含绝对日期的完整句子"
  ]
}}

注意:
- 只提取明确提及的信息
- 统一实体名称（同一实体使用相同名称）
- 确保所有日期是绝对日期
- 用中文输出所有内容
- 如果没有值得提取的内容，返回空数组

对话:
{conversation}
"""

ANSWER_PROMPT = """Based on the following retrieved memories, answer the user's question. If the answer is not found in the memories, say "I don't have that record" rather than guessing.

Memories:
{memories}

Question: {question}

Answer:"""

ANSWER_PROMPT_CN = """根据以下检索到的记忆，回答用户的问题。如果记忆中没有相关信息，请说"我没有这个记录"，不要猜测。

记忆：
{memories}

问题：{question}

回答："""
