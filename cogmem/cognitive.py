"""
CognitiveMemory: FTS5 + Spreading Activation (no vector retrieval).

Standalone implementation matching the original cognitive_memory.py used in
benchmark testing. This is the intermediate ablation configuration between
BaselineMemory (FTS5 only) and full CogMem (FTS5 + Vector + Spreading Activation).

Architecture:
    1. Flat fragments — FTS5 full-text search (primary, same as baseline)
    2. Entity-relation network — spreading activation (enhancement)
    3. Fusion — flat results primary, entity results supplementary

No vector embeddings, no SentenceTransformer dependency.
"""

import json
import sqlite3
import time
from typing import Optional

from .llm_client import LLMClient
from .search_utils import contains_cjk, build_fts_query, extract_cn_keywords, cn_like_search
from .prompts import EXTRACTION_PROMPT_EN, EXTRACTION_PROMPT_CN


class CognitiveMemory:
    """
    FTS5 + Spreading Activation memory (no vector path).

    Cognitive memory is a superset of baseline:
    - Flat fragments with FTS5 (identical to baseline) — ensures no worse than baseline
    - Entity-relation network — structural modeling, supports multi-hop reasoning
    - Spreading activation — spreads from seed entities along relation edges

    Retrieval base: SQLite FTS5 (same as baseline, ensures fair comparison).

    Args:
        db_path: SQLite database path for persistent storage.
        llm_client: LLMClient instance for entity extraction. If None,
            creates one from environment variables.

    Example:
        .. code-block:: python

            from cogmem import CognitiveMemory

            mem = CognitiveMemory(db_path="cognitive.db")
            mem.add("I met Alice at the park yesterday.", user_id="user1")
            results = mem.search("Who did I meet?", user_id="user1")
    """

    def __init__(self, db_path: str = "cognitive_memory.db",
                 llm_client: Optional[LLMClient] = None):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.llm = llm_client or LLMClient()
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS fragments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                created_at REAL NOT NULL,
                user_id TEXT DEFAULT ''
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_frag_user ON fragments(user_id)")

        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS fragment_fts USING fts5(
                content,
                content_rowid,
                tokenize = 'unicode61'
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                type TEXT NOT NULL DEFAULT 'thing',
                properties TEXT DEFAULT '{}',
                summary TEXT DEFAULT '',
                importance REAL DEFAULT 5.0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                user_id TEXT DEFAULT ''
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ent_name ON entities(name)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ent_type ON entities(type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_ent_user ON entities(user_id)")

        self.conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS entity_fts USING fts5(
                name,
                summary,
                properties,
                content_rowid,
                tokenize = 'unicode61'
            )
        """)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                rel_type TEXT NOT NULL,
                properties TEXT DEFAULT '{}',
                weight REAL DEFAULT 1.0,
                created_at REAL NOT NULL,
                user_id TEXT DEFAULT ''
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_source ON relations(source_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_rel_target ON relations(target_id)")

        self.conn.commit()

    def _find_similar_entity(self, name: str, user_id: str = "") -> Optional[int]:
        name_lower = name.lower().strip()
        if len(name_lower) < 2:
            return None

        cursor = self.conn.execute(
            "SELECT id, name FROM entities WHERE LOWER(name) = ? AND (user_id = ? OR user_id = '') LIMIT 1",
            (name_lower, user_id)
        )
        row = cursor.fetchone()
        if row:
            return row[0]

        cursor = self.conn.execute(
            "SELECT id, name FROM entities WHERE (user_id = ? OR user_id = '')",
            (user_id,)
        )
        rows = cursor.fetchall()
        for ent_id, ent_name in rows:
            ent_lower = ent_name.lower()
            if name_lower in ent_lower or ent_lower in name_lower:
                if min(len(name_lower), len(ent_lower)) >= 3:
                    return ent_id

        return None

    def add(self, conversation_text: str, user_id: str = "", timestamp: Optional[float] = None):
        if timestamp is None:
            timestamp = time.time()

        if contains_cjk(conversation_text):
            prompt = EXTRACTION_PROMPT_CN
            system_msg = "你是一个精确的信息提取专家。只返回有效的 JSON。"
        else:
            prompt = EXTRACTION_PROMPT_EN
            system_msg = "You are a precise information extraction expert. Return ONLY valid JSON."

        result = self.llm.chat_completion_json(
            system=system_msg,
            user=prompt.format(conversation=conversation_text[:3000]),
            temperature=0.1,
        )

        entities_data = result.get("entities", [])
        relations_data = result.get("relations", [])
        facts_data = result.get("facts", [])

        if not isinstance(entities_data, list):
            entities_data = []
        if not isinstance(relations_data, list):
            relations_data = []
        if not isinstance(facts_data, list):
            facts_data = []

        fragments_added = 0

        for fact in facts_data:
            fact_str = str(fact).strip()
            if not fact_str or len(fact_str) < 3:
                continue
            self._add_fragment(fact_str, timestamp, user_id)
            fragments_added += 1

        entity_id_map = {}
        for ent in entities_data:
            if not isinstance(ent, dict):
                continue
            name = str(ent.get("name", "")).strip()
            if not name:
                continue

            ent_type = str(ent.get("type", "thing"))
            properties = ent.get("properties", {})
            if not isinstance(properties, dict):
                properties = {}
            summary = str(ent.get("summary", ""))

            existing_id = self._find_similar_entity(name, user_id)

            if existing_id:
                entity_id_map[name] = existing_id
                cursor = self.conn.execute(
                    "SELECT properties, summary FROM entities WHERE id = ?",
                    (existing_id,)
                )
                row = cursor.fetchone()
                if row:
                    old_props = json.loads(row[0] or "{}")
                    old_summary = row[1] or ""
                    merged_props = {**old_props, **properties}
                    new_summary = old_summary or summary

                    props_text = json.dumps(merged_props, ensure_ascii=False)

                    self.conn.execute(
                        "UPDATE entities SET properties = ?, summary = ?, updated_at = ? WHERE id = ?",
                        (props_text, new_summary, timestamp, existing_id)
                    )
                    self.conn.execute(
                        "UPDATE entity_fts SET name = ?, summary = ?, properties = ? WHERE rowid = ?",
                        (name, new_summary, props_text, existing_id)
                    )

                    for k, v in properties.items():
                        if v and k not in old_props:
                            frag = f"{name} {k}: {v}"
                            self._add_fragment(frag, timestamp, user_id)
                            fragments_added += 1
            else:
                props_text = json.dumps(properties, ensure_ascii=False)

                cursor = self.conn.execute(
                    """INSERT INTO entities
                       (name, type, properties, summary, created_at, updated_at, user_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (name, ent_type, props_text, summary, timestamp, timestamp, user_id)
                )
                ent_id = cursor.lastrowid
                entity_id_map[name] = ent_id

                self.conn.execute(
                    "INSERT INTO entity_fts (rowid, name, summary, properties) VALUES (?, ?, ?, ?)",
                    (ent_id, name, summary, props_text)
                )

                if summary:
                    self._add_fragment(f"{name}: {summary}", timestamp, user_id)
                    fragments_added += 1
                for k, v in properties.items():
                    if v:
                        frag = f"{name} {k}: {v}"
                        self._add_fragment(frag, timestamp, user_id)
                        fragments_added += 1

        relations_added = 0
        for rel in relations_data:
            if not isinstance(rel, dict):
                continue
            source_name = str(rel.get("source", "")).strip()
            target_name = str(rel.get("target", "")).strip()
            rel_type = str(rel.get("relation", "")).strip()
            rel_props = rel.get("properties", {})
            if not isinstance(rel_props, dict):
                rel_props = {}

            if not source_name or not target_name or not rel_type:
                continue

            src_id = entity_id_map.get(source_name) or self._find_similar_entity(source_name, user_id)
            tgt_id = entity_id_map.get(target_name) or self._find_similar_entity(target_name, user_id)

            if not src_id or not tgt_id or src_id == tgt_id:
                continue

            cursor = self.conn.execute(
                "SELECT id FROM relations WHERE source_id = ? AND target_id = ? AND rel_type = ?",
                (src_id, tgt_id, rel_type)
            )
            if cursor.fetchone():
                continue

            self.conn.execute(
                """INSERT INTO relations (source_id, target_id, rel_type, properties, weight, created_at, user_id)
                   VALUES (?, ?, ?, ?, 1.0, ?, ?)""",
                (src_id, tgt_id, rel_type, json.dumps(rel_props, ensure_ascii=False), timestamp, user_id)
            )
            relations_added += 1

            frag = f"{source_name} {rel_type} {target_name}"
            self._add_fragment(frag, timestamp, user_id)
            fragments_added += 1

        self.conn.commit()

        return {
            "entities_added": len(entity_id_map),
            "relations_added": relations_added,
            "fragments_added": fragments_added,
        }

    def _add_fragment(self, content: str, timestamp: float, user_id: str):
        content = content.strip()
        if not content or len(content) < 3:
            return

        cursor = self.conn.execute(
            "SELECT id FROM fragments WHERE content = ? AND (user_id = ? OR user_id = '')",
            (content, user_id)
        )
        if cursor.fetchone():
            return

        cursor = self.conn.execute(
            "INSERT INTO fragments (content, created_at, user_id) VALUES (?, ?, ?)",
            (content, timestamp, user_id)
        )
        frag_id = cursor.lastrowid

        self.conn.execute(
            "INSERT INTO fragment_fts (rowid, content) VALUES (?, ?)",
            (frag_id, content)
        )

    def search(self, query: str, user_id: str = "", top_k: int = 30) -> list[dict]:
        """
        Cognitive retrieval strategy:
        1. Flat fragment FTS search (primary, same as baseline)
        2. Entity FTS search + spreading activation (enhancement)
        3. Fusion: flat results by original score, entity results as supplement
        """
        is_cn = contains_cjk(query)
        fts_query = build_fts_query(query) if not is_cn else ""

        # --- Step 1: Flat fragment search (primary) ---
        frag_results = []
        if is_cn:
            rows = cn_like_search(self.conn, query, user_id, top_k * 2, table_name="fragments")
            for r in rows:
                frag_results.append({
                    "memory": r[1],
                    "score": r[3],
                    "type": "fragment",
                    "dedup_key": r[1].lower()[:80],
                })
        elif fts_query:
            try:
                cursor = self.conn.execute(
                    """
                    SELECT f.id, f.content, f.created_at, bm25(fragment_fts) as rank
                    FROM fragment_fts
                    JOIN fragments f ON f.id = fragment_fts.rowid
                    WHERE fragment_fts MATCH ?
                      AND (f.user_id = ? OR f.user_id = '')
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, user_id, top_k * 2)
                )
                rows = cursor.fetchall()
                for row in rows:
                    frag_id, content, created_at, rank = row
                    score = 1.0 / (1.0 + rank) if rank > 0 else 1.0
                    frag_results.append({
                        "memory": content,
                        "score": score,
                        "type": "fragment",
                        "dedup_key": content.lower()[:80],
                    })
            except Exception:
                pass

        if not frag_results:
            cursor = self.conn.execute(
                "SELECT id, content, created_at FROM fragments WHERE user_id = ? OR user_id = '' ORDER BY created_at DESC LIMIT ?",
                (user_id, top_k)
            )
            rows = cursor.fetchall()
            for row in rows:
                frag_results.append({
                    "memory": row[1],
                    "score": 0.5,
                    "type": "fragment",
                    "dedup_key": row[1].lower()[:80],
                })

        # --- Step 2: Entity search + spreading activation (enhancement) ---
        entity_results = []
        if is_cn:
            cn_keywords = extract_cn_keywords(query)
            ent_match_counts = {}
            for kw in cn_keywords:
                cursor = self.conn.execute(
                    "SELECT id, name, type, properties, summary, importance FROM entities "
                    "WHERE (name LIKE ? OR summary LIKE ? OR properties LIKE ?) "
                    "AND (user_id = ? OR user_id = '')",
                    (f"%{kw}%", f"%{kw}%", f"%{kw}%", user_id)
                )
                for row in cursor.fetchall():
                    eid = row[0]
                    if eid not in ent_match_counts:
                        ent_match_counts[eid] = [row, 0]
                    ent_match_counts[eid][1] += 1
            sorted_ents = sorted(ent_match_counts.items(), key=lambda x: -x[1][1])
            for eid, (row, cnt) in sorted_ents[:top_k]:
                entity_results.append((*row, cnt / len(cn_keywords) if cn_keywords else 0.5))
        elif fts_query:
            try:
                cursor = self.conn.execute(
                    """
                    SELECT e.id, e.name, e.type, e.properties, e.summary, e.importance,
                           bm25(entity_fts) as rank
                    FROM entity_fts
                    JOIN entities e ON e.id = entity_fts.rowid
                    WHERE entity_fts MATCH ?
                      AND (e.user_id = ? OR e.user_id = '')
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, user_id, top_k)
                )
                entity_results = cursor.fetchall()
            except Exception:
                pass

        activated = {}
        for row in entity_results:
            ent_id, name, ent_type, props, summary, importance, rank = row
            base_score = 1.0 / (1.0 + rank) if rank > 0 else 1.0
            activated[ent_id] = {
                "score": base_score,
                "name": name,
                "type": ent_type,
                "properties": json.loads(props or "{}"),
                "summary": summary,
                "importance": importance,
                "hops": 0,
            }

        cursor = self.conn.execute(
            "SELECT source_id, target_id, rel_type, weight FROM relations WHERE user_id = ? OR user_id = ''",
            (user_id,)
        )
        all_relations = cursor.fetchall()

        adj = {}
        for src_id, tgt_id, rel_type, weight in all_relations:
            if src_id not in adj:
                adj[src_id] = []
            if tgt_id not in adj:
                adj[tgt_id] = []
            adj[src_id].append((tgt_id, rel_type, weight))
            adj[tgt_id].append((src_id, rel_type, weight))

        decay = 0.5
        new_activated = {}
        for ent_id, info in activated.items():
            if ent_id in adj:
                for neighbor_id, rel_type, weight in adj[ent_id]:
                    if neighbor_id in activated:
                        continue
                    if neighbor_id in new_activated:
                        activation = info["score"] * decay * weight
                        if activation > new_activated[neighbor_id]["score"]:
                            new_activated[neighbor_id]["score"] = activation
                            new_activated[neighbor_id]["via_relation"] = rel_type
                            new_activated[neighbor_id]["via_entity"] = info["name"]
                        continue

                    cursor = self.conn.execute(
                        "SELECT name, type, properties, summary, importance FROM entities WHERE id = ?",
                        (neighbor_id,)
                    )
                    nrow = cursor.fetchone()
                    if nrow:
                        activation = info["score"] * decay * weight
                        new_activated[neighbor_id] = {
                            "score": activation,
                            "name": nrow[0],
                            "type": nrow[1],
                            "properties": json.loads(nrow[2] or "{}"),
                            "summary": nrow[3],
                            "importance": nrow[4],
                            "hops": 1,
                            "via_relation": rel_type,
                            "via_entity": info["name"],
                        }

        activated.update(new_activated)

        entity_memories = []
        for ent_id, info in activated.items():
            props_items = [(k, v) for k, v in info["properties"].items() if v]
            props_text = "; ".join(f"{k}: {v}" for k, v in props_items[:5])

            mem_text = f"[Entity: {info['type']}] {info['name']}"
            if info.get("summary"):
                mem_text += f" — {info['summary']}"
            if props_text:
                mem_text += f" ({props_text})"
            if info.get("hops", 0) > 0:
                mem_text += f" [related to {info['via_entity']} ({info['via_relation']})]"

            hop_factor = 0.8 ** info.get("hops", 0)
            final_score = info["score"] * 0.7 * hop_factor

            entity_memories.append({
                "memory": mem_text,
                "score": final_score,
                "type": "entity",
                "dedup_key": f"entity_{info['name'].lower()[:50]}",
            })

        # --- Step 3: Fusion and dedup ---
        seen = set()
        all_results = []

        for fm in frag_results:
            key = fm["dedup_key"]
            if key not in seen:
                seen.add(key)
                all_results.append(fm)

        for em in entity_memories:
            key = em["dedup_key"]
            if key not in seen:
                seen.add(key)
                all_results.append(em)

        all_results.sort(key=lambda x: x["score"], reverse=True)

        final_results = []
        for r in all_results[:top_k]:
            final_results.append({
                "memory": r["memory"],
                "score": r["score"],
                "type": r.get("type", "unknown"),
            })

        return final_results

    def count_memories(self, user_id: str = "") -> int:
        if user_id:
            cur = self.conn.execute(
                "SELECT COUNT(*) FROM fragments WHERE user_id = ?", (user_id,))
        else:
            cur = self.conn.execute("SELECT COUNT(*) FROM fragments")
        return cur.fetchone()[0]

    def close(self):
        self.conn.close()
