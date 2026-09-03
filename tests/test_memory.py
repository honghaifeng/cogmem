"""
Unit tests for CogMem core functionality.

These tests use a mock LLM client to avoid API calls.
Run with: pytest tests/
"""

import os
import tempfile
import json
from unittest.mock import MagicMock

from cogmem import CogMem, BaselineMemory
from cogmem.llm_client import LLMClient
from cogmem.search_utils import contains_cjk, build_fts_query, extract_cn_keywords


class TestSearchUtils:
    def test_contains_cjk(self):
        assert contains_cjk("你好世界") is True
        assert contains_cjk("Hello World") is False
        assert contains_cjk("Hello 世界") is True

    def test_build_fts_query(self):
        q = build_fts_query("What did Alice eat?")
        assert "Alice" in q
        assert "OR" in q

    def test_extract_cn_keywords(self):
        kws = extract_cn_keywords("小王推荐了什么书？")
        assert len(kws) > 0
        assert any("小王" in k for k in kws)

    def test_contains_cjk_empty(self):
        assert contains_cjk("") is False


class TestBaselineMemory:
    def setup_method(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(self.db_fd)

        self.llm = MagicMock(spec=LLMClient)
        self.llm.chat_completion_json.return_value = {
            "memories": [
                "Alice works at Google",
                "Bob likes pizza",
            ]
        }
        self.llm.chat_completion.return_value = "I don't have that record."

        self.mem = BaselineMemory(db_path=self.db_path, llm_client=self.llm)

    def teardown_method(self):
        self.mem.close()
        os.unlink(self.db_path)

    def test_add_and_count(self):
        result = self.mem.add("Alice works at Google. Bob likes pizza.", user_id="u1")
        assert self.mem.count_memories("u1") >= 2

    def test_search_english(self):
        self.mem.add("Alice works at Google. Bob likes pizza.", user_id="u1")
        results = self.mem.search("Where does Alice work?", user_id="u1")
        assert len(results) > 0

    def test_search_empty(self):
        results = self.mem.search("nonexistent", user_id="u1")
        assert isinstance(results, list)


class TestCogMem:
    def setup_method(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(self.db_fd)

        self.llm = MagicMock(spec=LLMClient)
        self.llm.chat_completion_json.return_value = {
            "entities": [
                {"name": "Alice", "type": "person", "properties": {"job": "engineer"}, "summary": "Alice is an engineer"},
                {"name": "Google", "type": "organization", "properties": {}, "summary": "Tech company"},
            ],
            "relations": [
                {"source": "Alice", "target": "Google", "relation": "works_at", "properties": {}},
            ],
            "facts": [
                "Alice works at Google as an engineer",
                "Alice lives in San Francisco",
            ],
        }
        self.llm.chat_completion.return_value = "Alice works at Google."

        self.mem = CogMem(db_path=self.db_path, llm_client=self.llm, embed_model="")

    def teardown_method(self):
        self.mem.close()
        os.unlink(self.db_path)

    def test_add_entities_and_relations(self):
        result = self.mem.add("Alice works at Google as an engineer.", user_id="u1")
        assert result["entities_added"] >= 1
        assert result["relations_added"] >= 1
        assert result["fragments_added"] >= 1

    def test_search(self):
        self.mem.add("Alice works at Google as an engineer.", user_id="u1")
        results = self.mem.search("Where does Alice work?", user_id="u1")
        assert len(results) > 0

    def test_count_memories(self):
        self.mem.add("Alice works at Google.", user_id="u1")
        assert self.mem.count_memories("u1") > 0

    def test_multi_user_isolation(self):
        self.mem.add("Alice works at Google.", user_id="u1")
        self.mem.add("Bob works at Apple.", user_id="u2")
        u1_count = self.mem.count_memories("u1")
        u2_count = self.mem.count_memories("u2")
        assert u1_count > 0
        assert u2_count > 0
