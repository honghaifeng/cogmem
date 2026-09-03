"""
CognitiveMemory: FTS5 + Spreading Activation (no vector retrieval).

This is the intermediate ablation configuration between BaselineMemory
(FTS5 only) and full CogMem (FTS5 + Vector + Spreading Activation).

It shares the same entity-relation network and extraction logic as CogMem,
but skips dense vector embeddings entirely — useful for environments without
SentenceTransformer or for evaluating the contribution of vector retrieval.
"""

from .memory import CogMem


class CognitiveMemory(CogMem):
    """
    FTS5 + Spreading Activation memory (no vector path).

    Same entity-relation extraction and spreading activation as CogMem,
    but without dense vector embeddings. This isolates the contribution of
    symbolic + structured retrieval from neural retrieval.

    Example:
        .. code-block:: python

            from cogmem import CognitiveMemory

            mem = CognitiveMemory(db_path="cognitive.db")
            mem.add("I met Alice at the park yesterday.", user_id="user1")
            results = mem.search("Who did I meet?", user_id="user1")
    """

    def _init_vector_layer(self, embed_model: str):
        self._embed_model = None
        self._embed_dim = 0
        self._embed_matrix = None
        self._frag_id_to_row = {}
        self._row_to_frag_id = {}
        print("[cognitive] Vector retrieval disabled (FTS + Spreading Activation only)")
