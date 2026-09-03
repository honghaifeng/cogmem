"""
Search utilities for Chinese and English text retrieval.

Provides CJK detection, Chinese keyword extraction, date-aware LIKE search,
and FTS5 query construction for English text.
"""

import re
from datetime import datetime, timezone


_CN_STOP_WORDS = {
    '的', '了', '是', '我', '你', '他', '她', '它', '和', '与', '过',
    '什么', '哪', '哪个', '哪里', '多少', '在', '有', '不', '也', '都',
    '这', '那', '个', '就', '还', '把', '被', '给', '对', '到',
    '吗', '呢', '吧', '啊', '哦', '嗯', '要', '会', '能', '可以',
    '着', '地', '得', '所', '以', '但', '而', '及', '或',
    '跟', '同', '向', '从', '往', '为', '于', '由', '按',
    '关于', '根据', '通过', '按照', '对于', '至于', '虽然', '因为', '所以',
    '如果', '尽管', '即使', '除非', '只要', '只有', '无论',
    '怎么', '怎样', '如何', '为何', '为什么', '谁', '哪位',
    '何时', '何地', '何事', '何物', '哪种', '哪些', '几', '几号',
    '一次', '一样', '这种', '那种', '这样', '那样',
    '曾经', '通常', '一般', '之前', '之后', '之间',
    '一个', '一部', '一本', '一首', '一位', '一种',
    '我和', '和你', '你聊', '聊了', '聊到', '到了', '了一',
    '分享', '享过', '过我', '我看', '我曾', '经在', '在4',
    '月2', '号这', '这天', '我和你', '你分享',
}

_LOW_VALUE_BIGRAMS = {
    '我和', '和你', '你聊', '聊了', '聊到', '到了', '了一',
    '我曾', '经在', '在4', '月2', '号这', '这天',
    '你分', '分享', '享过', '过我', '我看',
    '你分', '享过', '是一', '是一',
    '我通', '常打', '打哪', '哪个', '个位',
    '是什', '什么', '么', '名的',
}


def contains_cjk(text: str) -> bool:
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


def extract_cn_keywords(query: str) -> list[str]:
    proper_nouns = []
    for m in re.finditer(r'《([^》]+)》', query):
        proper_nouns.append(m.group(1))
    for m in re.finditer(r'[A-Za-z]{2,}', query):
        proper_nouns.append(m.group(0))

    date_patterns = []
    for m in re.finditer(r'\d{1,2}月\d{1,2}[日号]', query):
        date_patterns.append(m.group(0))
    for m in re.finditer(r'(?<!\d)\d{1,2}月(?!d)', query):
        date_patterns.append(m.group(0))
    for m in re.finditer(r'\d{4}年', query):
        date_patterns.append(m.group(0))
    for m in re.finditer(r'\d{1,2}日', query):
        date_patterns.append(m.group(0))

    parts = re.split(
        r"[\uff0c\u3002\uff1f\uff01\u3001\uff1b\uff1a\u201c\u201d\u2018\u2019\uff08\uff09\(\)\[\]\u3010\u3011\u300a\u300b\s\?\.\,\!\;\:\'\-—_/\\]+",
        query
    )

    core_terms = []
    for part in parts:
        part = part.strip()
        if len(part) < 2 or part in _CN_STOP_WORDS or part.isdigit():
            continue
        if 2 <= len(part) <= 3:
            if part not in _CN_STOP_WORDS and part not in _LOW_VALUE_BIGRAMS:
                core_terms.append(part)
        elif len(part) > 3:
            if part not in _CN_STOP_WORDS:
                core_terms.append(part)
            for j in range(len(part) - 1):
                bi = part[j:j+2]
                if len(bi) >= 2 and bi not in _CN_STOP_WORDS and bi not in _LOW_VALUE_BIGRAMS:
                    if re.match(r'^[\d月年日号]$', bi[0]) and re.match(r'^[\d月年日号]$', bi[1]):
                        continue
                    core_terms.append(bi)
            if len(part) >= 5:
                tri_start = part[:3]
                tri_end = part[-3:]
                for tri in [tri_start, tri_end]:
                    if tri not in _CN_STOP_WORDS and len(tri) >= 3 and tri not in _LOW_VALUE_BIGRAMS:
                        core_terms.append(tri)

    all_keywords = []
    seen = set()
    for kw in proper_nouns:
        if kw not in seen and len(kw) >= 2:
            seen.add(kw)
            all_keywords.append(kw)
    for kw in date_patterns:
        if kw not in seen:
            seen.add(kw)
            all_keywords.append(kw)
    for kw in core_terms:
        if kw not in seen:
            seen.add(kw)
            all_keywords.append(kw)

    if len(all_keywords) > 20:
        result = []
        result.extend(all_keywords[:len(proper_nouns) + len(date_patterns)])
        remaining = all_keywords[len(proper_nouns) + len(date_patterns):]
        result.extend(remaining[:20 - len(result)])
        all_keywords = result

    return all_keywords[:25]


def extract_date_from_query(query: str) -> str | None:
    m = re.search(r'(\d{1,2})月(\d{1,2})[日号]', query)
    if m:
        month = m.group(1).zfill(2)
        day = m.group(2).zfill(2)
        return f"{month}月{day}日"
    m = re.search(r'(\d{1,2})月(\d{1,2})号', query)
    if m:
        month = m.group(1).zfill(2)
        day = m.group(2).zfill(2)
        return f"{month}月{day}日"
    m = re.search(r'(\d{1,2})日', query)
    if m:
        day = m.group(1).zfill(2)
        return f"{day}日"
    return None


def cn_like_search(conn, query: str, user_id: str, top_k: int, table_name: str = "memories",
                   reference_year: int = 2023) -> list[tuple]:
    keywords = extract_cn_keywords(query)
    query_date = extract_date_from_query(query)

    content_col = "content"

    if not keywords:
        cursor = conn.execute(
            f"SELECT id, {content_col}, created_at, 0.0 as score FROM {table_name} "
            f"WHERE user_id = ? OR user_id = '' ORDER BY created_at DESC LIMIT ?",
            (user_id, top_k)
        )
        return cursor.fetchall()

    match_counts = {}
    for kw in keywords:
        like_pattern = f"%{kw}%"
        try:
            cursor = conn.execute(
                f"SELECT id, {content_col}, created_at FROM {table_name} "
                f"WHERE {content_col} LIKE ? AND (user_id = ? OR user_id = '')",
                (like_pattern, user_id)
            )
            for row in cursor.fetchall():
                mem_id = row[0]
                if mem_id not in match_counts:
                    match_counts[mem_id] = [row[1], row[2], 0, set()]
                match_counts[mem_id][2] += 1
                match_counts[mem_id][3].add(kw)
        except Exception:
            continue

    if query_date:
        date_patterns_to_try = [query_date]
        parts = re.match(r'(\d+)月(\d+)日', query_date)
        if parts:
            m, d = parts.group(1), parts.group(2)
            if m.startswith('0'):
                date_patterns_to_try.append(f"{int(m)}月{d}日")
                date_patterns_to_try.append(f"{int(m)}月{int(d)}日")
            date_patterns_to_try.append(f"{m}月{d}日")
            date_patterns_to_try.append(f"{m}月{int(d)}日")

        for dp in date_patterns_to_try:
            like_pattern = f"%{dp}%"
            try:
                cursor = conn.execute(
                    f"SELECT id, {content_col}, created_at FROM {table_name} "
                    f"WHERE {content_col} LIKE ? AND (user_id = ? OR user_id = '')",
                    (like_pattern, user_id)
                )
                for row in cursor.fetchall():
                    mem_id = row[0]
                    if mem_id not in match_counts:
                        match_counts[mem_id] = [row[1], row[2], 0, set()]
                    match_counts[mem_id][2] += 2
                    match_counts[mem_id][3].add(f"date:{dp}")
            except Exception:
                continue

    if query_date:
        m = re.match(r'(\d+)月(\d+)日', query_date)
        if m:
            month, day = int(m.group(1)), int(m.group(2))
            try:
                from datetime import datetime as _dt, timezone as _tz
                start_dt = _dt(reference_year, month, day, tzinfo=_tz.utc)
                end_dt = _dt(reference_year, month, day, 23, 59, 59, tzinfo=_tz.utc)
                start_ts = start_dt.timestamp()
                end_ts = end_dt.timestamp()

                for mem_id in match_counts:
                    created_at = match_counts[mem_id][1]
                    if start_ts <= created_at <= end_ts:
                        match_counts[mem_id][2] += 3
                        match_counts[mem_id][3].add(f"ts_date:{month}月{day}日")

                try:
                    cursor = conn.execute(
                        f"SELECT id, {content_col}, created_at FROM {table_name} "
                        f"WHERE created_at >= ? AND created_at <= ? "
                        f"AND (user_id = ? OR user_id = '')",
                        (start_ts, end_ts, user_id)
                    )
                    for row in cursor.fetchall():
                        mem_id = row[0]
                        if mem_id not in match_counts:
                            match_counts[mem_id] = [row[1], row[2], 2, {f"ts_date:{month}月{day}日"}]
                except Exception:
                    pass
            except Exception:
                pass

    sorted_matches = sorted(
        match_counts.items(),
        key=lambda x: (-x[1][2], -x[1][1])
    )

    results = []
    for mem_id, (content, created_at, count, matched_kws) in sorted_matches[:top_k]:
        score = count / max(len(keywords), 1)
        results.append((mem_id, content, created_at, score))

    if len(results) < max(top_k // 2, 10):
        existing_ids = {r[0] for r in results}
        try:
            if existing_ids:
                placeholders = ",".join("?" * len(existing_ids))
                cursor = conn.execute(
                    f"SELECT id, {content_col}, created_at FROM {table_name} "
                    f"WHERE (user_id = ? OR user_id = '') AND id NOT IN ({placeholders}) "
                    f"ORDER BY created_at DESC LIMIT ?",
                    (user_id, *existing_ids, top_k - len(results))
                )
            else:
                cursor = conn.execute(
                    f"SELECT id, {content_col}, created_at FROM {table_name} "
                    f"WHERE (user_id = ? OR user_id = '') "
                    f"ORDER BY created_at DESC LIMIT ?",
                    (user_id, top_k - len(results))
                )
            for row in cursor.fetchall():
                results.append((row[0], row[1], row[2], 0.05))
        except Exception:
            pass

    return results[:top_k]


def build_fts_query(query: str) -> str:
    words = re.findall(r'[a-zA-Z]+', query)
    if not words:
        return ""
    words = [w for w in words if len(w) >= 2]
    if not words:
        return ""
    return " OR ".join(words)
