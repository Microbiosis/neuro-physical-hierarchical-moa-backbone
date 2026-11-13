#!/usr/bin/env python3
"""载荷存活率：从文档到上下文，载荷要过三道关口，每道都在丢东西。

    切块   文档被切成固定长度的块，载荷可能被切断；
    检索   只取 top-k 个块，包含载荷的块不一定被选中；
    截断   上下文有预算，末尾的块会被丢掉。

语料按主题分组（finance / security / hr / ops），每篇文档属于一个主题。
用户查询是主题词（"budget revenue expense"），攻击者把载荷写成**同主题但含指令标记**的样子，
蹭用户的查询被检索到。载荷块和同主题的正常块竞争 top-k——检索是一个会丢东西的关口。

要验证的 4 件事：

    ① 预算越大，完整存活率越高；
    ② 块更细时载荷更容易被切断；
    ③ 任何配置下存活率都不是 100%；
    ④ 存活率足以跨过 Allee 阈值。

    python3 payload_survival.py            # 五种配置
    python3 payload_survival.py --check
"""

from __future__ import annotations

import argparse

import numpy as np

import payload_core as core

DOC_WORDS = 220
CORPUS = 60
PAYLOAD_WORDS = 40
PAYLOAD_MARKERS = 8
TRIALS = 40

TOPICS = {
    "finance":  ["budget", "revenue", "expense", "quarterly", "forecast",
                 "margin", "profit", "audit"],
    "security": ["firewall", "breach", "patch", "vulnerability", "incident",
                 "monitor", "alert", "scan"],
    "hr":       ["onboard", "review", "policy", "benefit", "leave",
                 "training", "roster", "handbook"],
    "ops":      ["deploy", "rollback", "pipeline", "monitor", "downtime",
                 "capacity", "scaling", "release"],
}
TOPIC_NAMES = list(TOPICS.keys())
GENERIC = ["the", "and", "for", "our", "this", "with", "from", "about",
           "team", "please", "update", "report"]
_MARKERS = [f"m{i}" for i in range(PAYLOAD_MARKERS)]


def make_corpus(rng, payload_at=None):
    """按主题生成语料；payload_at=(doc, position) 时在那篇文档的那个位置插入载荷。

    载荷用与文档相同的主题词填充（贴着用户会搜的东西写），
    末尾加 PAYLOAD_MARKERS 个唯一标记（代表指令性内容——真实注入里的"请执行…"部分）。
    标记越多→TF-IDF 得分越低→越容易被同主题普通块挤出 top-k。
    """
    documents = []
    for doc in range(CORPUS):
        topic = TOPIC_NAMES[doc % len(TOPIC_NAMES)]
        vocab = TOPICS[topic] + GENERIC
        indices = rng.integers(len(vocab), size=DOC_WORDS)
        words = [vocab[i] for i in indices]
        if payload_at and payload_at[0] == doc:
            topic_vocab = TOPICS[topic]
            n_topic = PAYLOAD_WORDS - PAYLOAD_MARKERS
            bait = [topic_vocab[i] for i in rng.integers(len(topic_vocab), size=n_topic)]
            words[payload_at[1]:payload_at[1]] = bait + _MARKERS
        documents.append(" ".join(words))
    return documents


def query_for_doc(doc_index):
    """用户查询：该文档主题的前 3 个词。"""
    topic = TOPIC_NAMES[doc_index % len(TOPIC_NAMES)]
    return TOPICS[topic][:3]


def chunk(documents, size, stride=None):
    stride = size if stride is None else stride
    chunks = []
    for doc, text in enumerate(documents):
        words = text.split()
        for start in range(0, max(1, len(words) - size + 1), stride):
            chunks.append((doc, start, words[start:start + size]))
    return chunks


def score(query, chunks):
    """TF-IDF 余弦（向量化）。"""
    vocabulary = {}
    for _, _, words in chunks:
        for word in words:
            vocabulary.setdefault(word, len(vocabulary))
    vocab_size = len(vocabulary)
    num_chunks = len(chunks)
    tf = np.zeros((num_chunks, vocab_size))
    for idx, (_, _, words) in enumerate(chunks):
        for word in words:
            tf[idx, vocabulary[word]] += 1
    df = (tf > 0).sum(axis=0).astype(float)
    idf = np.log((1 + num_chunks) / (1 + df)) + 1.0
    tfidf = tf * idf
    query_vec = np.zeros(vocab_size)
    for word in query:
        if word in vocabulary:
            query_vec[vocabulary[word]] += 1
    query_vec = query_vec * idf
    norms = np.linalg.norm(tfidf, axis=1) * np.linalg.norm(query_vec) + 1e-12
    return (tfidf @ query_vec) / norms


def assemble(documents, query_words, *, size=96, top_k=6, budget=None):
    """切块 → 检索 → 取 top-k → 按预算截断，返回进入上下文的词。"""
    chunks = chunk(documents, size)
    ranking = np.argsort(-score(query_words, chunks))[:top_k]
    budget = size * top_k if budget is None else budget
    context = []
    for index in ranking:
        context.extend(chunks[index][2])
        if len(context) >= budget:
            break
    return context[:budget]


def survival(documents, payload_at, query_words, *, size, top_k, budget):
    """载荷的唯一标记有多少比例真正进了上下文。

    只计量 marker 词（m0–m7）——它们只存在于载荷块中，不会从同主题的普通块泄露。
    主题词在任何同主题块里都出现，用它们做计量会虚高。
    """
    context = assemble(documents, query_words, size=size, top_k=top_k, budget=budget)
    present = set(context)
    return sum(1 for marker in _MARKERS if marker in present) / max(len(_MARKERS), 1)


def experiment(size, top_k, budget, trials=TRIALS, seed=0):
    """返回（平均完整度，完整存活的概率）。"""
    rng = np.random.default_rng(seed)
    values = []
    for trial in range(trials):
        doc = int(rng.integers(0, CORPUS))
        position = int(rng.integers(0, DOC_WORDS - 1))
        documents = make_corpus(rng, payload_at=(doc, position))
        query = query_for_doc(doc)
        values.append(survival(documents, (doc, position), query,
                               size=size, top_k=top_k, budget=budget))
    values = np.array(values)
    return float(values.mean()), float((values > 0.999).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    rows = []
    for label, size, top_k, budget in (
            ("紧预算 128 词", 96, 8, 128),
            ("中预算 384 词", 96, 8, 384),
            ("宽预算 768 词", 96, 8, 768),
            ("重切块 48 词", 48, 12, 384),
            ("细切块 192 词", 192, 8, 384)):
        mean, full = experiment(size, top_k, budget)
        rows.append({"setting": label, "size": size, "top_k": top_k,
                     "survival": mean, "full": full})

    print(f"{'配置':16s} {'块长':>4s} {'预算':>5s} {'平均完整度':>10s} {'完整存活':>8s}")
    for row in rows:
        print(f"{row['setting']:16s} {row['size']:4d} {'':>5s} "
              f"{row['survival']:10.1%} {row['full']:8.0%}")

    if args.check:
        by = {row["setting"]: row for row in rows}
        checks = [
            ("预算越大，完整存活的概率越高",
             by["宽预算 768 词"]["full"] > by["紧预算 128 词"]["full"]),
            ("块更细时，载荷更容易被切断（48 词 < 192 词）",
             by["重切块 48 词"]["survival"] < by["细切块 192 词"]["survival"]),
            ("任何配置下都不是 100%（投递一定丢东西）",
             all(row["survival"] < 0.99 for row in rows)),
            ("存活率足以跨过 Allee（> 6%）", by["宽预算 768 词"]["survival"] > 0.06),
        ]
        raise SystemExit(core.report(checks))


if __name__ == "__main__":
    main()
