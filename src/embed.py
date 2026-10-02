"""Multilingual sentence embeddings (MIT-licensed, ~118M params, well under the 8B cap) as a semantic similarity
signal, on top of the existing TF-IDF blocking (embedding the full 10M-record pool measured at ~375 sentences/sec
on this CPU -> 8-10+ hours, infeasible; so embeddings are computed only for S1 queries and their already-retrieved
candidates, never for retrieval over the full pool)."""
import time
import numpy as np

_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(20)
        _MODEL = SentenceTransformer("intfloat/multilingual-e5-small")
    return _MODEL


def embed_texts(texts, batch_size=128, prefix="query: "):
    t0 = time.time()
    emb = model().encode([prefix + t for t in texts], batch_size=batch_size, show_progress_bar=False,
                         normalize_embeddings=True, convert_to_numpy=True)
    print(f"  [embed] {len(texts)} texts in {time.time() - t0:.0f}s ({len(texts) / max(time.time() - t0, 1e-6):.0f}/s)", flush=True)
    return emb.astype(np.float32)


def pair_cosine(s1, pool, pairs, text_col="name_full", batch_size=128):
    """Cosine similarity between the S1 record's and the candidate's embedding, for every pair. Embeds only the
    UNIQUE texts that actually appear among the pairs' S1 / pool rows (never the whole pool)."""
    si, ci = pairs.s1_idx.to_numpy(), pairs.c_idx.to_numpy()
    us1, us1_inv = np.unique(si, return_inverse=True)
    uc, uc_inv = np.unique(ci, return_inverse=True)
    print(f"  [embed] {len(us1)} unique S1 texts, {len(uc)} unique candidate texts", flush=True)
    Es1 = embed_texts(s1[text_col].to_numpy()[us1].tolist(), batch_size)
    Ec = embed_texts(pool[text_col].to_numpy()[uc].tolist(), batch_size)
    return np.einsum("ij,ij->i", Es1[us1_inv], Ec[uc_inv])
