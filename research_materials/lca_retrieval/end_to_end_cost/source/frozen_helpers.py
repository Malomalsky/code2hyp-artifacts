from __future__ import annotations
import io, tokenize
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
K1, B = 1.2, .75

def lexical_token_stream(source: str) -> tuple[str, ...]:
    """Return D1 tokens with comments and formatting removed."""

    ignored = {
        tokenize.ENCODING,
        tokenize.ENDMARKER,
        tokenize.INDENT,
        tokenize.DEDENT,
        tokenize.NEWLINE,
        tokenize.NL,
        tokenize.COMMENT,
    }
    tokens: list[str] = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in ignored:
            continue
        if token.type == tokenize.ERRORTOKEN and token.string.isspace():
            continue
        tokens.append(f"{tokenize.tok_name[token.type]}:{token.string}")
    return tuple(tokens)

def vectorize(bags, vocabulary):
    data, indices, indptr = [], [], [0]
    for bag in bags:
        for key, value in bag.items():
            if key in vocabulary:
                indices.append(vocabulary[key])
                data.append(value)
        indptr.append(len(data))
    return csr_matrix((data, indices, indptr), shape=(len(bags), len(vocabulary)), dtype=np.float64)

def normalized(matrix):
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A1
    return matrix.multiply(1 / np.where(norms > 0, norms, 1)[:, None]).tocsr()

def weighted_gallery(counts):
    counts = counts.tocsr().astype(np.float64)
    assert np.isfinite(counts.data).all() and (counts.data > 0).all()
    lengths = counts.sum(axis=1).A1
    if len(lengths) == 0 or lengths.mean() == 0:
        raise ValueError("BM25 requires a nonempty gallery with some tokens")
    df = (counts > 0).sum(axis=0).A1
    idf = np.log1p((len(lengths) - df + .5) / (df + .5))
    norm = K1 * (1 - B + B * lengths / lengths.mean())
    weighted = counts.copy()
    weighted.data = (idf[counts.indices] * (K1 + 1) * counts.data
                     / (counts.data + np.repeat(norm, np.diff(counts.indptr))))
    return weighted

def minmax(scores):
    scores = np.asarray(scores, dtype=np.float64)
    assert np.isfinite(scores).all()
    span = scores.max() - scores.min()
    return (scores - scores.min()) / span if span > 0 else np.zeros_like(scores)

def hybrid(dense, lexical, alpha):
    if alpha == 0:
        return dense
    if alpha == 1:
        return lexical
    return (1-alpha) * minmax(dense) + alpha * minmax(lexical)

def _node_token(tree: RawAstTree, node: NodeId, *, terminal_policy: TerminalPolicy = "type") -> str:
    attributes = tree.attributes.get(node, {})
    terminal = attributes.get("terminal_token")
    if tree.labels.get(node) == "TerminalToken" and terminal is not None:
        if terminal_policy == "value":
            return f"terminal:{terminal}"
        if terminal_policy == "class":
            return f"terminal_class:{_terminal_class(terminal)}"
    return f"node:{tree.labels.get(node, '')}"

def _terminal_class(token: str) -> str:
    if token == "":
        return "empty"
    if token.isidentifier():
        return "identifier"
    try:
        float(token)
    except ValueError:
        return "symbol"
    return "number"

def path_keys(tree, path, vocabulary):
    ancestor = path.lca(tree)
    nodes = path.nodes
    assert nodes.count(ancestor) == 1 and ancestor not in (path.start, path.end)
    tokens = [_node_token(tree, n, terminal_policy="class") for n in nodes]
    labels = [vocabulary.get(t, vocabulary["<unk>"]) for t in tokens]
    retained = tuple((int(n == ancestor), label) for n, label in zip(nodes, labels))
    masked = tuple((marker, -1 if marker else label) for marker, label in retained)
    assert sum(a != b for a, b in zip(retained, masked)) == 1
    return min(retained, retained[::-1]), min(masked, masked[::-1]), sum(t not in vocabulary for t in tokens)

def fragments(sequence):
    return Counter(min(sequence[i:i+n], sequence[i:i+n][::-1])
                   for n in (2, 3) for i in range(len(sequence) - n + 1))
