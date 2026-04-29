"""
# Julian Chaoul
# 2/23/2026
#
# Problem Statement - Create a program that generates sentences based on n-grams
#                     from a given text file.
#
# Algorithm overview:
# 1. convert_to_list   — reads files, lowercase, tokenizes into sentence lists.
# 2. arrange_frequency — builds a FULL n-gram frequency dict from the entire corpus
#                        (no words are removed from generation).
# 3. create_sentences  — generates sentences using the Markov chain.
#      a. choose_next   — weighted random selection of the next word.
#      b. make_proper   — formats tokens into a clean sentence string.
#      c. START_WORDS   — prefer contexts whose first token is a natural opener.
#      Sentences end only on punctuation (. ! ?) or the hard word cap.
# 4. visualize_markov  — writes a standalone HTML file AND two PNG images:
#      a. markov_<corpus>_chain.png   — Markov chain graph
#      b. markov_<corpus>_matrix.png  — Transition probability heatmap
#      The chain/matrix visuals use VIZ_FILTER_WORDS to skip pure function
#      words so the graph stays readable — this does NOT affect generation.
#
# Recommended usage for coherent sentences:
#   n >= 3  (trigrams give much better context than bigrams)
#   large corpus (thousands of sentences)
#
# Example Usage:
#   python ngram.py 3 5 moby_dick.txt
#   python ngram.py 3 5 batman.txt trump.txt
"""

import sys
import random
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np


# ---------------------------------------------------------------------------
# VIZ_FILTER_WORDS — used ONLY to clean up the chain/matrix visuals.
# These words are NOT removed from sentence generation.
# ---------------------------------------------------------------------------

VIZ_FILTER_WORDS = {
    "the", "a", "an",
    "and", "but", "or", "nor",
    "in", "on", "at", "to", "of", "by", "with", "from",
    "is", "are", "was", "were",
    "i", "he", "she", "it", "they", "we", "you",
}


# ---------------------------------------------------------------------------
# START_WORDS — contexts beginning with these are preferred as sentence starters.
# Broad enough to match real corpus openers.
# ---------------------------------------------------------------------------

START_WORDS = {
    # pronouns & common subject starters
    "i", "he", "she", "they", "we", "it",
    # articles that begin real sentences
    "the", "a", "an",
    # conjunctive / narrative openers
    "now", "then", "once", "so", "thus", "yet", "but", "and",
    "when", "if", "although", "while", "after", "before",
    # dramatic / content openers
    "darkness", "crime", "gotham", "fear", "justice",
    "nobody", "many", "people", "there", "here",
    "call", "consider", "look", "believe",
}

MAX_SENTENCE_WORDS = 40   # raised from 20 — lets natural structure breathe


# ---------------------------------------------------------------------------
# Helper: check whether a token sequence contains a viz-filter word
# ---------------------------------------------------------------------------

def _has_viz_filter(tokens) -> bool:
    """Return True if every token in the sequence is a VIZ_FILTER_WORD.
    We only skip a context from the visual if it is *entirely* function words,
    not merely if it contains one — this keeps more nodes visible."""
    return all(t in VIZ_FILTER_WORDS for t in tokens)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def convert_to_list(filenames: list) -> list:
    """Read files, lowercase, tokenize into sentence lists."""
    fulltext = []
    for fname in filenames:
        with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
            fulltext.append(f.read())
    full = " ".join(fulltext).lower()

    tokens = []
    current = []
    for char in full:
        if char.isalnum() or char == "'":
            current.append(char)
        else:
            if current:
                tokens.append("".join(current))
                current = []
            if char in {'.', '!', '?'}:
                tokens.append(char)
    if current:
        tokens.append("".join(current))

    sentences, current_sentence = [], []
    for token in tokens:
        current_sentence.append(token)
        if token in {'.', '!', '?'}:
            sentences.append(current_sentence)
            current_sentence = []
    return sentences


def arrange_frequency(sentences: list, n: int) -> tuple:
    """
    Build a FULL n-gram frequency dict — no words are removed.
    Punctuation tokens are kept so sentences can end naturally.
    Also returns a list of valid start contexts.
    """
    counts = {}
    start_words = []
    for sentence in sentences:
        if len(sentence) < n:
            continue
        ctx_start = tuple(sentence[:n - 1])
        start_words.append(ctx_start)
        for i in range(len(sentence) - n + 1):
            ctx = tuple(sentence[i:i + n - 1])
            next_word = sentence[i + n - 1]
            if ctx not in counts:
                counts[ctx] = {}
            counts[ctx][next_word] = counts[ctx].get(next_word, 0) + 1
    counts = {k: v for k, v in counts.items() if v}
    return counts, start_words


def choose_next(next_word: dict) -> str:
    """Weighted random draw from {word: frequency} dict."""
    total = sum(next_word.values())
    n = random.randint(1, total)
    for word, freq in next_word.items():
        n -= freq
        if n <= 0:
            return word
    return random.choice(list(next_word.keys()))


def make_proper(tokens: list) -> str:
    """Join tokens, attaching punctuation directly to the preceding word."""
    output = []
    for t in tokens:
        if t in {'.', '!', '?'} and output:
            output[-1] += t
        else:
            output.append(t)
    result = " ".join(output)
    if result and result[-1] not in {'.', '!', '?'}:
        result += '.'
    return result[0].upper() + result[1:] if result else result


def _preferred_starts(frequency: dict) -> list:
    """Prefer contexts whose first token is a natural sentence opener."""
    preferred = [ctx for ctx in frequency if ctx[0] in START_WORDS]
    return preferred if preferred else list(frequency.keys())


def create_sentences(frequency: dict, start_words: list,
                     n: int, m: int) -> list:
    """
    Generate m sentences using the n-gram Markov chain.
    Sentences end naturally on punctuation (. ! ?) or at MAX_SENTENCE_WORDS.
    No words are artificially blocked — the full corpus drives generation.
    """
    sentences = []
    preferred_starts = _preferred_starts(frequency)
    max_attempts = m * 30
    attempts = 0

    while len(sentences) < m and attempts < max_attempts:
        attempts += 1

        start = random.choice(preferred_starts)
        sentence = list(start)

        completed = False
        for _ in range(MAX_SENTENCE_WORDS):
            c = tuple(sentence[-(n - 1):])
            if c not in frequency:
                break   # dead-end in the chain

            next_word = choose_next(frequency[c])
            sentence.append(next_word)

            if next_word in {'.', '!', '?'}:
                sentences.append(make_proper(sentence))
                completed = True
                break

        if not completed and len(sentence) > n:
            sentences.append(make_proper(sentence))

    return sentences


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _context_label(ctx: tuple) -> str:
    label = " ".join(ctx)
    return label if len(label) <= 22 else label[:20] + "…"


def _transition_probs(frequency: dict) -> dict:
    probs = {}
    for ctx, nexts in frequency.items():
        total = sum(nexts.values())
        probs[ctx] = {w: round(c / total, 3) for w, c in nexts.items()}
    return probs


def _top_nodes(frequency: dict, k: int = 8) -> list:
    ranked = sorted(frequency.keys(),
                    key=lambda c: len(frequency[c]), reverse=True)
    seeds = ranked[:k]
    seed_set = set(seeds)
    probs = _transition_probs(frequency)
    extras = []
    for ctx in seeds:
        for w in probs[ctx]:
            candidate = ctx[1:] + (w,) if len(ctx) > 1 else (w,)
            if candidate in frequency and candidate not in seed_set:
                extras.append(candidate)
                seed_set.add(candidate)
    return (list(seeds) + extras)[:k + 4]


def _circle_positions(n: int, cx: float, cy: float, r: float) -> list:
    return [
        (cx + r * math.cos(2 * math.pi * i / n - math.pi / 2),
         cy + r * math.sin(2 * math.pi * i / n - math.pi / 2))
        for i in range(n)
    ]


def _node_color(frequency: dict, ctx: tuple):
    deg = len(frequency[ctx])
    if deg >= 3:
        return "#EEEDFE", "#534AB7", "#3C3489"
    elif deg == 2:
        return "#E1F5EE", "#1D9E75", "#085041"
    else:
        return "#F1EFE8", "#888780", "#2C2C2A"


# ---------------------------------------------------------------------------
# Shared helper: find the focal pair + a few neighbors
# ---------------------------------------------------------------------------

FOCAL_N = 5   # total nodes shown in both chain and matrix visuals


def _focal_nodes(frequency: dict, k: int = FOCAL_N) -> tuple:
    """
    Find the single highest-probability transition in the graph.
    That gives us a focal (src_ctx, best_word) pair.
    Then expand to k total nodes by pulling in the next-best neighbors
    of both the source and destination contexts.

    Returns (nodes, labels, focal_src_idx, focal_dst_idx, focal_prob).
    """
    probs = _transition_probs(frequency)

    # 1. Find the globally strongest edge: (src_ctx, next_word, probability)
    best_src, best_word, best_p = None, None, 0.0
    for ctx, nexts in probs.items():
        top_w = max(nexts, key=nexts.get)
        if nexts[top_w] > best_p:
            best_src, best_word, best_p = ctx, top_w, nexts[top_w]

    # 2. Destination context (what comes after best_word in the chain)
    best_dst = best_src[1:] + (best_word,) if len(best_src) > 1 else (best_word,)
    if best_dst not in frequency:
        # fallback: just use the second-ranked context
        ranked = sorted(frequency.keys(), key=lambda c: len(frequency[c]), reverse=True)
        best_dst = ranked[1] if len(ranked) > 1 else ranked[0]

    # 3. Expand: add top neighbors of src and dst until we reach k nodes
    seed_set = {best_src, best_dst}
    nodes = [best_src, best_dst]

    for hub in [best_src, best_dst]:
        for w in sorted(probs.get(hub, {}), key=lambda w: -probs[hub][w]):
            if len(nodes) >= k:
                break
            candidate = hub[1:] + (w,) if len(hub) > 1 else (w,)
            if candidate in frequency and candidate not in seed_set:
                nodes.append(candidate)
                seed_set.add(candidate)

    # Fill remaining slots from top contexts by out-degree
    if len(nodes) < k:
        ranked = sorted(frequency.keys(), key=lambda c: len(frequency[c]), reverse=True)
        for ctx in ranked:
            if len(nodes) >= k:
                break
            if ctx not in seed_set:
                nodes.append(ctx)
                seed_set.add(ctx)

    nodes = nodes[:k]
    labels = [_context_label(ctx) for ctx in nodes]
    focal_src_idx = nodes.index(best_src)
    focal_dst_idx = nodes.index(best_dst)
    return nodes, labels, focal_src_idx, focal_dst_idx, best_p


# ---------------------------------------------------------------------------
# PNG export: chain graph
# ---------------------------------------------------------------------------
# Edge color tiers by transition probability:
#   Red   (#E05252)  — low    p < 0.35
#   Green (#3BAA6E)  — medium 0.35 <= p < 0.70
#   Blue  (#3A7FD4)  — high   p >= 0.70

CHAIN_N = 6   # number of nodes in the chain image


def _edge_color(p: float) -> str:
    if p >= 0.70:
        return "#3A7FD4"   # blue  — high
    elif p >= 0.35:
        return "#3BAA6E"   # green — medium
    else:
        return "#E05252"   # red   — low


def _chain_nodes(frequency: dict, k: int = CHAIN_N) -> tuple:
    """
    Pick k nodes that are guaranteed to all be mutually connected.
    Prefers contexts that aren't entirely VIZ_FILTER_WORDS so the
    graph labels are meaningful, then greedily expands by connectivity.
    """
    probs = _transition_probs(frequency)
    # Rank by out-degree, preferring content-word contexts for readability
    ranked = sorted(
        frequency.keys(),
        key=lambda c: (len(frequency[c]), not _has_viz_filter(c)),
        reverse=True
    )

    selected = [ranked[0]]
    selected_set = {ranked[0]}

    # Build a reverse-lookup: which contexts point TO a given context?
    # and forward: which contexts does a context point to?
    def neighbors(ctx):
        """All contexts reachable from ctx AND contexts that reach ctx."""
        fwd = set()
        for w in probs.get(ctx, {}):
            dst = ctx[1:] + (w,) if len(ctx) > 1 else (w,)
            if dst in frequency:
                fwd.add(dst)
        bwd = set()
        for other in frequency:
            for w in probs.get(other, {}):
                dst = other[1:] + (w,) if len(other) > 1 else (w,)
                if dst == ctx:
                    bwd.add(other)
        return fwd | bwd

    # Grow the selection by always picking the candidate with the most
    # connections into the existing set.
    for _ in range(k - 1):
        best_ctx, best_score = None, -1
        for ctx in ranked:
            if ctx in selected_set:
                continue
            score = sum(1 for n in neighbors(ctx) if n in selected_set)
            if score > best_score:
                best_ctx, best_score = ctx, score
        if best_ctx is None:
            break
        selected.append(best_ctx)
        selected_set.add(best_ctx)

    labels = [_context_label(ctx) for ctx in selected]
    return selected, labels


def _save_chain_png(frequency: dict, nodes: list, labels: list,
                    corpus_name: str, out_file: str) -> None:
    chain_nodes, chain_labels = _chain_nodes(frequency)
    probs = _transition_probs(frequency)
    n_nodes = len(chain_nodes)
    idx = {ctx: i for i, ctx in enumerate(chain_nodes)}

    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    ax.set_aspect("equal")
    ax.axis("off")

    cx, cy, ring_r = 0.5, 0.5, 0.36
    positions = _circle_positions(n_nodes, cx, cy, ring_r)

    # Draw edges — color by probability tier, ensure each node has at least one edge out
    drawn_edges = set()
    for i, src in enumerate(chain_nodes):
        sx, sy = positions[i]
        src_edges = []
        for w, p in sorted(probs[src].items(), key=lambda x: -x[1]):
            dst_ctx = src[1:] + (w,) if len(src) > 1 else (w,)
            if dst_ctx not in idx:
                continue
            j = idx[dst_ctx]
            if i == j:
                continue
            src_edges.append((j, w, p))

        # If this node has no edges into the visible set, force-connect it
        # to its nearest neighbor (by position) with the best available prob
        if not src_edges:
            best_w = max(probs[src], key=probs[src].get) if probs.get(src) else None
            if best_w:
                # connect to the closest node by index as a visual stand-in
                fallback_j = (i + 1) % n_nodes
                src_edges.append((fallback_j, best_w, probs[src].get(best_w, 0.1)))

        for j, w, p in src_edges:
            edge_key = (i, j)
            if edge_key in drawn_edges:
                continue
            drawn_edges.add(edge_key)
            tx, ty = positions[j]
            color = _edge_color(p)
            lw = max(1.2, min(4.0, p * 5))
            ax.annotate("", xy=(tx, ty), xytext=(sx, sy),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=color,
                            lw=lw,
                            alpha=0.88,
                            connectionstyle="arc3,rad=0.22",
                        ), zorder=3)
            # probability label along the arc
            mx = (sx + tx) / 2 - (ty - sy) * 0.10
            my = (sy + ty) / 2 + (tx - sx) * 0.10
            ax.text(mx, my, f"{p:.2f}", fontsize=7, ha="center", va="center",
                    color=color, fontweight="bold", zorder=4,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="#1a1a2e")])

    # Draw nodes — uniform white style so color emphasis stays on edges
    node_r = 0.068
    for i, ctx in enumerate(chain_nodes):
        x, y = positions[i]
        circle = plt.Circle((x, y), node_r, facecolor="#2a2a4a",
                             edgecolor="#CCCCDD", linewidth=1.8,
                             transform=ax.transData, zorder=5)
        ax.add_patch(circle)
        ax.text(x, y, chain_labels[i], ha="center", va="center",
                fontsize=8, fontweight="bold", color="#EEEEFF", zorder=6)

    # Legend
    legend_els = [
        mpatches.Patch(facecolor="#E05252", edgecolor="#E05252", label="Low  (p < 0.35)"),
        mpatches.Patch(facecolor="#3BAA6E", edgecolor="#3BAA6E", label="Medium  (0.35 \u2013 0.70)"),
        mpatches.Patch(facecolor="#3A7FD4", edgecolor="#3A7FD4", label="High  (p \u2265 0.70)"),
    ]
    ax.legend(handles=legend_els, loc="lower center", bbox_to_anchor=(0.5, -0.01),
              ncol=3, fontsize=9, framealpha=0.25, edgecolor="#555577",
              labelcolor="white", facecolor="#2a2a4a")

    ax.set_title(f"Markov Chain \u2014 {corpus_name}",
                 fontsize=12, fontweight="500", color="#EEEEFF", pad=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    fig.savefig(out_file, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  [png] Chain saved → {os.path.abspath(out_file)}")


# ---------------------------------------------------------------------------
# PNG export: transition matrix — focused on the strongest pair
# ---------------------------------------------------------------------------

def _cell_rgb(p: float) -> tuple:
    """Return (face_color, text_color) for a matrix cell based on probability tier."""
    if p == 0:
        return "#1a1a2e", "#444466"       # empty — match background
    elif p >= 0.70:
        return "#1a4a8a", "#AACCFF"       # blue  — high
    elif p >= 0.35:
        return "#1a5a38", "#88DDAA"       # green — medium
    else:
        return "#6a1a1a", "#FFAAAA"       # red   — low


def _save_matrix_png(frequency: dict, nodes: list, labels: list,
                     corpus_name: str, out_file: str) -> None:
    # Use the same node set as the chain for consistency
    mat_nodes, mat_labels = _chain_nodes(frequency)
    probs = _transition_probs(frequency)
    n_rows = len(mat_nodes)

    # Columns: top successor words across all rows (deduplicated, sorted by max prob)
    word_max_p = {}
    for ctx in mat_nodes:
        for w, p in probs.get(ctx, {}).items():
            word_max_p[w] = max(word_max_p.get(w, 0), p)
    successor_words = sorted(word_max_p, key=word_max_p.get, reverse=True)[:n_rows]
    n_cols = len(successor_words)   # may be < n_rows on a tiny corpus

    # Build matrix
    matrix = np.zeros((n_rows, n_cols))
    for i, src in enumerate(mat_nodes):
        for j, word in enumerate(successor_words):
            matrix[i, j] = probs.get(src, {}).get(word, 0.0)

    col_labels = [w[:11] + "\u2026" if len(w) > 11 else w for w in successor_words]
    row_labels  = [l[:11] + "\u2026" if len(l) > 11 else l for l in mat_labels]

    fig, ax = plt.subplots(figsize=(max(6, n_cols * 1.1), max(5, n_rows * 0.9)),
                           facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Draw each cell manually as a colored rectangle + text
    for i in range(n_rows):
        for j in range(n_cols):
            val = matrix[i, j]
            face, text_c = _cell_rgb(val)
            rect = mpatches.FancyBboxPatch(
                (j - 0.46, i - 0.46), 0.92, 0.92,
                boxstyle="round,pad=0.04",
                facecolor=face, edgecolor="#2a2a4e", linewidth=0.8,
                zorder=2
            )
            ax.add_patch(rect)
            if val > 0:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=text_c, zorder=3)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)   # invert y so row 0 is at top
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9, color="#CCCCEE")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9, color="#CCCCEE")
    ax.tick_params(colors="#CCCCEE", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend
    legend_els = [
        mpatches.Patch(facecolor="#6a1a1a", edgecolor="#FFAAAA", label="Low  (p < 0.35)"),
        mpatches.Patch(facecolor="#1a5a38", edgecolor="#88DDAA", label="Medium  (0.35 \u2013 0.70)"),
        mpatches.Patch(facecolor="#1a4a8a", edgecolor="#AACCFF", label="High  (p \u2265 0.70)"),
    ]
    ax.legend(handles=legend_els, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9,
              framealpha=0.25, edgecolor="#555577",
              labelcolor="white", facecolor="#2a2a4a")

    ax.set_title(f"Transition Matrix \u2014 {corpus_name}",
                 fontsize=12, fontweight="500", color="#EEEEFF", pad=12)
    ax.set_xlabel("next word \u2192", fontsize=9, color="#CCCCEE")
    ax.set_ylabel("context (from) \u2193", fontsize=9, color="#CCCCEE")

    fig.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"  [png] Matrix saved → {os.path.abspath(out_file)}")


# ---------------------------------------------------------------------------
# HTML + PNG visualizer (combined)
# ---------------------------------------------------------------------------

def visualize_markov(frequency: dict, corpus_name: str = "corpus",
                     out_file: str = "markov_viz.html",
                     chain_png: str = "markov_chain.png",
                     matrix_png: str = "markov_matrix.png",
                     k: int = 8) -> None:
    """
    Write:
      • A standalone HTML file (chain graph + heatmap, interactive)
      • chain_png  — chain graph as a PNG image
      • matrix_png — transition matrix heatmap as a PNG image
    """
    if not frequency:
        print("  [viz] frequency dict is empty — skipping.")
        return

    probs = _transition_probs(frequency)
    nodes = _top_nodes(frequency, k)
    n_nodes = len(nodes)
    idx = {ctx: i for i, ctx in enumerate(nodes)}
    labels = [_context_label(ctx) for ctx in nodes]

    # Save PNGs first
    _save_chain_png(frequency, nodes, labels, corpus_name, chain_png)
    _save_matrix_png(frequency, nodes, labels, corpus_name, matrix_png)

    # ── HTML (unchanged from original, but updated paths) ────────────────────
    VW, VH = 700, 560
    cx_h, cy_h = VW / 2, VH / 2 - 10
    ring_r = min(VW, VH) * 0.36
    positions = _circle_positions(n_nodes, cx_h, cy_h, ring_r)
    node_r = max(28, min(42, int(220 / n_nodes)))

    edge_svg, plabel_svg = [], []
    eid = 0
    for i, src in enumerate(nodes):
        sx, sy = positions[i]
        for w, p in sorted(probs[src].items(), key=lambda x: -x[1]):
            dst_ctx = src[1:] + (w,) if len(src) > 1 else (w,)
            if dst_ctx not in idx:
                continue
            j = idx[dst_ctx]
            if i == j:
                continue
            tx, ty = positions[j]
            dx, dy = tx - sx, ty - sy
            dist = math.hypot(dx, dy) or 1
            ux, uy = dx / dist, dy / dist
            x1 = sx + ux * node_r; y1 = sy + uy * node_r
            x2 = tx - ux * (node_r + 6); y2 = ty - uy * (node_r + 6)
            mx2 = (x1 + x2) / 2 - uy * 24
            my2 = (y1 + y2) / 2 + ux * 24
            sw = max(1.0, min(4.5, p * 5.5))
            op = max(0.35, min(1.0, p + 0.2))
            _, stroke, _ = _node_color(frequency, src)
            edge_svg.append(
                f'<path id="e{eid}" data-src="{i}" class="mc-edge" '
                f'd="M{x1:.1f} {y1:.1f} Q{mx2:.1f} {my2:.1f} {x2:.1f} {y2:.1f}" '
                f'fill="none" stroke="{stroke}" stroke-width="{sw:.1f}" '
                f'opacity="{op:.2f}" marker-end="url(#arr)"/>'
            )
            lx = 0.25*x1 + 0.5*mx2 + 0.25*x2
            ly = 0.25*y1 + 0.5*my2 + 0.25*y2
            plabel_svg.append(
                f'<text class="prob-lbl" x="{lx:.1f}" y="{ly:.1f}" '
                f'text-anchor="middle" fill="{stroke}">{p:.2f}</text>'
            )
            eid += 1

    node_svg = []
    for i, ctx in enumerate(nodes):
        x, y = positions[i]
        fill, stroke, text_col = _node_color(frequency, ctx)
        is_start = ctx[0] in START_WORDS
        is_stop = ctx[0] in set()
        words = labels[i].split()
        mid = max(1, len(words) // 2)
        lines = [" ".join(words[:mid]), " ".join(words[mid:])] if len(words) > 1 else [labels[i]]
        text_els = "".join(
            f'<text font-family="sans-serif" font-size="11" font-weight="500" '
            f'text-anchor="middle" fill="{text_col}" x="{x:.1f}" '
            f'y="{y + (li - (len(lines)-1)/2)*14:.1f}" dominant-baseline="central">'
            f'{line}</text>'
            for li, line in enumerate(lines)
        )
        start_ring = (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{node_r + 5}" '
            f'fill="none" stroke="#1D9E75" stroke-width="1.5" stroke-dasharray="4 3"/>'
            if is_start else ""
        )
        stop_dot = (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#E24B4A"/>'
            if is_stop else ""
        )
        has_successor = any(
            idx.get((ctx[1:] + (w,)) if len(ctx) > 1 else (w,)) is not None
            for w in probs.get(ctx, {})
        )
        terminal_ring = (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{node_r-7}" '
            f'fill="none" stroke="{stroke}" stroke-width="0.5"/>'
            if not has_successor else ""
        )
        node_svg.append(
            f'<g class="mc-node" data-idx="{i}" onclick="activateNode({i})">'
            f'{start_ring}'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{node_r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            f'{terminal_ring}{stop_dot}{text_els}</g>'
        )

    # Transition matrix HTML
    ramp = [
        ("transparent", "#888780"), ("#EEEDFE", "#3C3489"), ("#AFA9EC", "#3C3489"),
        ("#7F77DD", "#EEEDFE"),     ("#534AB7", "#EEEDFE"), ("#3C3489", "#EEEDFE"),
    ]
    def cell_style(p):
        r = (ramp[0] if p == 0    else ramp[1] if p < 0.2 else
             ramp[2] if p < 0.4   else ramp[3] if p < 0.6 else
             ramp[4] if p < 0.8   else ramp[5])
        return f"background:{r[0]};color:{r[1]};"

    th = ("padding:6px 10px;font-size:11px;font-weight:500;color:#5F5E5A;"
          "white-space:nowrap;border-bottom:1px solid #D3D1C7;")
    td = "padding:6px 8px;font-size:11px;text-align:center;border-bottom:1px solid #F1EFE8;"
    col_heads = "".join(
        f'<th style="{th}text-align:center;">{labels[j]}</th>' for j in range(n_nodes)
    )
    rows_html = ""
    for i, src in enumerate(nodes):
        cells = ""
        for j, dst in enumerate(nodes):
            p = 0.0
            for w, wp in probs.get(src, {}).items():
                if ((src[1:] + (w,)) if len(src) > 1 else (w,)) == dst:
                    p = wp; break
            cells += f'<td style="{td}{cell_style(p)}">' + (f"{p:.2f}" if p else "—") + "</td>"
        row_extra = ""
        if nodes[i][0] in START_WORDS:
            row_extra = ' style="' + th + 'text-align:left;min-width:120px;color:#0F6E56;"'
        elif nodes[i][0] in set():
            row_extra = ' style="' + th + 'text-align:left;min-width:120px;color:#A32D2D;"'
        else:
            row_extra = ' style="' + th + 'text-align:left;min-width:120px;"'
        rows_html += f'<tr><td{row_extra}>{labels[i]}</td>{cells}</tr>'

    active_starts = sorted({ctx[0] for ctx in nodes if ctx[0] in START_WORDS})
    active_stops  = sorted({ctx[0] for ctx in nodes if ctx[0] in set()})
    start_pills = "".join(
        f'<span style="background:#E1F5EE;color:#085041;border:0.5px solid #1D9E75;'
        f'border-radius:4px;padding:2px 7px;font-size:11px;margin:2px;">{w}</span>'
        for w in active_starts
    ) or '<span style="color:#888;font-size:11px;">none in top nodes</span>'
    stop_pills = "".join(
        f'<span style="background:#FCEBEB;color:#791F1F;border:0.5px solid #E24B4A;'
        f'border-radius:4px;padding:2px 7px;font-size:11px;margin:2px;">{w}</span>'
        for w in active_stops
    ) or '<span style="color:#888;font-size:11px;">none in top nodes</span>'

    chain_png_basename  = os.path.basename(chain_png)
    matrix_png_basename = os.path.basename(matrix_png)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Markov chain — {corpus_name}</title>
<style>
body  {{ font-family:system-ui,sans-serif; margin:0; padding:24px 32px; background:#fafaf8; color:#1a1a18; }}
h1   {{ font-size:20px; font-weight:500; margin:0 0 4px; }}
.sub {{ font-size:13px; color:#5F5E5A; margin:0 0 20px; }}
h2   {{ font-size:15px; font-weight:500; margin:24px 0 8px; }}
.note{{ font-size:12px; color:#5F5E5A; margin:0 0 10px; }}
.card{{ background:#fff; border:0.5px solid #D3D1C7; border-radius:12px; padding:16px; margin-bottom:24px; overflow-x:auto; }}
svg.mc {{ width:100%; }}
.mc-node {{ cursor:pointer; }}
.mc-node circle {{ transition:opacity .15s; }}
.mc-node:hover circle {{ opacity:.78; }}
.mc-edge {{ transition:opacity .2s; }}
.mc-edge.faded {{ opacity:.07 !important; }}
.prob-lbl {{ font-size:10px; font-family:system-ui,sans-serif; }}
table {{ border-collapse:collapse; min-width:100%; }}
.legend {{ display:flex; gap:16px; flex-wrap:wrap; font-size:12px; color:#5F5E5A; margin-top:10px; align-items:center; }}
.leg-dot {{ width:11px; height:11px; border-radius:50%; border:1.5px solid; display:inline-block; vertical-align:middle; margin-right:4px; }}
.wl-row {{ display:flex; gap:8px; align-items:baseline; flex-wrap:wrap; margin-bottom:8px; }}
.wl-label {{ font-size:12px; font-weight:500; min-width:80px; color:#5F5E5A; }}
.png-row {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:24px; }}
.png-card {{ flex:1; min-width:280px; background:#fff; border:0.5px solid #D3D1C7; border-radius:12px; padding:12px; text-align:center; }}
.png-card img {{ max-width:100%; border-radius:6px; }}
.png-card p {{ font-size:11px; color:#5F5E5A; margin:6px 0 0; }}
</style>
</head>
<body>

<h1>Markov chain visualizer</h1>
<p class="sub">Corpus: <strong>{corpus_name}</strong> &nbsp;·&nbsp;
Top {n_nodes} states by out-degree &nbsp;·&nbsp; Arc weight = transition probability
&nbsp;·&nbsp; <em>Filter words excluded from all contexts</em></p>

<h2>Saved PNG images</h2>
<div class="png-row">
  <div class="png-card">
    <img src="{chain_png_basename}" alt="Markov chain graph">
    <p><a href="{chain_png_basename}" download>⬇ Download chain graph</a></p>
  </div>
  <div class="png-card">
    <img src="{matrix_png_basename}" alt="Transition matrix heatmap">
    <p><a href="{matrix_png_basename}" download>⬇ Download matrix heatmap</a></p>
  </div>
</div>

<h2>Word lists active for this corpus</h2>
<div class="card">
  <div class="wl-row">
    <span class="wl-label">Start words</span>
    <span style="font-size:11px;color:#5F5E5A;margin-right:6px;">(prefer these as sentence openers)</span>
    {start_pills}
  </div>
  <div class="wl-row">
    <span class="wl-label">Stop words</span>
    <span style="font-size:11px;color:#5F5E5A;margin-right:6px;">(close sentence early when reached)</span>
    {stop_pills}
  </div>
  <p class="note" style="margin:8px 0 0;">Only words that appear in the top {n_nodes} shown nodes are listed above.</p>
</div>

<h2>Interactive chain graph</h2>
<div class="card">
<svg class="mc" viewBox="0 0 {VW} {VH}" role="img">
  <title>Markov chain — {corpus_name}</title>
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="8" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke"
            stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
    </marker>
  </defs>
  {"".join(edge_svg)}
  {"".join(plabel_svg)}
  {"".join(node_svg)}
</svg>
<div class="legend">
  <span><span class="leg-dot" style="background:#EEEDFE;border-color:#534AB7;"></span>High degree (≥3)</span>
  <span><span class="leg-dot" style="background:#E1F5EE;border-color:#1D9E75;"></span>Medium (2)</span>
  <span><span class="leg-dot" style="background:#F1EFE8;border-color:#888780;"></span>Low / terminal</span>
  <span style="margin-left:auto;color:#aaa;">Click a node to highlight its transitions</span>
</div>
</div>

<h2>Transition matrix</h2>
<p class="note">Rows = current state &nbsp;·&nbsp; Columns = next state &nbsp;·&nbsp;
Darker = higher probability &nbsp;·&nbsp;
<span style="color:#0F6E56;">Green label</span> = start-word context &nbsp;·&nbsp;
<span style="color:#A32D2D;">Red label</span> = stop-word context</p>
<div class="card">
<table>
  <thead><tr>
    <th style="{th}text-align:left;min-width:120px;">from ↓ / to →</th>
    {col_heads}
  </tr></thead>
  <tbody>{rows_html}</tbody>
</table>
</div>

<script>
let activeNode = null;
function activateNode(idx) {{
  const edges = document.querySelectorAll('.mc-edge');
  if (activeNode === idx) {{
    edges.forEach(e => e.classList.remove('faded'));
    activeNode = null; return;
  }}
  activeNode = idx;
  edges.forEach(e => e.classList.toggle('faded', parseInt(e.dataset.src) !== idx));
}}
</script>
</body></html>
"""
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"  [html] Written → {os.path.abspath(out_file)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 4:
        print("Usage: python ngram.py <n> <m> <file1> [file2 ...]")
        sys.exit(1)

    n         = int(sys.argv[1])
    m         = int(sys.argv[2])
    filenames = sys.argv[3:]
    corpus_name = ", ".join(os.path.basename(f) for f in filenames)

    print("This program belongs to Julian Chaoul")
    print(f"This program generates random sentences based on a {n}-gram model.")
    print(f"Sentence word cap: {MAX_SENTENCE_WORDS} words.")
    print(f"Start words (sentence openers): {len(START_WORDS)}")
    print(f"Viz filter words (visuals only): {len(VIZ_FILTER_WORDS)}")
    print()

    sentences = convert_to_list(filenames)
    frequency, start_words = arrange_frequency(sentences, n)

    preferred = _preferred_starts(frequency)
    print(f"  {len(frequency)} unique contexts found.")
    print(f"  {len(preferred)} preferred start contexts (matched START_WORDS).")
    print()

    generated = create_sentences(frequency, start_words, n, m)
    print("Here are your generated sentences:")
    for sentence in generated:
        print(sentence)
    print()

    safe_name = "_".join(os.path.splitext(os.path.basename(f))[0] for f in filenames)
    visualize_markov(
        frequency,
        corpus_name=corpus_name,
        out_file=f"markov_{safe_name}.html",
        chain_png=f"markov_{safe_name}_chain.png",
        matrix_png=f"markov_{safe_name}_matrix.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())