#!/usr/bin/env python3
"""Semantic code/doc index via the local embedding server (port 8081).

Usage:
    python3 embed_index.py build   # Rebuild index (SQLite: .embed_index.db)
    python3 embed_index.py query "meaning text" [-k 5]
    python3 embed_index.py stats   # Index statistics

Chunking limit: 511 tokens = model context length (bert.context_length).
1 token ≈ 3.5 chars (t5/SentencePiece) -> MAX_CHARS = 800 as safety limit.
"""
import json
import math
import os
import sqlite3
import struct
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))  # repo root
DB_PATH = os.path.join(ROOT, '.embed_index.db')
EMBED_URL = 'http://127.0.0.1:8081/embedding'

# t5/SentencePiece tokenisiert Code dichter als Prosa:
# gemessen 1600 Zeichen (ai.sh) = 624 Token -> 0.39 Token/Zeichen,
# aber check_evidence.py: 1000 Zeichen = 604 Token -> 0.60 Token/Zeichen.
# 800 Zeichen * 0.60 = 480 Token -> sicher unter der 511-Token-Grenze.
MAX_CHARS = 800
OVERLAP_LINES = 2     # Zeilen-Überlappung bei Code-Chunking

# Ordner, die nie indexiert werden
SKIP_DIRS = {'.git', '.venv', 'venv', 'node_modules', '__pycache__',
             '.pytest_cache', '.obsidian', 'build'}
SKIP_ROOTS = set()

EXTS = {'.py', '.md', '.sh', '.toml', '.txt', '.jinja', '.yml', '.yaml'}


def iter_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS
                       and os.path.join(dirpath, d) not in SKIP_ROOTS]
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in EXTS:
                yield os.path.join(dirpath, fn)


def chunk_markdown(path, text):
    """Splitte Markdown nach Überschriften; lange Sektionen nach Absätzen."""
    chunks, cur, heading = [], [], None
    for line in text.splitlines():
        if line.startswith('#'):
            if cur:
                chunks.append('\n'.join(cur))
            heading = line
            cur = [line]
        else:
            cur.append(line)
    if cur:
        chunks.append('\n'.join(cur))
    # lange Chunks nach Absätzen aufteilen
    result = []
    for c in chunks:
        if len(c) <= MAX_CHARS:
            result.append(c)
            continue
        para, buf = [], []
        for p in c.split('\n\n'):
            if buf and len('\n\n'.join(buf)) + len(p) > MAX_CHARS:
                result.append('\n\n'.join(para))
                para, buf = [], []
            buf.append(p)
        if buf:
            result.append('\n\n'.join(para))
    return [c for c in result if c.strip()]


def chunk_lines(path, text):
    """Code/Config: Zeilen-Blöcke mit Überlappung."""
    lines = text.splitlines()
    chunks, buf, start = [], [], 0
    for i, line in enumerate(lines):
        buf.append(line)
        if len('\n'.join(buf)) > MAX_CHARS:
            chunks.append('\n'.join(buf))
            # Überlappung: letzte N Zeilen als Start des nächsten Blocks
            keep = lines[i - OVERLAP_LINES:i + 1]
            buf = keep
            start = i - OVERLAP_LINES
    if buf:
        chunks.append('\n'.join(buf))
    return [c for c in chunks if c.strip()]


def make_chunks(path, text):
    if path.endswith('.md'):
        chunks = chunk_markdown(path, text)
    else:
        chunks = chunk_lines(path, text)
    out = []
    for c in chunks:
        # Harte Grenze: nie mehr als MAX_CHARS (Modell-Kontext 511 Token)
        while c:
            piece, c = c[:MAX_CHARS], c[MAX_CHARS:]
            line_no = text.count('\n', 0, text.find(piece)) + 1 if piece else 1
            out.append((path, len(out), line_no, piece))
    return out


def _embed_request(texts):
    req = urllib.request.Request(
        EMBED_URL,
        data=json.dumps({'input': texts}).encode(),
        headers={'Content-Type': 'application/json'})
    resp = json.loads(urllib.request.urlopen(req, timeout=300).read())
    items = resp if isinstance(resp, list) else [resp]
    vecs = []
    for item in items:
        e = item['embedding']
        if isinstance(e, list) and e and isinstance(e[0], list):
            e = e[0]
        vecs.append(e)
    return vecs


def embed_batch(texts):
    """Batch-Embedding; liefert Liste von 384-d Float-Listen.

    Bei HTTP 500 (z. B. Chunk > 511 Token) wird der Batch halbiert und
    erneut versucht.
    """
    try:
        return _embed_request(texts)
    except urllib.error.HTTPError as e:
        if e.code != 500 or len(texts) == 1:
            raise
        mid = len(texts) // 2
        return embed_batch(texts[:mid]) + embed_batch(texts[mid:])


def init_db():
    db = sqlite3.connect(DB_PATH)
    db.execute('''CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY,
        path TEXT NOT NULL,
        chunk_idx INTEGER NOT NULL,
        line_no INTEGER NOT NULL,
        text TEXT NOT NULL,
        emb BLOB NOT NULL,
        mtime REAL)''')
    db.commit()
    # Migration für alte DBs (vor der mtime-Spalte)
    cols = [r[1] for r in db.execute('PRAGMA table_info(chunks)')]
    if 'mtime' not in cols:
        db.execute('ALTER TABLE chunks ADD COLUMN mtime REAL')
        db.commit()
    return db


def _embed_file(db, path, mtime):
    """(Re-)Embeddet EINE Datei: alte Chunks löschen, neue einfügen."""
    db.execute('DELETE FROM chunks WHERE path=?', (path,))
    with open(os.path.join(ROOT, path), 'r', encoding='utf-8',
              errors='replace') as f:
        text = f.read()
    for ap, ci, ln, txt in make_chunks(os.path.join(ROOT, path), text):
        v = embed_batch([txt])[0]
        db.execute(
            'INSERT INTO chunks (path, chunk_idx, line_no, text, emb, mtime) '
            'VALUES (?,?,?,?,?,?)',
            (path, ci, ln, txt, struct.pack(f'{len(v)}f', *v), mtime))


def cmd_reindex():
    """Inkrementeller Re-Index: nur geänderte/neue Dateien (mtime),
    gelöschte Dateien werden entfernt. Viel schneller als build."""
    db = init_db()
    files = {}
    for p in iter_files():
        try:
            files[os.path.relpath(p, ROOT)] = os.path.getmtime(p)
        except OSError:
            pass
    # 1. Gelöschte Dateien: Chunks entfernen
    for (path,) in db.execute('SELECT DISTINCT path FROM chunks').fetchall():
        if path not in files:
            db.execute('DELETE FROM chunks WHERE path=?', (path,))
    # 2. Geänderte/neue Dateien: (re-)embedden
    changed = 0
    for path, mtime in sorted(files.items()):
        row = db.execute(
            'SELECT MAX(mtime) FROM chunks WHERE path=?', (path,)).fetchone()
        if row is None or row[0] is None or abs(row[0] - mtime) > 1e-6:
            _embed_file(db, path, mtime)
            changed += 1
            print(f'  {path} ...', end='\r')
    db.commit()
    print(f'\n✓ Reindex: {changed} Datei(en) aktualisiert -> {DB_PATH}')


def cmd_build():
    db = init_db()
    db.execute('DELETE FROM chunks')
    files = list(iter_files())
    print(f'{len(files)} Dateien gefunden')
    n_chunks = 0
    # 1 Chunk pro Request: Chunk (max ~450 Token) passt in das 512-Token-Batch
    BATCH = 1
    batch_paths, batch_texts = [], []
    rel = lambda p: os.path.relpath(p, ROOT)

    def flush():
        nonlocal n_chunks
        if not batch_texts:
            return
        vecs = embed_batch(batch_texts)
        for (p, ci, ln, txt), v in zip(batch_paths, vecs):
            assert len(v) == 384, f'unerwartete Dimension {len(v)}'
            try:
                mt = os.path.getmtime(p)
            except OSError:
                mt = 0.0
            db.execute(
                'INSERT INTO chunks (path, chunk_idx, line_no, text, emb, mtime) '
                'VALUES (?,?,?,?,?,?)',
                (rel(p), ci, ln, txt, struct.pack(f'{len(v)}f', *v), mt))
            n_chunks += 1
        db.commit()
        batch_paths.clear()
        batch_texts.clear()

    for path in files:
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
        except OSError:
            continue
        for p, ci, ln, txt in make_chunks(path, text):
            batch_paths.append((p, ci, ln, txt))
            batch_texts.append(txt)
            if len(batch_texts) >= BATCH:
                flush()
                print(f'  {n_chunks} Chunks ...', end='\r')
    flush()
    print(f'\n✓ Index: {n_chunks} Chunks aus {len(files)} Dateien -> {DB_PATH}')


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def cmd_query(query, k=5):
    qvec = embed_batch([query])[0]
    db = sqlite3.connect(DB_PATH)
    rows = db.execute('SELECT path, chunk_idx, line_no, text, emb FROM chunks').fetchall()
    scored = []
    for path, ci, ln, text, blob in rows:
        v = struct.unpack(f'{len(blob)//4}f', blob)
        scored.append((cosine(qvec, v), path, ln, text))
    scored.sort(reverse=True)
    for score, path, ln, text in scored[:k]:
        preview = ' '.join(text.split())[:120]
        print(f'{score:.4f}  {path}:{ln}  [{preview}]')


def cmd_stats():
    db = sqlite3.connect(DB_PATH)
    n = db.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
    f = db.execute('SELECT COUNT(DISTINCT path) FROM chunks').fetchone()[0]
    print(f'{n} Chunks aus {f} Dateien ({DB_PATH})')


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == 'build':
        cmd_build()
    elif cmd == 'reindex':
        cmd_reindex()
    elif cmd == 'query':
        if len(sys.argv) < 3:
            print('Query-Text fehlt')
            sys.exit(1)
        k = 5
        if '-k' in sys.argv:
            k = int(sys.argv[sys.argv.index('-k') + 1])
            qargs = sys.argv[2:sys.argv.index('-k')]
        else:
            qargs = sys.argv[2:]
        cmd_query(' '.join(qargs), k)
    elif cmd == 'stats':
        cmd_stats()
    else:
        print(f'Unbekannter Befehl: {cmd}')
        sys.exit(1)


if __name__ == '__main__':
    main()
