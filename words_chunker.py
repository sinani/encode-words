#!/usr/bin/env python3
"""
words_chunker.py — Encode/Decode files as OCR-friendly word blocks with compression + ECC.

Features
- Compresses whole file (zlib level 9), then splits into chunks measured in WORDS.
- Optional Reed–Solomon per-chunk parity using `reedsolo` (handles long messages by blockifying).
- Each chunk = SENTINEL + total_len + header_len + header_json + rs_payload  (all bytes)
  then mapped to base-N digits (N = len(wordlist)) → words.
- Header has: part index/total, filename, original/compressed sizes, raw-chunk len, ecc bytes, CRC32.
- Decoder reassembles, RS-decodes, CRC-checks, concatenates compressed chunks, then zlib-decompresses.

Wordlist
- Provide a UTF-8 file with one word per line. 65,536 words gives 16 bits/word (2 bytes/word).
- Avoid duplicates. You can use Niceware or any 65k curated list.

Build as single EXE (Windows):
  pip install reedsolo pyinstaller
  pyinstaller --onefile words_chunker.py
"""

import argparse, os, json, zlib, math, hashlib
from pathlib import Path
from typing import List, Tuple

# ---------- Helpers ----------
MAGIC = b"W1"  # 2-byte magic
SENTINEL = b"\x01"  # prevents leading-zero loss in big-int roundtrip

def load_wordlist(path: str) -> List[str]:
    words = [w.strip() for w in open(path, "r", encoding="utf-8") if w.strip()]
    if len(words) != len(set(words)):
        raise SystemExit("Wordlist contains duplicate words; please de-duplicate.")
    return words

def bytes_to_int(b: bytes) -> int:
    return int.from_bytes(b, "big", signed=False)

def int_to_bytes(v: int, length: int) -> bytes:
    return v.to_bytes(length, "big")

def big_int_to_word_indices(value: int, base: int) -> List[int]:
    if value == 0:
        return [0]
    out = []
    while value:
        value, r = divmod(value, base)
        out.append(r)
    return list(reversed(out))

def word_indices_to_big_int(indices: List[int], base: int) -> int:
    v = 0
    for d in indices:
        v = v * base + d
    return v

def bytes_to_words(b: bytes, wordlist: List[str]) -> List[str]:
    base = len(wordlist)
    big = bytes_to_int(b)
    idx = big_int_to_word_indices(big, base)
    return [wordlist[i] for i in idx]

def words_to_bytes(words: List[str], wordlist: List[str]) -> bytes:
    index = {w: i for i, w in enumerate(wordlist)}
    try:
        idx = [index[w] for w in words]
    except KeyError as e:
        raise SystemExit(f"Unknown word in input: {e}")
    big = word_indices_to_big_int(idx, len(wordlist))
    nbytes = max(1, (big.bit_length() + 7) // 8)
    return int_to_bytes(big, nbytes)

def chunk_bytes(data: bytes, n: int):
    for i in range(0, len(data), n):
        yield data[i:i+n]

def crc32_hex(b: bytes) -> str:
    return f"{zlib.crc32(b) & 0xFFFFFFFF:08x}"

# ---------- ECC (Reed–Solomon) ----------
def rs_encode(payload: bytes, ecc_nsym: int) -> bytes:
    if ecc_nsym <= 0:
        return payload
    try:
        import reedsolo
    except Exception as e:
        raise SystemExit("reedsolo not installed. Run: pip install reedsolo") from e
    rs = reedsolo.RSCodec(ecc_nsym)
    return rs.encode(payload)

# ---------- ECC (Reed–Solomon) ----------
def rs_decode(rs_payload: bytes, ecc_nsym: int) -> bytes:
    """
    Decode a Reed–Solomon protected chunk. Some reedsolo versions return a tuple
    (decoded, ecc); normalize to pure bytes.
    """
    if ecc_nsym <= 0:
        return rs_payload
    import reedsolo  # validated at encode
    rs = reedsolo.RSCodec(ecc_nsym)
    out = rs.decode(rs_payload)
    if isinstance(out, tuple):
        out = out[0]
    return bytes(out)

# ---------- Encode ----------
def encode_file_to_words(
    infile: str,
    wordlist_path: str,
    out_dir: str,
    chunk_words: int = 1000,
    ecc_pct: int = 10,
):
    words = load_wordlist(wordlist_path)
    base = len(words)
    if base < 1024:
        print(f"Warning: small wordlist ({base}). Larger lists reduce total words.")

    raw = Path(infile).read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    comp = zlib.compress(raw, 9)

    # Payload per chunk in bytes (for 65,536 words, 2 bytes/word)
    # Generally: floor(chunk_words * log2(base) / 8) — but for base=65536 that is exactly 2*chunk_words
    bytes_per_word = math.log2(base) / 8.0
    payload_bytes = int(chunk_words * bytes_per_word)

    if payload_bytes < 8:
        raise SystemExit("chunk too small; increase --chunk-words")

    # Number of parts
    total = math.ceil(len(comp) / payload_bytes)

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    manifest = {
        "file": os.path.basename(infile),
        "orig_bytes": len(raw),
        "compressed_bytes": len(comp),
        "sha256": sha256,
        "wordlist_size": base,
        "bytes_per_word": bytes_per_word,
        "chunk_words": chunk_words,
        "payload_bytes_per_chunk": payload_bytes,
        "ecc_pct": ecc_pct,
        "parts": total,
    }
    (outp / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Iterate chunks
    for i, chunk in enumerate(chunk_bytes(comp, payload_bytes), start=1):
        # Decide ECC symbols per chunk from percentage (nsym is bytes of parity)
        ecc_nsym = 0
        if ecc_pct and ecc_pct > 0:
            ecc_nsym = max(2, math.ceil(len(chunk) * ecc_pct / 100.0))
            # keep it reasonable; you can tune these caps
            ecc_nsym = min(ecc_nsym, 128)

        pre_crc = crc32_hex(chunk)
        rs_payload = rs_encode(chunk, ecc_nsym)

        header = {
            "i": i,
            "n": total,
            "fn": os.path.basename(infile),
            "ol": len(raw),         # original length
            "cl": len(comp),        # compressed length
            "rl": len(chunk),       # raw chunk length (before RS)
            "rs": ecc_nsym,         # RS parity bytes used
            "crc": pre_crc,         # CRC of raw chunk (before RS)
            "base": base,
        }
        hjson = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

        # Blob layout: MAGIC + SENTINEL + total_len(4B) + header_len(2B) + header + rs_payload
        body = MAGIC + SENTINEL
        total_len = 2 + 4 + 2 + len(hjson) + len(rs_payload)  # MAGIC+SENTINEL already counted
        body += int(total_len).to_bytes(4, "big")
        body += len(hjson).to_bytes(2, "big")
        body += hjson + rs_payload

        # Map bytes → words
        tokens = bytes_to_words(body, words)

        # Write .words (wrap for OCR: 10 words per line)
        out_txt = outp / f"part_{i:04d}.words"
        with open(out_txt, "w", encoding="utf-8") as f:
            per_line = 10
            for j in range(0, len(tokens), per_line):
                f.write(" ".join(tokens[j:j+per_line]) + "\n")

        print(f"Wrote {out_txt.name}  words={len(tokens)}  (chunk={len(chunk)}B, ecc={ecc_nsym}B)")

    print(f"Done. Parts: {total}. Manifest: {outp/'manifest.json'}")

# ---------- Decode helpers ----------
def parse_blob(blob: bytes) -> tuple[dict, bytes]:
    """
    Expect: MAGIC + SENTINEL + total_len(4B) + header_len(2B) + header_json + rs_payload
    Returns (header_dict, rs_payload_bytes).
    """
    if len(blob) < 2 + 1 + 4 + 2:
        raise ValueError("Blob too short")
    if not blob.startswith(MAGIC):
        raise ValueError("Missing MAGIC")
    if blob[2:3] != SENTINEL:
        raise ValueError("Missing SENTINEL")
    pos = 3
    total_len = int.from_bytes(blob[pos:pos+4], "big"); pos += 4
    hlen = int.from_bytes(blob[pos:pos+2], "big"); pos += 2
    if len(blob) < 3 + 4 + 2 + hlen:
        raise ValueError("Blob incomplete (header)")
    header = json.loads(blob[pos:pos+hlen].decode("utf-8")); pos += hlen
    rs_payload = blob[pos:]
    # (Optional sanity: total_len == 2+4+2+hlen+len(rs_payload))
    return header, rs_payload

# ---------- Main decode ----------
def decode_words_dir(indir: str, wordlist_path: str, outfile: str | None = None):
    """
    Reads *.words files from `indir`, reconstructs the compressed stream,
    zlib-decompresses, verifies sizes (and manifest SHA-256 if present),
    and writes `outfile` (or the original filename from headers).
    """
    words = load_wordlist(wordlist_path)
    paths = sorted([p for p in Path(indir).glob("*.words")])
    if not paths:
        raise SystemExit("No .words files found in input dir.")

    parts: dict[int, bytes] = {}
    meta = None

    # Optional manifest SHA for final verification
    manifest_path = Path(indir) / "manifest.json"
    manifest_sha = None
    if manifest_path.exists():
        try:
            manifest_sha = json.loads(manifest_path.read_text(encoding="utf-8")).get("sha256")
        except Exception:
            manifest_sha = None

    # Read every part
    for p in paths:
        # 1) tokens -> bytes
        tokens = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tokens.extend(line.split())
        blob = words_to_bytes(tokens, words)

        # 2) parse container + header
        header, rs_payload = parse_blob(blob)
        i, n = int(header["i"]), int(header["n"])

        # carry top-level meta from first header
        if meta is None:
            meta = {
                "fn": header["fn"],
                "ol": int(header["ol"]),  # original length
                "cl": int(header["cl"]),  # compressed length
            }
        else:
            # basic consistency checks
            if header["fn"] != meta["fn"]:
                raise SystemExit(f"Inconsistent filename in {p.name}")
            if int(header["cl"]) != meta["cl"] or int(header["ol"]) != meta["ol"]:
                raise SystemExit(f"Inconsistent sizes in {p.name}")

        # 3) RS decode (normalize to bytes)
        ecc_nsym = int(header.get("rs", 0))
        raw_chunk = rs_decode(rs_payload, ecc_nsym)

        # 4) CRC check against pre-RS chunk CRC
        expected_crc = header.get("crc")
        got_crc = crc32_hex(raw_chunk)
        if expected_crc and got_crc != expected_crc:
            raise SystemExit(f"CRC mismatch in {p.name}: {got_crc} != {expected_crc}")

        parts[i] = raw_chunk
        print(f"Read {p.name}  (part {i}/{n}, chunk={len(raw_chunk)}B, ecc={ecc_nsym}B)")

    # Ensure we have all parts 1..N
    n_expected = max(parts.keys())
    missing = [k for k in range(1, n_expected + 1) if k not in parts]
    if missing:
        raise SystemExit(f"Missing parts: {missing}")

    # Reassemble compressed stream (in order)
    comp = b"".join(parts[i] for i in range(1, n_expected + 1))
    if meta and len(comp) != meta["cl"]:
        print(f"Warning: compressed length {len(comp)} != header {meta['cl']}")

    # Decompress
    try:
        raw = zlib.decompress(comp)
    except Exception as e:
        raise SystemExit(f"Decompression failed: {e}")

    if meta and len(raw) != meta["ol"]:
        print(f"Warning: original length {len(raw)} != header {meta['ol']}")

    # Optional final SHA-256 check (if manifest provided)
    if manifest_sha:
        got_sha = hashlib.sha256(raw).hexdigest()
        if got_sha.lower() != manifest_sha.lower():
            raise SystemExit(f"SHA-256 mismatch: got {got_sha}, manifest has {manifest_sha}")
        else:
            print("SHA-256 OK (matches manifest)")

    out_path = outfile or (meta["fn"] if meta else "recovered.bin")
    Path(out_path).write_bytes(raw)
    print(f"Reconstructed file written to: {out_path}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Encode/Decode files as words (compression + RS ECC).")
    sub = ap.add_subparsers(dest="cmd")

    enc = sub.add_parser("encode", help="Encode to .words blocks")
    enc.add_argument("-i", "--input", required=True)
    enc.add_argument("-w", "--wordlist", required=True)
    enc.add_argument("-o", "--output", required=True)
    enc.add_argument("--chunk-words", type=int, default=1000,
                     help="words per chunk (default 1000). For 65,536 list ≈ 2000 bytes/chunk.")
    enc.add_argument("--ecc-pct", type=int, default=10,
                     help="Reed–Solomon parity percent per chunk (default 10). Use 0 to disable.")

    dec = sub.add_parser("decode", help="Decode .words blocks")
    dec.add_argument("-i", "--input", required=True, help="directory containing part_*.words")
    dec.add_argument("-w", "--wordlist", required=True)
    dec.add_argument("-o", "--output", help="output file path (defaults to original filename)")

    args = ap.parse_args()
    if args.cmd == "encode":
        encode_file_to_words(args.input, args.wordlist, args.output, args.chunk_words, args.ecc_pct)
    elif args.cmd == "decode":
        decode_words_dir(args.input, args.wordlist, args.output)
    else:
        ap.print_help()

if __name__ == "__main__":
    main()
