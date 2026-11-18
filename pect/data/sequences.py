ndard library and numpy are used, so no pyfaidx dependency.
"""
from __future__ import annotations

import gzip
import os
import re
from typing import Dict, Iterable, Optional, Tuple

_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


class FastaIndex:
    """Minimal random-access FASTA reader (samtools .fai format).

    The genome FASTA must be uncompressed (or bgzip is NOT supported). If no
    ``.fai`` exists it is built by one linear scan.
    """

    def __init__(self, fasta_path: str):
        if fasta_path.endswith(".gz"):
            raise ValueError("Please decompress the genome FASTA (gunzip) first.")
        self.path = fasta_path
        fai = fasta_path + ".fai"
        if not os.path.exists(fai):
            self._build_fai(fai)
        self.index: Dict[str, Tuple[int, int, int, int]] = {}
        with open(fai) as f:
            for line in f:
                name, length, offset, lbases, lbytes = line.split("\t")[:5]
                self.index[name] = (int(length), int(offset), int(lbases), int(lbytes))
        self._fh = open(fasta_path, "rb")

    def _build_fai(self, fai):
        entries = []
        with open(self.path, "rb") as f:
            name, length, offset, lbases, lbytes = None, 0, 0, 0, 0
            pos = 0
            for raw in f:
                if raw.startswith(b">"):
                    if name is not None:
                        entries.append((name, length, offset, lbases, lbytes))
                    name = raw[1:].split()[0].decode()
                    length, lbases, lbytes = 0, 0, 0
                    offset = pos + len(raw)
                else:
                    seq = raw.rstrip(b"\r\n")
                    if lbases == 0:
                        lbases, lbytes = len(seq), len(raw)
                    length += len(seq)
                pos += len(raw)
            if name is not None:
                entries.append((name, length, offset, lbases, lbytes))
        with open(fai, "w") as out:
            for e in entries:
                out.write("\t".join(map(str, e)) + "\n")

    def resolve(self, chrom: str) -> Optional[str]:
        if chrom in self.index:
            return chrom
        alt = chrom[3:] if chrom.startswith("chr") else "chr" + chrom
        if alt in self.index:
            return alt
        if chrom in ("MT", "chrM"):
            for c in ("MT", "chrM", "M"):
                if c in self.index:
                    return c
        return None

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """0-based half-open [start, end); out-of-range positions become N."""
        name = self.resolve(chrom)
        if name is None:
            return "N" * (end - start)
        length, offset, lbases, lbytes = self.index[name]
        left_pad = max(0, -start)
        right_pad = max(0, end - length)
        s, e = max(0, start), min(length, end)
        if e <= s:
            return "N" * (end - start)
        b0 = offset + (s // lbases) * lbytes + s % lbases
        b1 = offset + ((e - 1) // lbases) * lbytes + (e - 1) % lbases + 1
        self._fh.seek(b0)
        seq = self._fh.read(b1 - b0).replace(b"\n", b"").replace(b"\r", b"").decode()
        return "N" * left_pad + seq.upper() + "N" * right_pad


_ATTR = re.compile(r'(\S+) "([^"]*)"')


def read_gtf_tss(gtf_path: str) -> Tuple[Dict[str, tuple], Dict[str, tuple]]:
    """Return {gene_name: (chrom, tss, strand)} and {gene_id: (...)}."""
    by_name, by_id = {}, {}
    with _open(gtf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "gene":
                continue
            chrom, start, end, strand = parts[0], int(parts[3]), int(parts[4]), parts[6]
            attrs = dict(_ATTR.findall(parts[8]))
            tss = start - 1 if strand == "+" else end - 1  # 0-based TSS position
            rec = (chrom, tss, strand)
            gid = attrs.get("gene_id", "").split(".")[0]
            gname = attrs.get("gene_name")
            if gid:
                by_id.setdefault(gid, rec)
            if gname:
                by_name.setdefault(gname.upper(), rec)
    return by_name, by_id


def extract_tss_sequences(
    genes: Iterable[str],
    fasta: str,
    gtf: str,
    length: int = 1000,
    gene_ids: Optional[Iterable[str]] = None,
):
    """Sequences of ``length`` bp centred on the TSS of every gene.

    Returns ``(sequences, found)`` where ``sequences`` maps gene -> str and
    ``found`` maps gene -> bool (False when the gene is not in the GTF).
    """
    by_name, by_id = read_gtf_tss(gtf)
    fa = FastaIndex(fasta)
    genes = list(genes)
    gene_ids = list(gene_ids) if gene_ids is not None else [None] * len(genes)
    half = length // 2
    seqs, found = {}, {}
    for g, gid in zip(genes, gene_ids):
        rec = None
        if gid:
            rec = by_id.get(str(gid).split(".")[0])
        if rec is None:
            rec = by_name.get(g.upper())
        if rec is None:
            found[g] = False
            continue
        chrom, tss, strand = rec
        if strand == "-":
            # upstream lies at higher coordinates: [tss-(L-half)+1, tss+half+1), reverse-complemented
            s = revcomp(fa.fetch(chrom, tss - (length - half) + 1, tss + half + 1))
        else:
            s = fa.fetch(chrom, tss - half, tss - half + length)
        seqs[g] = s
        found[g] = True
    return seqs, found


def write_fasta(seqs: Dict[str, str], path: str):
    with open(path, "w") as f:
        for g, s in seqs.items():
            f.write(f">{g}\n")
            for i in range(0, len(s), 80):
                f.write(s[i : i + 80] + "\n")


def read_fasta(path: str) -> Dict[str, str]:
    seqs, name, buf = {}, None, []
    with _open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            else:
                buf.append(line)
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs
