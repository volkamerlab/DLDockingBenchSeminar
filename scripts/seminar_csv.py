#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seminar_csv.py - map a seminar / full-split CSV to KarmaDock complex records.

Two CSV schemas occur in this project and must map to the SAME complex id (the
.dgl graph stem that preprocessing writes and training reads):

  * prototype CSVs   carry explicit `ligand_file_name` / `protein_file_name` columns
    (e.g. proto_train.csv, proto_test.csv).
  * full-split CSVs  carry raw PDBBind/MOAD metadata; the refined filenames are
    {PDBID}_{Ligand Name}_{Ligand Chain}_{Ligand Residue Number}_ligand_refined.sdf
    and ..._protein_refined.pdb (e.g. full_train.csv, full_val.csv).

Centralising the mapping here keeps the converter (preprocess) and train.py
(training) in agreement: if they derived the id differently the .dgl stems would
not match and training would silently see zero complexes. Columns are read as
strings so a numeric residue number is never coerced to '210.0'.
"""
import pandas as pd

ID_COLUMNS = ["PDBID", "Ligand Name", "Ligand Chain", "Ligand Residue Number"]


def _strip_suffix(ligand_filename):
    return ligand_filename.replace("_ligand_refined.sdf", "").replace("_ligand.sdf", "")


def _rows_to_triples(df):
    """Yield (complex_id, ligand_filename, protein_filename) per CSV row, either schema."""
    if "ligand_file_name" in df.columns:
        for lig, prot in zip(df["ligand_file_name"], df["protein_file_name"]):
            yield _strip_suffix(lig), lig, prot
    else:
        for parts in zip(*[df[c] for c in ID_COLUMNS]):
            cid = "_".join(parts)
            yield cid, f"{cid}_ligand_refined.sdf", f"{cid}_protein_refined.pdb"


def complex_records(csv_path):
    """Unique (complex_id, ligand_filename, protein_filename), in first-seen order."""
    df = pd.read_csv(csv_path, dtype=str)
    seen, records = set(), []
    for cid, lig, prot in _rows_to_triples(df):
        if cid not in seen:
            seen.add(cid)
            records.append((cid, lig, prot))
    return records


def complex_ids(csv_path):
    """The unique complex ids (== .dgl graph stems), in first-seen order."""
    return [cid for cid, _lig, _prot in complex_records(csv_path)]
