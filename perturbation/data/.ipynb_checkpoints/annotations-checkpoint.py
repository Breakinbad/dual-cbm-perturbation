from __future__ import annotations

from collections import Counter
from typing import Any

from mygene import MyGeneInfo


mg = MyGeneInfo()


def ensembl_to_symbol(ensembl_ids: list[str], species: str = "human") -> dict[str, str]:
    results = mg.querymany(
        ensembl_ids,
        scopes="ensembl.gene",
        fields="symbol",
        species=species,
    )
    mapping: dict[str, str] = {}
    for row in results:
        query = row.get("query")
        if not query:
            continue
        mapping[query] = row.get("symbol", query)
    return mapping


def query_interpro(gene_symbols: list[str], species: str = "human") -> dict[str, dict[str, Any]]:
    results = mg.querymany(
        gene_symbols,
        scopes="symbol",
        fields="interpro,name",
        species=species,
        returnall=True,
    )
    out: dict[str, dict[str, Any]] = {}
    for row in results.get("out", []):
        query = row.get("query")
        if query and query not in out:
            out[query] = row
    return out


def infer_primary_family(annotation: dict[str, Any]) -> str | None:
    interpros = annotation.get("interpro")
    if not interpros:
        return None
    if isinstance(interpros, dict):
        interpros = [interpros]
    names = [entry.get("desc") for entry in interpros if isinstance(entry, dict) and entry.get("desc")]
    if not names:
        return None
    return Counter(names).most_common(1)[0][0]


def build_drug_to_primary_family(
    drug_to_target_vec: dict[str, Any],
    gene_vocab: dict[str, Any],
    annotations: dict[str, dict[str, Any]],
) -> dict[str, str | None]:
    genes = gene_vocab["genes"]
    output: dict[str, str | None] = {}
    for drug, target_vec in drug_to_target_vec.items():
        active = [genes[i] for i, value in enumerate(target_vec.tolist()) if value == 1]
        families = [infer_primary_family(annotations[g]) for g in active if g in annotations]
        families = [f for f in families if f]
        output[drug] = Counter(families).most_common(1)[0][0] if families else None
    return output
