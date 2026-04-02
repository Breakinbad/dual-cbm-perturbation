from __future__ import annotations

import argparse
import json

from perturbation.data.annotations import build_drug_to_primary_family, query_interpro
from perturbation.data.io import load_torch_object


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene-vocab", required=True)
    parser.add_argument("--drug-target-vectors", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    gene_vocab = load_torch_object(args.gene_vocab)
    drug_to_target_vec = load_torch_object(args.drug_target_vectors)
    annotations = query_interpro(gene_vocab["genes"])
    result = build_drug_to_primary_family(drug_to_target_vec, gene_vocab, annotations)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
