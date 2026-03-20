from __future__ import annotations

from pathlib import Path

from bo_core import build_round_candidate_parser, run_round_candidate_script


def main() -> None:
    parser = build_round_candidate_parser(
        description="Ingest latest round outputs and propose Round 04 candidates (GP + NN hybrid).",
        inputs_default=Path(r"c:\Users\tom_m\Downloads\inputs.txt"),
        outputs_default=Path(r"c:\Users\tom_m\Downloads\outputs.txt"),
        seed_default=20260218,
        prefix_default="round_04",
    )
    args = parser.parse_args()
    run_round_candidate_script(args, snapshot_filename="round_03_outputs_canonical.txt")


if __name__ == "__main__":
    main()
