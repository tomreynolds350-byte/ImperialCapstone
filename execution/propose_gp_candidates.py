from __future__ import annotations

from bo_core import build_gp_candidate_parser, run_gp_candidate_script


def main() -> None:
    parser = build_gp_candidate_parser(
        description="Propose GP-based candidates for each function.",
        seed_default=20260204,
        output_prefix="round_02",
    )
    args = parser.parse_args()
    run_gp_candidate_script(args)


if __name__ == "__main__":
    main()
