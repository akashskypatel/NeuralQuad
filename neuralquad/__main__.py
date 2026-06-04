from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m neuralquad",
        description=(
            "NeuralQuad extracts cross-field-aligned quad meshes from triangle meshes. "
            "Use it with NeurCross cross-field output, .rosy files, or Directional .rawfield files."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", metavar="command")

    extract = subparsers.add_parser(
        "extract-quad-mesh",
        help="Extract a quad mesh from a triangle mesh and field input.",
    )
    extract.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    parser.epilog = (
        "High-level functionality:\n"
        "  extract-quad-mesh  Generate a quad mesh from .txt, .rosy, or .rawfield input.\n\n"
        "Examples:\n"
        "  python -m neuralquad --help\n"
        "  python -m neuralquad extract-quad-mesh --help\n"
        "  python -m neuralquad extract-quad-mesh mesh.ply field.rosy output.obj"
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "extract-quad-mesh":
        from .extract_quad_mesh import main as extract_main
        import sys

        sys.argv = ["neuralquad.extract_quad_mesh", *args.args]
        extract_main()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
