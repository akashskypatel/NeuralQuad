from __future__ import annotations

import argparse
import sys
from pathlib import Path


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
    convert = subparsers.add_parser(
        "convert",
        help="Convert a mesh to a different format.",
    )
    extract.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    convert.add_argument("args", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)

    parser.epilog = (
        "High-level functionality:\n"
        "  extract-quad-mesh  Generate a quad mesh from .txt, .rosy, or .rawfield input.\n\n"
        "  convert  Convert a mesh to a different format.\n\n"
        "Examples:\n"
        "  `python -m neuralquad --help`\n"
        "  `python -m neuralquad extract-quad-mesh --help`\n"
        "  `python -m neuralquad extract-quad-mesh mesh.ply field.rosy output.obj`\n"
        "  `python -m neuralquad convert input.obj stl`"
    )
    return parser


def main() -> None:
    parser = build_parser()
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        parser.print_help()
        return

    command, command_args = argv[0], argv[1:]

    if command == "extract-quad-mesh":
        from neuralquad.extract_quad_mesh import extract_main

        sys.argv = ["neuralquad.extract_quad_mesh", *command_args]
        extract_main()
        return

    if command == "convert":
        from neuralquad.extract_quad_mesh import convert_main

        sys.argv = ["neuralquad.extract_quad_mesh", *command_args]
        convert_main()
        return

    parser.error(f"unknown command: {command}")


if __name__ == "__main__":
    main()
