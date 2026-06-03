from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

def _safe_normalize(vectors):
    import numpy as np

    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return vectors / norms


def _load_triangle_mesh(mesh_path):
    import trimesh

    mesh = trimesh.load_mesh(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(
            g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)
        ))
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f'Unsupported mesh type loaded from {mesh_path!r}: {type(mesh)!r}')
    if mesh.faces.shape[1] != 3:
        raise ValueError('Quad extraction currently requires a triangle mesh input.')
    return mesh


def _clean_face_indices(face):
    cleaned = []
    for index in face:
        idx = int(index)
        if not cleaned or cleaned[-1] != idx:
            cleaned.append(idx)
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned


def _triangulate_faces(faces):
    import numpy as np

    triangles = []
    for face in faces:
        cleaned = _clean_face_indices(face)
        if len(cleaned) < 3:
            continue
        for offset in range(1, len(cleaned) - 1):
            triangles.append([cleaned[0], cleaned[offset], cleaned[offset + 1]])
    return np.asarray(triangles, dtype=np.int64)


def _mesh_topology_stats(vertices, faces):
    import trimesh

    edge_counts = {}
    normalized_faces = []
    for face in faces:
        cleaned = _clean_face_indices(face)
        if len(cleaned) < 3:
            continue
        normalized_faces.append(cleaned)
        for i, start in enumerate(cleaned):
            end = cleaned[(i + 1) % len(cleaned)]
            edge = tuple(sorted((int(start), int(end))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    boundary_edges = sum(1 for count in edge_counts.values() if count == 1)
    nonmanifold_edges = sum(1 for count in edge_counts.values() if count > 2)
    tri_faces = _triangulate_faces(normalized_faces)
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=tri_faces, process=False)
    return {
        'components': len(tri_mesh.split(only_watertight=False)),
        'boundary_edges': int(boundary_edges),
        'nonmanifold_edges': int(nonmanifold_edges),
        'is_watertight': bool(boundary_edges == 0 and nonmanifold_edges == 0 and tri_mesh.is_watertight),
    }


def _write_obj(path, vertices, faces):
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('# NeuralQuad quad mesh extraction via pyquadwild\n')
        for vertex in vertices:
            handle.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
        for face in faces:
            cleaned_face = _clean_face_indices(face)
            handle.write('f {}\n'.format(' '.join(str(int(index) + 1) for index in cleaned_face)))


def _write_crossfield_txt(path, alpha, beta):
    cross_field = np.concatenate((alpha, beta), axis=-1)
    np.savetxt(path, cross_field)
    return path


def _load_crossfield_txt(crossfield_path):
    import numpy as np

    cross_field = np.loadtxt(crossfield_path, dtype=np.float64)
    if cross_field.ndim == 1:
        cross_field = cross_field.reshape(1, -1)
    if cross_field.shape[1] < 6:
        raise ValueError(f'Expected at least 6 columns in cross-field file: {crossfield_path}')
    alpha = cross_field[:, 0:3]
    beta = cross_field[:, 3:6]
    return alpha, beta


def _convert_crossfield_to_rosy(input_path: Path, output_path: Path | None = None) -> Path:
    input_path = Path(input_path)
    output_path = input_path.with_suffix(".rosy") if output_path is None else Path(output_path)
    alpha, _beta = _load_crossfield_txt(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as outfile:
        outfile.write(f"{len(alpha)}\n")
        outfile.write("4\n")
        for x, y, z in _safe_normalize(alpha):
            outfile.write(f"{x} {y} {z}\n")
    return output_path


def _snapshot_iteration_key(path):
    match = re.search(r'_iter_(\d+)\.txt$', os.path.basename(path))
    return int(match.group(1)) if match else -1


def load_latest_crossfield_snapshot(crossfield_paths):
    if not crossfield_paths:
        raise ValueError('No cross-field snapshots were provided.')
    latest_path = max((str(path) for path in crossfield_paths), key=_snapshot_iteration_key)
    alpha, beta = _load_crossfield_txt(latest_path)
    return _safe_normalize(alpha), _safe_normalize(beta), latest_path


def _estimate_target_quad_count(num_triangles):
    return max(1, math.ceil(num_triangles / 2))


def _default_output_filename(mesh_path: Path, field_path: Path) -> str:
    return f"{mesh_path.stem}_{field_path.stem}_quad.obj"


def _resolve_output_path(mesh_path: Path, field_path: Path, output_path: Path | None) -> Path:
    if output_path is None:
        return mesh_path.with_name(_default_output_filename(mesh_path, field_path))

    output_path = Path(output_path)
    if output_path.exists() and output_path.is_dir():
        return output_path / _default_output_filename(mesh_path, field_path)
    if output_path.suffix == "":
        return output_path / _default_output_filename(mesh_path, field_path)
    return output_path


def _extract_quad_mesh_from_rosy(mesh_path, rosy_path, output_path):
    import numpy as np
    import pyquadwild
    import trimesh

    mesh = _load_triangle_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    triangles = np.asarray(mesh.faces, dtype=np.int64)
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    quadwild = pyquadwild.QuadWild()
    quad_vertices, quad_faces = quadwild.remesh(
        trimesh.Trimesh(vertices=vertices, faces=triangles, process=False),
        enable_preprocess=False,
        enable_sharp=True,
        sharp_angle=35.0,
        field_path=rosy_path,
        target_quad_count=_estimate_target_quad_count(len(triangles)),
        output_format='arrays',
        debug_dir=os.path.join(output_dir, 'pyquadwild_debug'),
    )

    stats = _mesh_topology_stats(quad_vertices, quad_faces)
    _write_obj(output_path, quad_vertices, quad_faces)
    return {
        'quad_vertices': quad_vertices,
        'quad_faces': quad_faces,
        'rosy_path': str(rosy_path),
        'output_path': output_path,
        'topology': stats,
        'extractor': 'pyquadwild',
    }


def extract_quad_mesh_from_field(mesh_path, alpha, beta, output_path):
    import numpy as np

    mesh = _load_triangle_mesh(mesh_path)
    triangles = np.asarray(mesh.faces, dtype=np.int64)
    if alpha.shape != beta.shape or alpha.shape[1] != 3:
        raise ValueError('alpha and beta must both have shape (num_faces, 3).')
    if alpha.shape[0] != triangles.shape[0]:
        raise ValueError('Cross field face count must match the mesh face count.')

    alpha = _safe_normalize(np.asarray(alpha, dtype=np.float64))
    beta = _safe_normalize(np.asarray(beta, dtype=np.float64))
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    output_base = os.path.splitext(os.path.abspath(output_path))[0]
    crossfield_path = _write_crossfield_txt(output_base + '_crossfield.txt', alpha, beta)
    rosy_path = _convert_crossfield_to_rosy(Path(crossfield_path), Path(output_base + '.rosy'))
    result = _extract_quad_mesh_from_rosy(mesh_path, rosy_path, output_path)
    result['crossfield_path'] = crossfield_path
    return result


def extract_quad_mesh(mesh_path: Path, field_path: Path, output_path: Path | None = None) -> Path:
    mesh_path = Path(mesh_path)
    field_path = Path(field_path)
    output_path = _resolve_output_path(mesh_path, field_path, output_path)

    if not mesh_path.is_file():
        raise FileNotFoundError(f"Input mesh file was not found: {mesh_path}")
    if not field_path.is_file():
        raise FileNotFoundError(f"Input field file was not found: {field_path}")

    if field_path.suffix.lower() == ".rosy":
        _extract_quad_mesh_from_rosy(str(mesh_path), str(field_path), str(output_path))
    elif field_path.suffix.lower() == ".txt":
        alpha, beta = _load_crossfield_txt(field_path)
        extract_quad_mesh_from_field(str(mesh_path), alpha, beta, str(output_path))
    else:
        raise ValueError(
            f"Field file extension {field_path.suffix!r} is not supported. Provide a .rosy or .txt file."
        )
    return output_path


def extract_quad_mesh_from_saved_crossfields(mesh_path, crossfield_paths, output_path):
    alpha, beta, latest_path = load_latest_crossfield_snapshot(crossfield_paths)
    result = extract_quad_mesh_from_field(mesh_path, alpha, beta, output_path)
    result['source_crossfield_path'] = latest_path
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract a quad mesh from an input triangle mesh using a .rosy or cross-field (.txt) file."
    )
    parser.add_argument("mesh_path", type=Path, help="Path to the input triangle mesh.")
    parser.add_argument(
        "field_path",
        type=Path,
        help="Path to the orientation field. Use a .rosy file (QuadWild format) or a .txt cross-field file (NeurCross format).",
    )
    parser.add_argument(
        "output_path",
        nargs="?",
        type=Path,
        help="Optional output OBJ path. Defaults to <mesh-stem>_<field-stem>_quad.obj.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = extract_quad_mesh(args.mesh_path, args.field_path, args.output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
