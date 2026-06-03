# NeuralQuad

NeuralQuad provides a practical quad-remeshing workflow built around trained cross fields.

At a high level, the pipeline is:

1. Start from a triangle mesh.
2. Train a cross field with the bundled `NeurCross` source.
3. Optionally convert the saved cross-field output to `.rosy`.
4. Extract an aligned quad mesh with `pyquadwild`.

The repository is split by responsibility:

- `neuralquad/`: quad mesh extraction package and CLI.
- `third_party/NeurCross/`: cross-field training and cross-field to `.rosy` conversion.
- `third_party/pyquadwild/`: quad extraction backend dependency.
- `third_party/Directional/`: related research dependency and buildable third-party library.

## Requirements

- Python `>=3.10`
- A working C/C++ build environment may be required by transitive dependencies such as `pyquadwild`
- For NeurCross training:
  - `neurcross` (installed via `pip install .\third_party\NeurCross`)
  - `torch`
  - optional CUDA-capable GPU if you want GPU training

## Python Dependencies

The root `NeuralQuad` package declares these runtime dependencies:

- `numpy`
- `scipy`
- `timm`
- `trimesh`
- `pyquadwild @ git+https://github.com/akashskypatel/pyquadwild.git`

## Installation

Install the root package from this repository:

```powershell
python -m pip install .
```

For editable development install:

```powershell
python -m pip install -e .
```

This installs the root extraction CLI:

```powershell
neuralquad-extract-quad-mesh
```

If you also want the standalone NeurCross training commands, install the bundled subproject separately:

```powershell
python -m pip install .\third_party\NeurCross
```

## Usage

### End-to-End Workflow

1. Train NeurCross on a triangle mesh to generate cross-field snapshots.
2. Use the latest saved cross-field snapshot or a `.rosy` file as extraction input.
3. Run NeuralQuad extraction to generate the final quad OBJ.

### Train a Cross Field

From the bundled NeurCross project:

```powershell
neurcross-train-quad-mesh --data_path D:\path\to\mesh.ply
```

NeurCross writes saved cross-field snapshots under a `save_crossField` directory in the training output location.

### Convert a Cross Field to `.rosy`

If you want a standalone `.rosy` file from a saved NeurCross cross-field snapshot:

```powershell
neurcross-crossfield-to-rosy D:\path\to\save_crossField\mesh_iter_999.txt
```

### Extract a Quad Mesh From a `.rosy` File

```powershell
neuralquad-extract-quad-mesh D:\path\to\mesh.ply D:\path\to\field.rosy
```

Equivalent module form:

```powershell
python -m neuralquad.extract_quad_mesh D:\path\to\mesh.ply D:\path\to\field.rosy
```

### Extract a Quad Mesh From a NeurCross Cross-Field File

NeuralQuad can take a saved NeurCross cross-field `.txt` directly:

```powershell
neuralquad-extract-quad-mesh D:\path\to\mesh.ply D:\path\to\save_crossField\mesh_iter_999.txt
```

When given a NeurCross `.txt` file, NeuralQuad will:

1. Read the saved cross field.
2. Write a sidecar `*_crossfield.txt`.
3. Convert that to `*.rosy`.
4. Run `pyquadwild` with the generated `.rosy`.
5. Write the final quad mesh as OBJ.

### Output Path Behavior

The third argument is optional:

```powershell
neuralquad-extract-quad-mesh D:\path\to\mesh.ply D:\path\to\field.rosy D:\path\to\output.obj
```

If you pass a directory instead of a filename, NeuralQuad creates a default OBJ name inside that directory:

```powershell
neuralquad-extract-quad-mesh D:\path\to\mesh.ply D:\path\to\mesh_iter_999.txt D:\output_dir
```

This resolves to:

```text
<output_dir>\<mesh-stem>_<field-stem>_quad.obj
```

If no output path is provided, the default output is written beside the input mesh using the same naming pattern.

## Generated Files

Depending on the input field type, extraction may produce:

- final quad mesh OBJ
- generated `*.rosy` file
- generated `*_crossfield.txt` file
- `pyquadwild_debug/` debug directory

## Programmatic API

The root Python package exposes:

- `neuralquad.extract_quad_mesh(mesh_path, field_path, output_path=None)`

`extract_quad_mesh()` accepts either:

- a triangulated input mesh
- a `.rosy` field file
- a NeurCross cross-field `.txt` file

```python
python -m neuralquad.extract_quad_mesh mesh.ply field.rosy output.obj
# or if using a NeurCross cross-field file:
python -m neuralquad.extract_quad_mesh mesh.ply crossfield.txt output.obj
```

## Acknowledgments

This project is built on top of the following excellent work:

- [NeurCross](https://github.com/QiujieDong/NeurCross) - Neural cross field learning
