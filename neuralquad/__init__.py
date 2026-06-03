__version__ = "0.1.0"


def extract_quad_mesh(*args, **kwargs):
    from .extract_quad_mesh import extract_quad_mesh as _extract_quad_mesh

    return _extract_quad_mesh(*args, **kwargs)


def extract_quad_mesh_from_field(*args, **kwargs):
    from .extract_quad_mesh import extract_quad_mesh_from_field as _extract_quad_mesh_from_field

    return _extract_quad_mesh_from_field(*args, **kwargs)


def extract_quad_mesh_from_saved_crossfields(*args, **kwargs):
    from .extract_quad_mesh import (
        extract_quad_mesh_from_saved_crossfields as _extract_quad_mesh_from_saved_crossfields,
    )

    return _extract_quad_mesh_from_saved_crossfields(*args, **kwargs)


__all__ = [
    "__version__",
    "extract_quad_mesh",
    "extract_quad_mesh_from_field",
    "extract_quad_mesh_from_saved_crossfields",
]
