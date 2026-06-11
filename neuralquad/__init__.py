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


def convert_field(*args, **kwargs):
    from .field_conversion import convert_field as _convert_field

    return _convert_field(*args, **kwargs)


def convert_crossfield_to_rosy(*args, **kwargs):
    from .field_conversion import convert_crossfield_to_rosy as _convert_crossfield_to_rosy

    return _convert_crossfield_to_rosy(*args, **kwargs)


def convert_crossfield_to_rawfield(*args, **kwargs):
    from .field_conversion import convert_crossfield_to_rawfield as _convert_crossfield_to_rawfield

    return _convert_crossfield_to_rawfield(*args, **kwargs)


def convert_rawfield_to_crossfield(*args, **kwargs):
    from .field_conversion import convert_rawfield_to_crossfield as _convert_rawfield_to_crossfield

    return _convert_rawfield_to_crossfield(*args, **kwargs)


def convert_rawfield_to_rosy(*args, **kwargs):
    from .field_conversion import convert_rawfield_to_rosy as _convert_rawfield_to_rosy

    return _convert_rawfield_to_rosy(*args, **kwargs)


def convert_rosy_to_crossfield(*args, **kwargs):
    from .field_conversion import convert_rosy_to_crossfield as _convert_rosy_to_crossfield

    return _convert_rosy_to_crossfield(*args, **kwargs)


def convert_rosy_to_rawfield(*args, **kwargs):
    from .field_conversion import convert_rosy_to_rawfield as _convert_rosy_to_rawfield

    return _convert_rosy_to_rawfield(*args, **kwargs)


__all__ = [
    "__version__",
    "convert_crossfield_to_rawfield",
    "convert_crossfield_to_rosy",
    "convert_field",
    "convert_rawfield_to_crossfield",
    "convert_rawfield_to_rosy",
    "convert_rosy_to_crossfield",
    "convert_rosy_to_rawfield",
    "extract_quad_mesh",
    "extract_quad_mesh_from_field",
    "extract_quad_mesh_from_saved_crossfields",
]
