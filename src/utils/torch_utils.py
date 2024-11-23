from inspect import isfunction

from einops import repeat


def exists(x):
    return x is not None


def expand_dims_like(x, y):
    while x.dim() != y.dim():
        x = x.unsqueeze(-1)
    return x


def default(val, d):
    if exists(val):
        return val
    else:
        return d() if isfunction(d) else d


def repeat_as_img_seq(x, num_frames):
    if x is not None:
        if isinstance(x, list):
            new_x = list()
            for item_x in x:
                new_x += [item_x] * num_frames
            return new_x
        else:
            x = x.unsqueeze(1)
            x = repeat(x, "b 1 ... -> (b t) ...", t=num_frames)
            return x
    else:
        return None


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""

    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"Input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]
