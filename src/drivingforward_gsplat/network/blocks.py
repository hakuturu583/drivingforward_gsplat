def pack_cam_feat(x):
    if isinstance(x, dict):
        for k, v in x.items():
            b, n_cam = v.shape[:2]
            x[k] = v.view(b * n_cam, *v.shape[2:])
        return x
    else:
        b, n_cam = x.shape[:2]
        x = x.view(b * n_cam, *x.shape[2:])
    return x


def unpack_cam_feat(x, b, n_cam):
    if isinstance(x, dict):
        for k, v in x.items():
            x[k] = v.view(b, n_cam, *v.shape[1:])
        return x
    else:
        x = x.view(b, n_cam, *x.shape[1:])
    return x
