import torch


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle rotations to rotation matrices.

    axis_angle: (..., 3)
    returns: (..., 3, 3)
    """
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / (angle + 1e-8)
    x, y, z = axis.unbind(dim=-1)
    cos = torch.cos(angle).squeeze(-1)
    sin = torch.sin(angle).squeeze(-1)
    one = 1.0 - cos

    r00 = cos + x * x * one
    r01 = x * y * one - z * sin
    r02 = x * z * one + y * sin
    r10 = y * x * one + z * sin
    r11 = cos + y * y * one
    r12 = y * z * one - x * sin
    r20 = z * x * one - y * sin
    r21 = z * y * one + x * sin
    r22 = cos + z * z * one

    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """Convert rotation matrices to Euler angles (radians)."""
    if convention != "XYZ":
        raise ValueError(f"Unsupported convention: {convention}")
    r00 = matrix[..., 0, 0]
    r10 = matrix[..., 1, 0]
    r20 = matrix[..., 2, 0]
    r21 = matrix[..., 2, 1]
    r22 = matrix[..., 2, 2]
    r12 = matrix[..., 1, 2]
    r11 = matrix[..., 1, 1]

    sy = torch.sqrt(r00 * r00 + r10 * r10)
    singular = sy < 1e-6

    x = torch.atan2(r21, r22)
    y = torch.atan2(-r20, sy)
    z = torch.atan2(r10, r00)

    x_s = torch.atan2(-r12, r11)
    z_s = torch.zeros_like(z)

    x = torch.where(singular, x_s, x)
    z = torch.where(singular, z_s, z)

    return torch.stack([x, y, z], dim=-1)
