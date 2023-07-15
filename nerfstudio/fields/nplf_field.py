"""Normal Plane Light Field field"""


from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.encodings import Encoding, Identity
from nerfstudio.field_components.field_heads import DensityFieldHead, FieldHead, FieldHeadNames, RGBFieldHead
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import FieldConfig


def get_rotation_mat(source: Tensor, target: Tensor) -> Tensor:
    """
    Get rotation matrix using Rodrigues' Rotation Formula
    """
    batchsize = source.shape[0]
    # Normalizing source and target vector
    source = source / torch.norm(source, p=2, dim=1).unsqueeze(1)
    target = target / torch.norm(target, p=2, dim=1).unsqueeze(1)
    n_vec = torch.cross(source, target, dim=1)  # Rotation axis
    theta = torch.acos(torch.sum(source * target, dim=1))

    n_vec = n_vec / torch.norm(n_vec, p=2, dim=1).unsqueeze(1)
    n_hat = torch.reshape(
        torch.stack(
            [
                torch.zeros(batchsize),
                -n_vec[:, 2],
                n_vec[:, 1],
                n_vec[:, 2],
                torch.zeros(batchsize),
                -n_vec[:, 0],
                -n_vec[:, 1],
                n_vec[:, 0],
                torch.zeros(batchsize),
            ]
        ).transpose(0, 1),
        (batchsize, 3, 3),
    )

    rot = (
        torch.stack([torch.eye(3)] * batchsize)
        + torch.sin(theta).view(-1, 1, 1) * n_hat
        + (1 - torch.cos(theta).view(-1, 1, 1)) * n_hat.matmul(n_hat)
    )
    return rot


def ray_to_np(ray_bundle: RayBundle, compress2d=False) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Converting (x,y,z,theta,phi) to (u,v,theta,phi,rho)
    for normal plane light fields
    """
    batchsize = ray_bundle.origins.shape[0]
    x_dot_d = torch.sum(ray_bundle.origins * ray_bundle.directions, dim=1)
    d_len = torch.norm(ray_bundle.directions, p=2, dim=1)
    rho = x_dot_d / d_len  # [batch]
    rho_vec = (ray_bundle.directions / d_len.unsqueeze(1)) * rho.unsqueeze(1)  # [batch, 3]
    np_vec = ray_bundle.origins - rho_vec
    # Rotate np_vec from normal plane to xOy plane,
    # where the rotation is same as "direction" to (0,0,1)
    rot = get_rotation_mat(ray_bundle.directions, torch.Tensor([[0, 0, 1]] * batchsize))

    if compress2d:
        # Positional (deprecate z)
        raw_pos = rot.matmul(np_vec.view(-1, 3, 1))
        if not (raw_pos[:, 2, 0] == 0).all():
            print("Error during rotation, pos:", raw_pos)
            raise RuntimeError()
        pos = raw_pos[:, :2, 0]
        # Directional (Euclidean to Radian)
        thetas = torch.acos(
            ray_bundle.directions[:, 0] / (ray_bundle.directions[:, 0] ** 2 + ray_bundle.directions[:, 1] ** 2).sqrt()
        )
        phis = torch.acos(ray_bundle.directions[:, 2] / d_len)

        return pos, torch.stack((thetas, phis)).transpose(0, 1), rho

    return rot.matmul(np_vec.view(-1, 3, 1)), ray_bundle.directions, rho


class NPLFField(nn.Module):
    """NPLF Field

    Args:
        position_encoding: Position encoder.
        direction_encoding: Direction encoder.
        base_mlp_num_layers: Number of layers for base MLP.
        base_mlp_layer_width: Width of base MLP layers.
        head_mlp_num_layers: Number of layer for output head MLP.
        head_mlp_layer_width: Width of output head MLP layers.
        skip_connections: Where to add skip connection in base MLP.
        use_integrated_encoding: Used integrated samples as encoding input.
        spatial_distortion: Spatial distortion.
    """

    def __init__(
        self,
        position_encoding: Encoding = Identity(in_dim=2),
        direction_encoding: Encoding = Identity(in_dim=2),
        base_mlp_num_layers: int = 8,
        base_mlp_layer_width: int = 256,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 128,
        skip_connections: Tuple[int] = (4,),
        field_heads: Optional[Tuple[FieldHead]] = (RGBFieldHead(),),
        use_integrated_encoding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding
        self.use_integrated_encoding = use_integrated_encoding
        self.spatial_distortion = spatial_distortion

        self.mlp_base = MLP(
            in_dim=self.position_encoding.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_layer_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )

        self.mlp_head = MLP(
            in_dim=self.mlp_base.get_out_dim() + self.direction_encoding.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            out_activation=nn.ReLU(),
        )

        # self.field_output_density = DensityFieldHead(in_dim=self.mlp_base.get_out_dim())
        self.field_heads = nn.ModuleList(field_heads)
        for field_head in self.field_heads:
            field_head.set_in_dim(self.mlp_head.get_out_dim())  # type: ignore

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[FieldHeadNames, Tensor]:
        if self.use_integrated_encoding:
            raise NotImplementedError("IPE not implemented for NPLF")
            # gaussian_samples = ray_samples.frustums.get_gaussian_blob()
            # if self.spatial_distortion is not None:
            #     gaussian_samples = self.spatial_distortion(gaussian_samples)
            # encoded_xyz = self.position_encoding(gaussian_samples.mean, covs=gaussian_samples.cov)
        else:
            positions, directions, offsets = ray_to_np(ray_bundle=ray_bundle, compress2d=True)
            if self.spatial_distortion is not None:
                raise NotImplementedError("Spatial distortion not implemented for NPLF")
            #     positions = self.spatial_distortion(positions)
            encoded_uv = self.position_encoding(positions)
            encoded_dir = self.direction_encoding(directions)
        base_mlp_out = self.mlp_base(torch.cat((encoded_uv, encoded_dir), dim=0))
        # density = self.field_output_density(base_mlp_out)
        outputs = {}
        for field_head in self.field_heads:
            mlp_out = self.mlp_head(torch.cat([encoded_dir, encoded_uv, base_mlp_out], dim=-1))  # type: ignore
            outputs[field_head.field_head_name] = field_head(mlp_out)
        return outputs

    def forward(self, ray_bundle: RayBundle) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field of the ray directly.

        Args:
            ray_bundle: Ray origin and direction to evaluate field on.
        """

        field_outputs = self.get_outputs(ray_bundle)

        return field_outputs
