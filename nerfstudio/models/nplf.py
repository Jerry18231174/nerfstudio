"""
Implementation of Normal Plane Light Field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Type

import torch
from torch.nn import Parameter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.config_utils import to_immutable_dict
from nerfstudio.field_components.encodings import NeRFEncoding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.temporal_distortions import TemporalDistortionKind
from nerfstudio.fields.nplf_field import NPLFField
from nerfstudio.model_components.losses import MSELoss

# from nerfstudio.model_components.ray_samplers import PDFSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, RGBRenderer
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps, colors, misc


@dataclass
class NPLFConfig(ModelConfig):
    """NPLF Config"""

    _target: Type = field(default_factory=lambda: NPLFModel)
    # num_coarse_samples: int = 64
    # """Number of samples in coarse field evaluation"""
    # num_importance_samples: int = 128
    # """Number of samples in fine field evaluation"""

    # enable_temporal_distortion: bool = False
    # """Specifies whether or not to include ray warping based on time."""
    # temporal_distortion_params: Dict[str, Any] = to_immutable_dict({"kind": TemporalDistortionKind.DNERF})
    # """Parameters to instantiate temporal distortion with"""


class NPLFModel(Model):
    """NPLF model

    Args:
        config: NPLF configuration to instantiate model
    """

    config: NPLFConfig

    def __init__(
        self,
        config: NPLFConfig,
        **kwargs,
    ) -> None:
        # self.field_coarse = None
        # self.field_fine = None
        # self.temporal_distortion = None

        super().__init__(
            config=config,
            **kwargs,
        )

    def populate_modules(self):
        """Set the fields and modules"""
        super().populate_modules()

        # fields
        position_encoding = NeRFEncoding(
            in_dim=2, num_frequencies=12, min_freq_exp=0.0, max_freq_exp=10.0, include_input=True
        )
        direction_encoding = NeRFEncoding(
            in_dim=2, num_frequencies=12, min_freq_exp=0.0, max_freq_exp=10.0, include_input=True
        )

        self.field = NPLFField(
            position_encoding=position_encoding,
            direction_encoding=direction_encoding,
        )

        # samplers (disabled for LF)
        # self.sampler_uniform = UniformSampler(num_samples=self.config.num_coarse_samples)
        # self.sampler_pdf = PDFSampler(num_samples=self.config.num_importance_samples)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=colors.WHITE)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer()

        # losses
        self.rgb_loss = MSELoss()

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        # if getattr(self.config, "enable_temporal_distortion", False):
        #     params = self.config.temporal_distortion_params
        #     kind = params.pop("kind")
        #     self.temporal_distortion = kind.to_temporal_distortion(params)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        if self.field is None:
            raise ValueError("populate_fields() must be called before get_param_groups")
        param_groups["fields"] = list(self.field.parameters())
        # if self.temporal_distortion is not None:
        #     param_groups["temporal_distortion"] = list(self.temporal_distortion.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        if self.field_coarse is None or self.field_fine is None:
            raise ValueError("populate_fields() must be called before get_outputs")

        # light field:
        field_outputs = self.field.forward(ray_bundle)
        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
        )

        outputs = {
            "rgb": rgb,
        }

        return outputs

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Scaling metrics by coefficients to create the losses.
        device = outputs["rgb"].device
        image = batch["image"].to(device)

        rgb_loss = self.rgb_loss(image, outputs["rgb"])

        loss_dict = {"rgb_loss": rgb_loss}
        loss_dict = misc.scale_dict(loss_dict, self.config.loss_coefficients)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(outputs["rgb"].device)
        rgb = outputs["rgb"]
        # acc = colormaps.apply_colormap(outputs["accumulation"])
        assert self.config.collider_params is not None
        # depth = colormaps.apply_depth_colormap(
        #     outputs["depth"],
        #     accumulation=outputs["accumulation"],
        #     near_plane=self.config.collider_params["near_plane"],
        #     far_plane=self.config.collider_params["far_plane"],
        # )
        combined_rgb = torch.cat([image, rgb], dim=1)
        # combined_acc = torch.cat([acc], dim=1)
        # combined_depth = torch.cat([depth_coarse, depth_fine], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        # coarse_psnr = self.psnr(image, rgb_coarse)
        fine_psnr = self.psnr(image, rgb)
        fine_ssim = self.ssim(image, rgb)
        fine_lpips = self.lpips(image, rgb)
        assert isinstance(fine_ssim, torch.Tensor)

        metrics_dict = {
            "psnr": float(fine_psnr.item()),
            # "coarse_psnr": float(coarse_psnr),
            "fine_psnr": float(fine_psnr),
            "fine_ssim": float(fine_ssim),
            "fine_lpips": float(fine_lpips),
        }
        images_dict = {"img": combined_rgb}
        return metrics_dict, images_dict
