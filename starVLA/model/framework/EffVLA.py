# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
EffVLA Framework
MindVLM + KV-shared regression action head (OFT-style)

Architecturally identical to Pi0 except the action head uses L1 regression
loss (like QwenOFT) instead of flow-matching loss.  No denoising loop at
inference time – a single forward pass produces the predicted actions.
"""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from starVLA.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

# Lazy import (avoids mandatory dependency at module load time)
def _get_llm_init_oft_action_model(config, llm_model):
    from starVLA.model.modules.action_model.EffVLA_VLMInitHead import get_llm_init_oft_action_model
    return get_llm_init_oft_action_model(config=config, llm_model=llm_model)


@FRAMEWORK_REGISTRY.register("EffVLA")
class EffVLA(baseframework):
    """
    EffVLA: Vision-Language-Action model with KV cache sharing and L1 regression.

    Components:
      - MindVLM for multimodal understanding
      - KV-shared transformer action head (regression, no diffusion)
      - Action experts attend to VLM's KV cache directly

    Focus: Predict future continuous actions via L1 regression conditioned on
           images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Determine VLM hidden dimensions
        vlm_name = self.config.framework.qwenvl.base_vlm
        if "Qwen" in vlm_name:
            num_vl_layers  = 36
            llm_hidden_size = self.qwen_vl_interface.model.config.hidden_size
        else:
            num_vl_layers   = self.qwen_vl_interface.model.config.num_hidden_layers
            llm_hidden_size = self.qwen_vl_interface.model.config.hidden_size

        self.config.framework.qwenvl.vl_hidden_dim  = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers  = num_vl_layers

        # EffVLA uses the VLM-initialized transformer-block head, which deep-copies the
        # backbone's last decoder layers into the action head:
        #   llm_init         → decoder layers copied with weights (the EffVLA recipe)
        #   llm_init_random  → same architecture, weights reinitialized (ablation)
        action_head_type = getattr(self.config.framework.action_model, "action_head_type", "llm_init")
        self.action_model = _get_llm_init_oft_action_model(
            config=self.config,
            llm_model=self.qwen_vl_interface.model,
        )

        print(
            f'EffVLA [{action_head_type}] Action Model total parameters: '
            f'{sum(p.numel() for p in self.action_model.parameters()) / 1e6:.2f}M, '
            f'trainable parameters: '
            f'{sum(p.numel() for p in self.action_model.parameters() if p.requires_grad) / 1e6:.2f}M',
            'current',
        )

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size   = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # Optional meta queries for soft-connection (only effective with MindVLM)
        num_queries = int(self.config.framework.qwenvl.get("num_queries", 0))
        print("meta queries' number:", num_queries)
        if num_queries > 0:
            self.meta_queries = nn.Parameter(
                torch.randn(num_queries, llm_hidden_size)
            )
        else:
            self.meta_queries = None

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Training forward pass with KV cache sharing.

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang:  str instruction
                - action: np.ndarray shaped [T, action_dim]
                - state:  np.ndarray shaped [L, state_dim] (optional)
        Returns:
            dict:
                action_loss (torch.Tensor): Scalar L1 regression loss.
        """
        batch_images  = [example["image"]  for example in examples]
        instructions  = [example["lang"]   for example in examples]
        actions       = [example["action"] for example in examples]
        state         = [example["state"]  for example in examples] if "state" in examples[0] else None

        # Step 1: Build VLM inputs
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )

        # Step 2: VLM forward with KV cache enabled
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                use_cache=True,
                meta_queries=self.meta_queries,  # only effective for MindVLM
            )

            vlm_past_key_values = vlm_outputs.past_key_values
            vlm_attention_mask  = qwen_inputs['attention_mask']
            device = vlm_outputs.logits.device

        # Step 3: Action loss computation (L1 regression)
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=device, dtype=torch.float32
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size + 1):, :]

            state_tensor = None
            if state is not None:
                state_tensor = torch.tensor(
                    np.array(state), device=device, dtype=torch.float32
                )
                B, L, D = state_tensor.shape
                if L > 1:
                    state_tensor = state_tensor[:, -1, :].unsqueeze(1)  # [B, 1, D]

            action_loss = self.action_model(
                vlm_past_key_values=vlm_past_key_values,
                vlm_attention_mask=vlm_attention_mask,
                actions=actions_target,
                state=state_tensor,
            )

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> dict:
        """
        Inference: single forward pass, no denoising loop.

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang:  str instruction
                - state: np.ndarray shaped [L, state_dim] (optional)
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim].
        """
        if not isinstance(examples, list):
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state        = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: VLM forward with KV cache
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                use_cache=True,
                meta_queries=self.meta_queries,
            )

            vlm_past_key_values = vlm_outputs.past_key_values
            vlm_attention_mask  = qwen_inputs['attention_mask']
            device = vlm_outputs.logits.device

        # Step 2: Prepare state tensor
        state_tensor = None
        if state is not None:
            state_tensor = torch.from_numpy(np.array(state)).to(
                device, dtype=torch.float32
            )
            B, L, D = state_tensor.shape
            if L > 1:
                state_tensor = state_tensor[:, -1, :].unsqueeze(1)  # [B, 1, D]

        # Step 3: Single-pass action prediction (no denoising loop)
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                vlm_past_key_values=vlm_past_key_values,
                vlm_attention_mask=vlm_attention_mask,
                state=state_tensor,
            )

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="/path/to/starVLA/examples/DROID/train_files/your_config.yaml",
        help="Path to YAML config",
    )
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)

    model  = EffVLA(cfg)

    image  = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image":  [image, image],
        "lang":   "This is a fake instruction for testing.",
        "state":  np.random.uniform(-1, 1, size=(16, 8)).astype(np.float16),
    }

    batch  = [sample, sample]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    forward_output = model(batch)
    print(f"Action Loss: {forward_output['action_loss'].item()}")

    predict_output = model.predict_action([sample])
    print(f"Predicted Actions shape: {predict_output['normalized_actions'].shape}")
