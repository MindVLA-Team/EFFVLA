# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Jinhui YE / HKUST University] in [2025].
"""
Pi0 Framework
MindVLM + Flow-matching head with KV cache sharing
"""
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from starVLA.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.Pi0_KVShareHead import get_kv_shared_action_model
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY

####################################################
# ⚠️ Warning: This framework implements KV cache sharing
# architecture and is NOT compatible with previous
# checkpoints created before 2025-03-26.
####################################################

@FRAMEWORK_REGISTRY.register("Pi0")
class Pi0(baseframework):
    """
    Pi0: Vision-Language-Action model with KV cache sharing.

    Components:
      - MindVLM for multimodal understanding
      - Flow-matching action head with KV cache sharing
      - DiT can attend to VLM's KV cache directly

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """

        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # dynamic get llm config
        vlm_name = self.config.framework.qwenvl.base_vlm
        if "Qwen" in vlm_name:
            num_vl_layers, llm_hidden_size = 36, self.qwen_vl_interface.model.config.hidden_size
        else:
            num_vl_layers, llm_hidden_size = self.qwen_vl_interface.model.config.num_hidden_layers, self.qwen_vl_interface.model.config.hidden_size
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        # Select action head type: "dit" (default) or "llm_init"
        action_head_type = getattr(self.config.framework.action_model, "action_head_type", "dit")
        # Pi0 uses the flow-matching (DiT) head with KV sharing.
        self.action_model = get_kv_shared_action_model(config=self.config)

        print(
            f'Pi0 [{action_head_type}] Action Model total parameters: '
            f'{sum(p.numel() for p in self.action_model.parameters()) / 1e6:.2f}M, '
            f'trainable parameters: '
            f'{sum(p.numel() for p in self.action_model.parameters() if p.requires_grad) / 1e6:.2f}M',
            'current',
        )

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
        # add meta query for soft-connection by xgy
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
        Forward pass with KV cache sharing.

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
        Returns:
            dict:
                action_loss (torch.Tensor): Scalar flow matching loss.
        """
        batch_images = [example["image"] for example in examples]  # [B，[PIL]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]

        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]

        # Step 1: Build VLM inputs
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

        # Step 2: VLM forward with KV cache enabled
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                use_cache=True,  # Enable KV cache
                meta_queries=self.meta_queries # only support mindvlm now, other backbone won't affect
            )

            # Get KV cache from VLM
            vlm_past_key_values = vlm_outputs.past_key_values  # List of tuples (k, v) per layer
            vlm_attention_mask = qwen_inputs['attention_mask']  # [B, vlm_seq_len]
            device = vlm_outputs.logits.device

        # Step 3: Action loss computation
        with torch.autocast("cuda", dtype=torch.float32):
            # Target actions: take last chunk_len tokens
            actions = torch.tensor(
                np.array(actions), device=device, dtype=torch.float32
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size+1):, :]

            # Prepare state tensor
            state_tensor = None
            if state is not None:
                state_tensor = torch.tensor(
                    np.array(state), device=device, dtype=torch.float32
                )
                
                B, L, D = state_tensor.shape
                if L > 1:
                    state_tensor = state_tensor[:,-1,:].unsqueeze(dim=1) # shape [B, 1, D], only the last state will be used

            # Compute action loss with KV cache sharing
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
        **kwargs: str,
    ) -> np.ndarray:
        """
        Inference: use diffusion sampling with KV cache sharing.

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - state: np.ndarray shaped [1, state_dim] (optional)
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim].
        """
        if type(examples) is not list:
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: VLM forward with KV cache
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                use_cache=True,
                meta_queries=self.meta_queries # only support mindvlm now, other backbone won't affect
            )

            vlm_past_key_values = vlm_outputs.past_key_values
            vlm_attention_mask = qwen_inputs['attention_mask']
            device = vlm_outputs.logits.device

        # Step 2: Prepare state
        state_tensor = None
        if state is not None:
            state_tensor = torch.from_numpy(np.array(state)).to(
                device, dtype=torch.float32
            )
            
            B, L, D = state_tensor.shape
            if L > 1:
                state_tensor = state_tensor[:,-1,:].unsqueeze(dim=1) # shape [B, 1, D], only the last state will be used

        # Step 3: Action prediction with KV cache sharing
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
    parser.add_argument("--config_yaml", type=str, default="/path/to/starVLA/examples/DROID/train_files/your_config.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    model = Pi0(cfg)

    # fake sample
    image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "This is a fake instruction for testing.",
        "state" : np.random.uniform(-1, 1, size=(16, 8)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action([sample])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")
