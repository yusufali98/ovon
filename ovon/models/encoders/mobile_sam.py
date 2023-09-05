# Taken from: https://github.com/facebookresearch/home-robot/blob/main/src/home_robot/home_robot/perception/detection/grounded_sam/grounded_sam_perception.py

import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import supervision as sv
import torch
import torchvision


sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent.parent.parent / "perception/Grounded-Segment-Anything/EfficientSAM")
)
from groundingdino.util.inference import Model
from MobileSAM.setup_mobile_sam import setup_model  # noqa: E402
from segment_anything import SamPredictor  # noqa: E402

from ovon.utils.utils import load_pickle


class GroundedSAMPerception:
    def __init__(
        self,
        config_file=None,
        device='cpu',
    ):
        """
        Wrapper class around groundingDINO + mobileSAM inference functions
        """

        self._config = config_file
        
        # Building GroundingDINO inference model
        self.grounding_dino_model = Model(
            model_config_path=config_file.GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=config_file.GROUNDING_DINO_CHECKPOINT_PATH,
        )

        # Extract OVON goals category list
        self.goals_cache = load_pickle(self._config.cache)
        self.ovon_goal_cats = list(self.goals_cache.keys())

        # Building MobileSAM predictor
        checkpoint = torch.load(config_file.MOBILE_SAM_CHECKPOINT_PATH)
        self.mobile_sam = setup_model()
        self.mobile_sam.load_state_dict(checkpoint, strict=True)
        self.mobile_sam.to(device=device)
        self.custom_vocabulary = self.ovon_goal_cats
        self.sam_predictor = SamPredictor(self.mobile_sam)


    def predict(
        self,
        observations,
        depth_threshold: Optional = None,
        draw_instance_predictions: bool = False,
    ):
        """
        Arguments:
            obs.rgb: image of shape (H, W, 3) (in RGB order)
            obs.depth: depth frame of shape (H, W), used for depth filtering
            depth_threshold: if specified, the depth threshold per instance

        Returns:
            obs.semantic: segmentation predictions of shape (H, W) with
             indices in [0, num_sem_categories - 1]
            obs.task_observations["semantic_frame"]: segmentation visualization
             image of shape (H, W, 3)
        """

        assert (
                'rgb' in observations
            ), "Agent not equipped with RGB sensor -> MobileSAMSemantic Sensor failed !"

        obs_rgb = observations['rgb']

        CLASSES = self.custom_vocabulary

        height, width, _ = obs_rgb.shape
        # detect objects
        detections = self.grounding_dino_model.predict_with_classes(
            image=obs_rgb,
            classes=CLASSES,
            box_threshold=self._config.BOX_THRESHOLD,
            text_threshold=self._config.TEXT_THRESHOLD,
        )
        # NMS post process
        # print(f"Before NMS: {len(detections.xyxy)} boxes")
        nms_idx = (
            torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                self._config.NMS_THRESHOLD,
            )
            .numpy()
            .tolist()
        )

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        # convert detections to masks
        detections.mask = self.segment(image=obs_rgb, xyxy=detections.xyxy)

        # if depth_threshold is not None and obs.depth is not None:
        #     detections.mask = np.array(
        #         [
        #             filter_depth(mask, obs.depth, depth_threshold)
        #             for mask in detections.mask
        #         ]
        #     )

        semantic_map, _ = self.overlay_masks(
            detections.mask, detections.class_id, (height, width)
        )

        return semantic_map.astype(int)


    def segment(self, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
        """
        Get masks for all detected bounding boxes using SAM
        Arguments:
            image: image of shape (H, W, 3)
            xyxy: bounding boxes of shape (N, 4) in (x1, y1, x2, y2) format
        Returns:
            masks: masks of shape (N, H, W)
        """
        # Prompting SAM with detected boxes

        self.sam_predictor.set_image(image)
        result_masks = []
        for box in xyxy:
            masks, scores, logits = self.sam_predictor.predict(
                box=box, multimask_output=True
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks)
    

    def overlay_masks(self, masks: np.ndarray, class_idcs: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """Overlays the masks of objects
        Determines the order of masks based on mask size
        """
        mask_sizes = [np.sum(mask) for mask in masks]
        sorted_mask_idcs = np.argsort(mask_sizes)

        semantic_mask = np.zeros(shape)
        instance_mask = -np.ones(shape)
        for i_mask in sorted_mask_idcs[::-1]:  # largest to smallest
            semantic_mask[masks[i_mask].astype(bool)] = class_idcs[i_mask]
            instance_mask[masks[i_mask].astype(bool)] = i_mask

        return semantic_mask, instance_mask