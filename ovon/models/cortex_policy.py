from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import clip
import numpy as np
import torch
from gym import spaces
from gym.spaces import Dict as SpaceDict
from habitat.tasks.nav.nav import EpisodicCompassSensor, EpisodicGPSSensor
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.rl.ddppo.policy.resnet import resnet18
from habitat_baselines.rl.models.rnn_state_encoder import \
    build_rnn_state_encoder
from habitat_baselines.rl.ppo import Net, NetPolicy
from habitat_baselines.utils.common import get_num_actions
from torch import nn as nn
from torchvision import transforms as T

from ovon.task.sensors import (ClipGoalSelectorSensor, ClipImageGoalSensor,
                               ClipObjectGoalSensor)

from ovon.models.encoders.cortex_encoder import (
    VisualEncoder as CortexEncoder,
)
from ovon.models.encoders.freeze_batchnorm import convert_frozen_batchnorm


@baseline_registry.register_policy
class CortexPolicy(NetPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        cortex_backbone,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        policy_config: "DictConfig" = None,
        aux_loss_config: Optional["DictConfig"] = None,
        add_clip_linear_projection: bool = False,
        late_fusion: bool = False,
        **kwargs,
    ):
        if policy_config is not None:
            discrete_actions = policy_config.action_distribution_type == "categorical"
            self.action_distribution_type = policy_config.action_distribution_type
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"

        super().__init__(
            CortexNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                cortex_backbone=cortex_backbone,
                rnn_type=rnn_type,
                discrete_actions=discrete_actions,
                add_clip_linear_projection=add_clip_linear_projection,
                late_fusion=late_fusion,
            ),
            action_space=action_space,
            policy_config=policy_config,
            aux_loss_config=aux_loss_config,
        )

    @classmethod
    def from_config(
        cls,
        config: "DictConfig",
        observation_space: spaces.Dict,
        action_space,
        **kwargs,
    ):
        # Exclude cameras for rendering from the observation space.
        ignore_names: List[str] = []
        for agent_config in config.habitat.simulator.agents.values():
            ignore_names.extend(
                agent_config.sim_sensors[k].uuid
                for k in config.habitat_baselines.video_render_views
                if k in agent_config.sim_sensors
            )
        filtered_obs = spaces.Dict(
            OrderedDict(
                ((k, v) for k, v in observation_space.items() if k not in ignore_names)
            )
        )
        try:
            late_fusion = config.habitat_baselines.rl.policy.late_fusion
        except:
            late_fusion = False
        return cls(
            observation_space=filtered_obs,
            action_space=action_space,
            hidden_size=config.habitat_baselines.rl.ppo.hidden_size,
            rnn_type=config.habitat_baselines.rl.ddppo.rnn_type,
            num_recurrent_layers=config.habitat_baselines.rl.ddppo.num_recurrent_layers,
            cortex_backbone=config.model,
            policy_config=config.habitat_baselines.rl.policy,
            aux_loss_config=config.habitat_baselines.rl.auxiliary_losses,
            add_clip_linear_projection=config.habitat_baselines.rl.policy.add_clip_linear_projection,
            late_fusion=late_fusion,
        )

    def freeze_visual_encoders(self):
        for param in self.net.visual_encoder.parameters():
            param.requires_grad_(False)
        for param in self.net.visual_fc.parameters():
            param.requires_grad_(False)

    def unfreeze_visual_encoders(self):
        for param in self.net.visual_encoder.parameters():
            param.requires_grad_(True)
        for param in self.net.visual_fc.parameters():
            param.requires_grad_(True)

    def freeze_state_encoder(self):
        for param in self.net.state_encoder.parameters():
            param.requires_grad_(False)

    def unfreeze_state_encoder(self):
        for param in self.net.state_encoder.parameters():
            param.requires_grad_(True)

    def freeze_actor(self):
        for param in self.action_distribution.parameters():
            param.requires_grad_(False)

    def unfreeze_actor(self):
        for param in self.action_distribution.parameters():
            param.requires_grad_(True)


class CortexNet(Net):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        cortex_backbone,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        discrete_actions: bool = True,
        clip_model: str = "RN50",
        add_clip_linear_projection: bool = False,
        late_fusion: bool = False,
    ):
        super().__init__()
        self.prev_action_embedding: nn.Module
        self.discrete_actions = discrete_actions
        self.add_clip_linear_projection = add_clip_linear_projection
        self.late_fusion = late_fusion
        self._n_prev_action = 32
        if discrete_actions:
            self.prev_action_embedding = nn.Embedding(
                action_space.n + 1, self._n_prev_action
            )
        else:
            num_actions = get_num_actions(action_space)
            self.prev_action_embedding = nn.Linear(num_actions, self._n_prev_action)
        rnn_input_size = self._n_prev_action  # test
        rnn_input_size_info = {"prev_action": self._n_prev_action}

        #####################################################################################################
        #################################       CORTEX Setup Starts here       ##############################
        # Create the Cortex Backbone 
        print("Initialising Cortex Encoder with following config.....")
        print(cortex_backbone)

        self.visual_encoder = CortexEncoder(
            image_size=224,
            backbone_config=cortex_backbone,
            normalize_visual_inputs=False,
            global_pool=False,
            use_cls=False,
            use_augmentations=False,
        )

        # TODO: rgb_config define the cortex backbone parameters - should be moved to main config file !
        from omegaconf import OmegaConf

        rgb_config = { 
                       "model_type" : "VisualEncoder",
                       "image_size" : 256,
                       "hidden_size" : hidden_size,
                       "use_augmentations" : False,
                       "use_augmentations_test_time" : False,
                       "freeze_batchnorm" : True,
                       "freeze_backbone" : True,
                       "global_pool" : False,
                       "use_cls" : False,
                     }
        rgb_config = OmegaConf.create(rgb_config)
        
        # Add input dimensions for the Cortex encoder visual features
        # if not self.is_blind:
        rnn_input_size += rgb_config.hidden_size
        rnn_input_size_info["visual_feats"] = rgb_config.hidden_size
        
        # freeze Cortex backbone
        if rgb_config.freeze_backbone:
            for p in self.visual_encoder.backbone.parameters():
                p.requires_grad = False
            if rgb_config.freeze_batchnorm:
                self.visual_encoder = convert_frozen_batchnorm(self.visual_encoder)
        
        print("Initialised Cortex Encoder ! ")
        #################################       CORTEX Setup Ends here       ##############################
        #####################################################################################################

        visual_fc_input = self.visual_encoder.output_shape[0]
        if ClipImageGoalSensor.cls_uuid in observation_space.spaces:
            # Goal image is a goal embedding, gets processed as a separate
            # input rather than along with the visual features
            visual_fc_input -= 1024
        self.visual_fc = nn.Sequential(
            nn.Linear(visual_fc_input, hidden_size),
            nn.ReLU(True),
        )
        print("Observation space info:")
        for k, v in observation_space.spaces.items():
            print(f"  {k}: {v}")

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]) + 1
            )
            self.obj_categories_embedding = nn.Embedding(self._n_object_categories, 32)
            rnn_input_size += 32
            rnn_input_size_info["object_goal"] = 32

        if (
            ClipObjectGoalSensor.cls_uuid in observation_space.spaces
            or ClipImageGoalSensor.cls_uuid in observation_space.spaces
        ):
            clip_embedding = 1024 if clip_model == "RN50" else 768
            print(
                f"CLIP embedding: {clip_embedding}, "
                f"Add CLIP linear: {add_clip_linear_projection}"
            )
            if self.add_clip_linear_projection:
                self.obj_categories_embedding = nn.Linear(clip_embedding, 256)
                object_goal_size = 256
            else:
                object_goal_size = clip_embedding

            if not late_fusion:
                rnn_input_size += object_goal_size
                rnn_input_size_info["clip_goal"] = object_goal_size

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[EpisodicGPSSensor.cls_uuid].shape[
                0
            ]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32
            rnn_input_size_info["gps_embedding"] = 32

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[0] == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding = nn.Linear(input_compass_dim, 32)
            rnn_input_size += 32
            rnn_input_size_info["compass_embedding"] = 32

        self._hidden_size = hidden_size

        print("RNN input size info: ")
        total = 0
        for k, v in rnn_input_size_info.items():
            print(f"  {k}: {v}")
            total += v
        if total - rnn_input_size != 0:
            print(f"  UNACCOUNTED: {total - rnn_input_size}")
        total_str = f"  Total RNN input size: {total}"
        print("  " + "-" * (len(total_str) - 2))
        print(total_str)

        self.state_encoder = build_rnn_state_encoder(
            rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    @property
    def perception_embedding_size(self):
        return self._hidden_size

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        x = []
        aux_loss_state = {}
        clip_image_goal = None
        object_goal = None
        if not self.is_blind:
            # We CANNOT use observations.get() here because
            # self.visual_encoder(observations) is an expensive operation. Therefore,
            # we need `# noqa: SIM401`
            if (  # noqa: SIM401
                PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY in observations
            ): 
                visual_feats = observations[
                    PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
                ]
            else:
                visual_feats = self.visual_encoder(observations)

            if ClipImageGoalSensor.cls_uuid in observations:
                clip_image_goal = visual_feats[:, :1024]
                visual_feats = self.visual_fc(visual_feats[:, 1024:])
            else:
                # print("-----------------------------------> Generating final visual FC....")
                # print("-----------------------------------> visual feats shape: ", visual_feats.shape)
                # print("-----------------------------------> fc layer: ", self.visual_fc)
                visual_feats = self.visual_fc(visual_feats)

            # print("Final visual feats shape: ", visual_feats.shape)
    
            aux_loss_state["perception_embed"] = visual_feats
            x.append(visual_feats)

        # print("x right now: ", len(x), x[0].shape)

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))

        if ClipObjectGoalSensor.cls_uuid in observations and not self.late_fusion:
            object_goal = observations[ClipObjectGoalSensor.cls_uuid]
            if self.add_clip_linear_projection:
                object_goal = self.obj_categories_embedding(object_goal)
            x.append(object_goal)
        
        # print("x after objectgoal: ")
        # for i,ele in enumerate(x):
        #     print(ele.shape)

        if ClipImageGoalSensor.cls_uuid in observations and not self.late_fusion:
            assert clip_image_goal is not None
            if self.add_clip_linear_projection:
                clip_image_goal = self.obj_categories_embedding(clip_image_goal)
            x.append(clip_image_goal)

        if ClipGoalSelectorSensor.cls_uuid in observations:
            assert (
                ClipObjectGoalSensor.cls_uuid in observations
                and ClipImageGoalSensor.cls_uuid in observations
            ), "Must have both object and image goals to use goal selector."
            image_goal_embedding = x.pop()
            object_goal = x.pop()
            assert image_goal_embedding.shape == object_goal.shape
            clip_goal = torch.where(
                observations[ClipGoalSelectorSensor.cls_uuid].bool(),
                image_goal_embedding,
                object_goal,
            )
            x.append(clip_goal)

        if EpisodicCompassSensor.cls_uuid in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(self.compass_embedding(compass_observations.squeeze(dim=1)))
        
        # print("x after compass: ")
        # for i,ele in enumerate(x):
        #     print(ele.shape)

        if EpisodicGPSSensor.cls_uuid in observations:
            x.append(self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid]))

        # print("x after gps: ")
        # for i,ele in enumerate(x):
        #     print(ele.shape)

        prev_actions = prev_actions.squeeze(-1)
        start_token = torch.zeros_like(prev_actions)
        # The mask means the previous action will be zero, an extra dummy action
        prev_actions = self.prev_action_embedding(
            torch.where(masks.view(-1), prev_actions + 1, start_token)
        )

        x.append(prev_actions)

        # print("x after actions: ")
        # for i,ele in enumerate(x):
        #     print(ele.shape)

        out = torch.cat(x, dim=1)

        # print("x after concat: ", out.shape)

        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )
        aux_loss_state["rnn_output"] = out

        # print("out from rnn: ", out.shape, rnn_hidden_states.shape)

        if self.late_fusion:
            object_goal = observations[ClipObjectGoalSensor.cls_uuid]
            out = (out + visual_feats) * object_goal

        return out, rnn_hidden_states, aux_loss_state

