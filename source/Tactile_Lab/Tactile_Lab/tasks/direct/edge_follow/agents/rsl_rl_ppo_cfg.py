# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationStudentTeacherRecurrentCfg,
    RslRlDistillationStudentTeacherCNNCfg,
    RslRlOnPolicyRunnerCfg,

    RslRlPpoAlgorithmCfg,
    
    RslRlPpoActorCriticCfg,
    RslRlActorCriticCNNCfg,
    RslRlPpoActorCriticRecurrentCfg,
    
)

############For oracle teacher Training#############


@configclass
class EdgeFollowPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 3
    num_steps_per_env = 16
    max_iterations = 5000
    save_interval = 500
    experiment_name = "edge_follow"
    run_name = "oracle_teacher"

    obs_groups = {
        "policy": ["oracle"],
        "critic": ["oracle"],
    }
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 128, 128],
        critic_hidden_dims=[256, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

############For tactile image teacher Training#############


@configclass
class EdgeFollowPPORunnerCNNCfg(RslRlOnPolicyRunnerCfg):
    seed = 3
    num_steps_per_env = 16
    max_iterations = 5000
    save_interval = 50
    experiment_name = "edge_follow"
    run_name = "tactile_teacher"
    # If we don't specify obs_groups, the runner will automatically infer them from the environment observations. which only get the 'policy' obs group.
    obs_groups = {
        "policy": ["image", "dummy_1d"],
        "critic": ["image", "dummy_1d"],
    }

    policy = RslRlActorCriticCNNCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_cnn_cfg=RslRlActorCriticCNNCfg.CNNCfg(
            output_channels=[32, 64, 64],
            kernel_size=[7, 5, 3],
            activation="elu",
            norm=["batch", "batch", "batch"],
            max_pool=[True, False, False],
            global_pool="avg",
        ),
        critic_cnn_cfg=RslRlActorCriticCNNCfg.CNNCfg(
            output_channels=[32, 64, 64],
            kernel_size=[7, 5, 3],
            activation="elu",
            norm=["batch", "batch", "batch"],
            max_pool=[True, False, False],
            global_pool="avg",
        ),
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

############For Tactile Distillation Training#############

@configclass
class EdgeFollowDistillationCNNRunnerCfg(EdgeFollowPPORunnerCfg):
    class_name = "DistillationRunner"
    seed = 42
    max_iterations = 10000
    save_interval = 100
    run_name = "tactile_distillation"

    obs_groups = {
        "policy": ["image", "dummy_1d"],
        "teacher": ["oracle",],
    }
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        gradient_length=5,
        learning_rate=1e-4,
        loss_type="mse",
    )
    policy = RslRlDistillationStudentTeacherCNNCfg(
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[256, 128, 128],  # same as oracle teacher
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        activation="elu",
        init_noise_std=1.0,
        student_cnn_cfg=RslRlActorCriticCNNCfg.CNNCfg(
            output_channels=[32, 64, 64],
            kernel_size=[7, 5, 3],
            activation="elu",
            norm=["batch", "batch", "batch"],
            max_pool=[True, False, False],
            global_pool="avg",
        ),
    )


    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 1500

#########################
# Student Fine Tuning ###
#########################

############For Tactile Finetuning#############
@configclass
class EdgeFollowStudentFinetunePPORunnerCNNCfg(EdgeFollowPPORunnerCfg):
    seed = 3
    obs_groups = {
        "policy": ["image", "dummy_1d"],
        "critic": ["image", "dummy_1d"],
    }
    policy = RslRlActorCriticCNNCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        actor_cnn_cfg=RslRlActorCriticCNNCfg.CNNCfg(
            output_channels=[32, 64, 64],
            kernel_size=[7, 5, 3],
            activation="elu",
            norm=["batch", "batch", "batch"],
            max_pool=[True, False, False],
            global_pool="avg",
        ),
        critic_cnn_cfg=RslRlActorCriticCNNCfg.CNNCfg(
            output_channels=[32, 64, 64],
            kernel_size=[7, 5, 3],
            activation="elu",
            norm=["batch", "batch", "batch"],
            max_pool=[True, False, False],
            global_pool="avg",
        ),
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=8,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 4000
        self.run_name = "tactile_student_finetune"