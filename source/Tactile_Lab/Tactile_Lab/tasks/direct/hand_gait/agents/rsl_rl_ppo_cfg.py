# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationStudentTeacherRecurrentCfg,
    RslRlDistillationStudentTeacherCNNCfg,
    RslRlExtrinsicsDistillationAlgorithmCfg,
    RslRlExtrinsicsDistillationStudentTeacherCfg,
    RslRlExtrinsicsMultiModalDistillationStudentTeacherCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlExtrinsicsDistillationRunnerCfg,
    RslRlExtrinsicsMultiModalDistillationRunnerCfg,
    RslRlPpoAlgorithmCfg,
    
    RslRlPpoActorCriticCfg,
    RslRlActorCriticCNNCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlActorCriticExtrinsicsCfg,
    RslRlDistillationStudentTeacherCNNRNNCfg,
    RslRlDistillationStudentTeacherCfg,
    
)

HISTORY_LEN = 15
############For stage 1 extrinsics oracle teacher Training#############


@configclass
class HandGaitExtrinsicsPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 13
    num_steps_per_env = 16
    max_iterations = 100000
    save_interval = 500
    experiment_name = "hand_gait"
    run_name = "extrinsics_oracle_teacher"

    obs_groups = {
        "policy": ["priv", "short_proprio"],  # required by runner
        "priv": ["priv"],          # privileged input to μ
        "obs_hist": ["long_proprio"],          # privileged input to μ
        "obs": ["short_proprio"],      # shared task input to π
        "critic": ["priv", "short_proprio"],  # critic can still use full info
    }

    policy = RslRlActorCriticExtrinsicsCfg(
        extrinsics_output_dims=9,   # choose latent size
        extrinsics_hidden_dims=[256,128],
        init_noise_std=1.0,
        actor_obs_normalization=True,
        priv_obs_normalization=False,
        critic_obs_normalization=True,
        actor_hidden_dims=[1024, 512, 256, 128],
        critic_hidden_dims=[1024, 512, 256, 128],
        activation="elu",
        last_activation="tanh",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )


############For oracle teacher Training#############


@configclass
class HandGaitPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 3
    num_steps_per_env = 16
    max_iterations = 100000
    save_interval = 500
    experiment_name = "hand_gait"
    run_name = "oracle_teacher"

    obs_groups = {
        "policy": ["priv", "short_proprio"],
        "critic": ["priv", "short_proprio"],
    }

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[1024, 512, 256, 128],
        critic_hidden_dims=[1024, 512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.016,
        max_grad_norm=1.0,
    )

############For tactile image teacher Training#############


@configclass
class HandGaitPPORunnerCNNCfg(RslRlOnPolicyRunnerCfg):
    seed = 3
    num_steps_per_env = 16
    max_iterations = 50000
    save_interval = 50
    experiment_name = "hand_gait"
    run_name = "tactile_teacher"
    # If we don't specify obs_groups, the runner will automatically infer them from the environment observations. which only get the 'policy' obs group.
    obs_groups = {
        "policy": [ "tactile_flow", "depth_map", "short_proprio",],
        "critic": [ "tactile_flow", "depth_map", "short_proprio",],
    }

    policy = RslRlActorCriticCNNCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[1024, 512, 256, 128],
        critic_hidden_dims=[1024, 512, 256, 128],
        actor_cnn_cfg=RslRlActorCriticCNNCfg.CNNCfg(
            output_channels=[16, 32, 32],
            kernel_size=[2, 2, 2],
            activation="elu",
            norm=["batch", "batch", "batch"],
            max_pool=[True, False, False],
            global_pool="avg",
        ),
        critic_cnn_cfg=RslRlActorCriticCNNCfg.CNNCfg(
            output_channels=[16, 32, 32],
            kernel_size=[2, 2, 2],
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

############For Stage 2 Extrinsics Proprio Distillation Training#############
@configclass
class HandGaitExtrinsicsProprioDistillationRunnerCfg(RslRlExtrinsicsDistillationRunnerCfg):

    seed = 3
    num_steps_per_env = 16
    max_iterations = 20000
    save_interval = 50

    experiment_name = "hand_gait"
    run_name = "extrinsics_proprio_distill"

    obs_groups = {
        "policy": ["priv", "short_proprio"],  # required by runner
        "obs": ["short_proprio"],
        "priv": ["priv"],
        "obs_hist": ["long_proprio"],
    }

    policy = RslRlExtrinsicsDistillationStudentTeacherCfg(
        extrinsics_output_dim=9,
        history_len=HISTORY_LEN,
        init_noise_std=1.0,
        actor_obs_normalization=True,
        priv_obs_normalization=False,
        student_1d_obs_normalization=True,
        actor_hidden_dims=[1024, 512, 256, 128],
        teacher_extrinsics_hidden_dims=[256, 128],
        activation="elu",
        # teacher_tanh_extrinsics=True,
    )

    algorithm = RslRlExtrinsicsDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1e-3,
        gradient_length=15,
        max_grad_norm=1.0,
    )

############For Stage 2 Extrinsics Proprio and dense tactile Distillation Training#############
@configclass
class HandGaitExtrinsicsProprioLowDimContactDistillationRunnerCfg(HandGaitExtrinsicsProprioDistillationRunnerCfg):


    run_name = "extrinsics_proprio_low_dim_contact_distill"

    obs_groups = {
        "policy": ["priv", "short_proprio"],  # required by runner
        "obs": ["short_proprio"],
        "priv": ["priv"],
        "obs_hist": ["long_proprio_and_contact_obs"],
    }


############For Stage 2 Multimodel Extrinsics Distillation Training#############
@configclass
class HandGaitExtrinsicsMultiModalDistillationRunnerCfg(RslRlExtrinsicsMultiModalDistillationRunnerCfg):

    seed: int = 3
    num_steps_per_env: int = 16
    max_iterations: int = 20000
    save_interval: int = 50

    experiment_name: str = "hand_gait"
    run_name: str = "extrinsics_multimodal_distill"

    obs_groups = {
        "policy": ["priv", "short_proprio"],
        "obs": ["short_proprio"],
        "priv": ["priv"],
        "obs_hist": ["long_proprio", "long_tactile_flow", "long_depth_map"],
        # "obs_hist": ["long_tactile_flow", "long_depth_map"],
    }

    policy = RslRlExtrinsicsMultiModalDistillationStudentTeacherCfg(
        extrinsics_output_dim=9,
        history_len=HISTORY_LEN,
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=True,
        priv_obs_normalization=False,
        student_1d_obs_normalization=True,
        actor_hidden_dims=[1024, 512, 256, 128],
        teacher_extrinsics_hidden_dims=[256, 128],
        activation="elu",
        student_cnn_cfg=RslRlActorCriticCNNCfg.CNNCfg(
            output_channels=[16, 32],
            kernel_size=[3, 3],
            activation="elu",
            norm=["batch", "batch", "batch"],
            max_pool=[True, False, False],
            global_pool="avg",
        ),
    )

    algorithm = RslRlExtrinsicsDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1e-3,
        gradient_length=15,
        max_grad_norm=1.0,
    )

@configclass
class HandGaitExtrinsicsMultiModalLowDimContactDistillationRunnerCfg(HandGaitExtrinsicsMultiModalDistillationRunnerCfg):


    run_name = "extrinsics_multimodal_low_dim_contact_distill"

    obs_groups = {
        "policy": ["priv", "short_proprio"],
        "obs": ["short_proprio"],
        "priv": ["priv"],
        "obs_hist": ["long_tactile_flow", "long_depth_map", "long_proprio_and_contact_obs"],
    }


############For Tactile Distillation Training#############
@configclass
class HandGaitDistillationRunnerCfg(HandGaitPPORunnerCfg):
    class_name = "DistillationRunner"
    seed = 42
    max_iterations = 100000
    save_interval = 500
    run_name = "oracle_distillation"

    obs_groups = {
        "policy": ["proprio", "priv",],
        "teacher": ["proprio", "priv",],
    }

    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        gradient_length=5,
        learning_rate=1e-4,  # 1e-3
        loss_type="mse",
    )
    policy = RslRlDistillationStudentTeacherCfg(
        student_hidden_dims=[1024, 512, 256, 128],
        teacher_hidden_dims=[1024, 512, 256, 128],  # same as oracle teacher
        student_obs_normalization=False,
        teacher_obs_normalization=True,
        activation="elu",
        init_noise_std=1.0,  # 0.1
    )


@configclass
class HandGaitDistillationRNNRunnerCfg(HandGaitPPORunnerCfg):
    class_name = "DistillationRunner"
    seed = 42
    max_iterations = 100000
    save_interval = 500
    run_name = "tactile_distillation_RNN"

    obs_groups = {
        # "policy": ["image", "motion", "feature"],
        # "teacher": ["proprio", "priv",],
        "policy": ["proprio", "priv",],
        "teacher": ["proprio", "priv",],
    }

    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        gradient_length=5,
        learning_rate=1e-4,  # 1e-3
        loss_type="mse",
    )
    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        student_hidden_dims=[1024, 512, 256, 128],
        teacher_hidden_dims=[1024, 512, 256, 128],  # same as oracle teacher
        student_obs_normalization=False,
        teacher_obs_normalization=True,
        activation="elu",
        init_noise_std=1.0,  # 0.1
        class_name="StudentTeacherRecurrent",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=3,
        teacher_recurrent=False,
    )


@configclass
class HandGaitDistillationCNNRNNRunnerCfg(HandGaitPPORunnerCfg):
    class_name = "DistillationRunner"
    seed = 42
    max_iterations = 100000
    save_interval = 500
    run_name = "tactile_distillation_CNN_RNN"

    obs_groups = {
        "policy": ["proprio", "tactile_flow", "depth_map"],
        "teacher": ["proprio", "priv",],
    }
    
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        gradient_length=5,
        learning_rate=1e-4,  # 1e-3
        loss_type="mse",
    )
    policy = RslRlDistillationStudentTeacherCNNRNNCfg(
        student_hidden_dims=[1024, 512, 256, 128],
        teacher_hidden_dims=[1024, 512, 256, 128],  # same as oracle teacher
        student_obs_normalization=False,
        teacher_obs_normalization=True,
        activation="elu",
        init_noise_std=1.0,  # 0.1
        student_cnn_cfg=RslRlActorCriticCNNCfg.CNNCfg(
            output_channels=[32, 64, 64],
            kernel_size=[3, 3, 3],
            activation="elu",
            norm=["batch", "batch", "batch"],
            max_pool=[True, False, False],
            global_pool="avg",
        ),
        class_name="StudentTeacherCNNRNN",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=3,
        teacher_recurrent=False,
    )
