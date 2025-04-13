# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train a PPO agent using JAX on the specified environment with live MuJoCo visualization."""

from datetime import datetime
import functools
import json
import os
import time
import warnings

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from etils import epath
from flax.training import orbax_utils
import jax
import jax.numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco
import mujoco.viewer
import numpy as np
from orbax import checkpoint as ocp
from tensorboardX import SummaryWriter
import wandb

import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params

# Set up some XLA and MuJoCo environment variables
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "metal"

logging.set_verbosity(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

# Define flags for command line usage.
_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "Go1JoystickFlatTerrain",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_VISION = flags.DEFINE_boolean("vision", False, "Use vision input")
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "Path to load checkpoint from"
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", True, "If true, only play with the model and do not train"
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb", False, "Use Weights & Biases for logging (ignored in play-only mode)"
)
_USE_TB = flags.DEFINE_boolean(
    "use_tb", False, "Use TensorBoard for logging (ignored in play-only mode)"
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer("num_timesteps", 1_000_000, "Number of timesteps")
_NUM_EVALS = flags.DEFINE_integer("num_evals", 5, "Number of evaluations")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 0.1, "Reward scaling")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Episode length")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 10, "Unroll length")
_NUM_MINIBATCHES = flags.DEFINE_integer("num_minibatches", 8, "Number of minibatches")
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer("num_updates_per_batch", 8, "Number of updates per batch")
_DISCOUNTING = flags.DEFINE_float("discounting", 0.97, "Discounting")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 5e-4, "Learning rate")
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", 5e-3, "Entropy cost")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 1024, "Number of environments")
_NUM_EVAL_ENVS = flags.DEFINE_integer("num_eval_envs", 128, "Number of evaluation environments")
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Batch size")
_MAX_GRAD_NORM = flags.DEFINE_float("max_grad_norm", 1.0, "Max grad norm")
_CLIPPING_EPSILON = flags.DEFINE_float("clipping_epsilon", 0.2, "Clipping epsilon for PPO")
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes", [64, 64, 64], "Policy hidden layer sizes"
)
_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "value_hidden_layer_sizes", [64, 64, 64], "Value hidden layer sizes"
)
_POLICY_OBS_KEY = flags.DEFINE_string("policy_obs_key", "state", "Policy obs key")
_VALUE_OBS_KEY = flags.DEFINE_string("value_obs_key", "state", "Value obs key")


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
  if env_name in mujoco_playground.manipulation._envs:
    if _VISION.value:
      return manipulation_params.brax_vision_ppo_config(env_name)
    return manipulation_params.brax_ppo_config(env_name)
  elif env_name in mujoco_playground.locomotion._envs:
    if _VISION.value:
      return locomotion_params.brax_vision_ppo_config(env_name)
    return locomotion_params.brax_ppo_config(env_name)
  elif env_name in mujoco_playground.dm_control_suite._envs:
    if _VISION.value:
      return dm_control_suite_params.brax_vision_ppo_config(env_name)
    return dm_control_suite_params.brax_ppo_config(env_name)
  raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")

# --------------------------------------------------------------------------------
# joystick-related global variables for manual control
# these globals represent the desired speed and turning commands.
forward_vel = 0.0     # forward/backward command in [-1, 1] (positive = move forward, negative = move backward)
lateral_vel = 0.0     # lateral command in [-1, 1] (positive = move right, negative = move left)
turn_rate = 0.0       # turning command in [-1, 1] (determines the yaw rate)
command_scale = 0.1   # the incremental change when a key is pressed

# function: key_callback
# this function is called whenever a key is pressed in the mujoco viewer.
# it adjusts the global command variables based on which key is pressed.
def key_callback(keycode):
    """
    keyboard callback to adjust forward, lateral, and turn commands.
    triggered each time a key is pressed in the viewer.
    """
    global forward_vel, lateral_vel, turn_rate, command_scale

    c = chr(keycode).lower()  # convert keycode to a lowercase character
    if c == 'w':
        forward_vel += command_scale  # increase forward command
        print(f"forward velocity = {forward_vel:.2f}")
    elif c == 's':
        forward_vel -= command_scale  # decrease forward command (i.e., move backward)
        print(f"forward velocity = {forward_vel:.2f}")
    elif c == 'd':
        lateral_vel -= command_scale  # decrease lateral command (here 'd' is mapped to leftward movement)
        print(f"lateral velocity = {lateral_vel:.2f}")
    elif c == 'a':
        lateral_vel += command_scale  # increase lateral command (here 'a' is mapped to rightward movement)
        print(f"lateral velocity = {lateral_vel:.2f}")
    elif c == 'e':
        turn_rate -= command_scale    # decrease turning command (turn one way)
        print(f"turn rate = {turn_rate:.2f}")
    elif c == 'q':
        turn_rate += command_scale    # increase turning command (turn the opposite way)
        print(f"turn rate = {turn_rate:.2f}")
    elif c == ' ':
        # reset all commands to zero when space is pressed, stopping all movement
        forward_vel = 0.0
        lateral_vel = 0.0
        turn_rate = 0.0
        print("reset all commands = 0.")
    elif c == 'z':
        # decrease the command scale to make smaller adjustments
        command_scale = max(command_scale / 2.0, 0.0001)
        print(f"command scale = {command_scale:.3f}")
    elif c == 'x':
        # increase the command scale to allow larger adjustments
        command_scale = min(command_scale * 2.0, 10.0)
        print(f"command scale = {command_scale:.3f}")

    # ensure that the command values remain within the range [-1, 1]
    forward_vel = max(min(forward_vel, 1.0), -1.0)
    lateral_vel = max(min(lateral_vel, 1.0), -1.0)
    turn_rate = max(min(turn_rate, 1.0), -1.0)
# --------------------------------------------------------------------------------

def main(argv):
  del argv  # ignore any command-line arguments not used here

  # 1) load the default environment configuration and ppo parameters based on the env name flag
  env_cfg = registry.get_default_config(_ENV_NAME.value)
  ppo_params = get_rl_config(_ENV_NAME.value)

  # 2) override ppo parameters with command line flag values, if provided
  # this ensures the runtime configuration matches the user's requirements
  if _NUM_TIMESTEPS.present:
    ppo_params.num_timesteps = _NUM_TIMESTEPS.value
  if _PLAY_ONLY.present:
    ppo_params.num_timesteps = 0
  if _NUM_EVALS.present:
    ppo_params.num_evals = _NUM_EVALS.value
  if _REWARD_SCALING.present:
    ppo_params.reward_scaling = _REWARD_SCALING.value
  if _EPISODE_LENGTH.present:
    ppo_params.episode_length = _EPISODE_LENGTH.value
  if _NORMALIZE_OBSERVATIONS.present:
    ppo_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
  if _ACTION_REPEAT.present:
    ppo_params.action_repeat = _ACTION_REPEAT.value
  if _UNROLL_LENGTH.present:
    ppo_params.unroll_length = _UNROLL_LENGTH.value
  if _NUM_MINIBATCHES.present:
    ppo_params.num_minibatches = _NUM_MINIBATCHES.value
  if _NUM_UPDATES_PER_BATCH.present:
    ppo_params.num_updates_per_batch = _NUM_UPDATES_PER_BATCH.value
  if _DISCOUNTING.present:
    ppo_params.discounting = _DISCOUNTING.value
  if _LEARNING_RATE.present:
    ppo_params.learning_rate = _LEARNING_RATE.value
  if _ENTROPY_COST.present:
    ppo_params.entropy_cost = _ENTROPY_COST.value
  if _NUM_ENVS.present:
    ppo_params.num_envs = _NUM_ENVS.value
  if _NUM_EVAL_ENVS.present:
    ppo_params.num_eval_envs = _NUM_EVAL_ENVS.value
  if _BATCH_SIZE.present:
    ppo_params.batch_size = _BATCH_SIZE.value
  if _MAX_GRAD_NORM.present:
    ppo_params.max_grad_norm = _MAX_GRAD_NORM.value
  if _CLIPPING_EPSILON.present:
    ppo_params.clipping_epsilon = _CLIPPING_EPSILON.value
  if _POLICY_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.policy_hidden_layer_sizes = list(
        map(int, _POLICY_HIDDEN_LAYER_SIZES.value)
    )
  if _VALUE_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.value_hidden_layer_sizes = list(
        map(int, _VALUE_HIDDEN_LAYER_SIZES.value)
    )
  if _POLICY_OBS_KEY.present:
    ppo_params.network_factory.policy_obs_key = _POLICY_OBS_KEY.value
  if _VALUE_OBS_KEY.present:
    ppo_params.network_factory.value_obs_key = _VALUE_OBS_KEY.value

  # 3) if the vision flag is set, update the environment config to expect vision inputs
  if _VISION.value:
    env_cfg.vision = True
    env_cfg.vision_config.render_batch_size = ppo_params.num_envs

  # 4) load the environment using the registry and the configuration we defined
  env = registry.load(_ENV_NAME.value, config=env_cfg)

  # 5) if a checkpoint path is provided, load the latest checkpoint for inference (rollout)
  if _LOAD_CHECKPOINT_PATH.value is not None:
    ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
    if ckpt_path.is_dir():
      latest_ckpts = list(ckpt_path.glob("*"))
      latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
      latest_ckpts.sort(key=lambda x: int(x.name))
      if latest_ckpts:
        latest_ckpt = latest_ckpts[-1]
        restore_checkpoint_path = latest_ckpt
        print(f"restoring from: {restore_checkpoint_path}")
      else:
        print(f"no checkpoint directories found in {ckpt_path}")
        restore_checkpoint_path = ckpt_path
    else:
      restore_checkpoint_path = ckpt_path
      print(f"restoring from checkpoint: {restore_checkpoint_path}")
  else:
    print("no checkpoint path provided, not restoring from checkpoint")
    restore_checkpoint_path = None

  # write the current environment configuration to a json file in the checkpoint directory
  with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4)

  # function: policy_params_fn
  # this helper function saves the policy parameters at a given training step.
  def policy_params_fn(current_step, make_policy, params):
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    path = ckpt_path / f"{current_step}"
    orbax_checkpointer.save(path, params, force=True, save_args=save_args)

  # 7) prepare training parameters and determine which network factory to use for ppo
  training_params = dict(ppo_params)
  if "network_factory" in training_params:
    del training_params["network_factory"]

  # choose the appropriate network function (with vision or without) based on the flag
  network_fn = (
      ppo_networks_vision.make_ppo_networks_vision
      if _VISION.value
      else ppo_networks.make_ppo_networks
  )
  if hasattr(ppo_params, "network_factory"):
    network_factory = functools.partial(network_fn, **ppo_params.network_factory)
  else:
    network_factory = network_fn

  # add domain randomization function if the flag is enabled
  if _DOMAIN_RANDOMIZATION.value:
    training_params["randomization_fn"] = registry.get_domain_randomizer(_ENV_NAME.value)

  # if vision is used, wrap the environment appropriately
  if _VISION.value:
    env = wrapper.wrap_for_brax_training(
        env,
        vision=True,
        num_vision_envs=env_cfg.vision_config.render_batch_size,
        episode_length=ppo_params.episode_length,
        action_repeat=ppo_params.action_repeat,
        randomization_fn=training_params.get("randomization_fn"),
    )

  # determine the number of evaluation environments to use
  num_eval_envs = ppo_params.num_envs if _VISION.value else ppo_params.get("num_eval_envs", 128)
  if "num_eval_envs" in training_params:
    del training_params["num_eval_envs"]

  # create the training (or inference) function by partially applying ppo.train with our parameters.
  # if play_only is true, training is skipped and we use the loaded policy.
  train_fn = functools.partial(
      ppo.train,
      **training_params,
      network_factory=network_factory,
      policy_params_fn=policy_params_fn,
      seed=_SEED.value,
      restore_checkpoint_path=restore_checkpoint_path,
      wrap_env_fn=None if _VISION.value else wrapper.wrap_for_brax_training,
      num_eval_envs=num_eval_envs,
  )

  times = [time.monotonic()]  # record start time for measuring jax compilation and training time

  # function: progress
  # logs progress during training/inference and optionally reports to wandb.
  def progress(num_steps, metrics):
    times.append(time.monotonic())
    if _USE_WANDB.value and not _PLAY_ONLY.value:
      wandb.log(metrics, step=num_steps)
    print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f}")

  # if not using vision, create a separate evaluation environment; otherwise, use the main one.
  eval_env = None if _VISION.value else registry.load(_ENV_NAME.value, config=env_cfg)

  # 8) run training or, in play-only mode, just load the checkpoint and prepare the inference function.
  make_inference_fn, params, _ = train_fn(
      environment=env,
      progress_fn=progress,
      eval_env=eval_env,
  )

  print("policy is loaded.")
  if len(times) > 1:
    print(f"time to jit compile: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

  print("starting inference...")

  # set up the inference function using the trained model parameters
  # specifying deterministic=True ensures the policy outputs are consistent (no randomness during inference (rollout))
  inference_fn = make_inference_fn(params, deterministic=True)
  # jit-compile the inference function to speed up repeated calls;
  # this compiles the function into optimized machine code so it can run faster.
  jit_inference_fn = jax.jit(inference_fn)

  # set number of environments; if vision is active, adjust the number accordingly.
  num_envs = 1
  if _VISION.value:
    eval_env = env
    num_envs = env_cfg.vision_config.render_batch_size

  # --------------------------------------------------------------------------------
  # get environment-specific jit functions for resetting and stepping the env
  # here, jax.jit compiles these functions so that they run much faster.
  # this is essential because the reset and step functions are called repeatedly
  # during the simulation loop.
  jit_reset = jax.jit(eval_env.reset)
  jit_step = jax.jit(eval_env.step)

  # --------------------------------------------------------------------------------
  # jax requires you to manage randomness explicitly using "random keys" instead of relying on a hidden state.
  # this design forces clarity about where random values come from and ensures that random operations are reproducible.

  rng = jax.random.PRNGKey(0)  
  # here we create a random key with a fixed seed (0).
  # using a fixed seed means that every time this code is run, the random numbers are generated in the same order.
  # in this context, it makes sure that the environment's initial state (or other random operations) is the same across runs.

  rng, reset_rng = jax.random.split(rng)
  # splitting the random key produces two separate keys:
  #  - 'reset_rng' is used to reset the environment, ensuring that the starting state (like the robot's position) is reproducible.
  #  - the remaining 'rng' is reserved for future random operations (e.g., during inference), so each part of the code gets its own independent randomness.

  state = jit_reset(reset_rng)
  # using 'reset_rng' to reset the environment yields an initial state that will always be identical across runs.
  # this reproducibility is essential for debugging, consistent comparisons, and ensuring that the simulation always "spawns in the same spot."


  # --------------------------------------------------------------------------------
  # set up mujoco model and data for live simulation visualization
  # the env should have an attribute 'mj_model' which contains the mujoco model
  if hasattr(eval_env, "mj_model"):
    print("creating mujoco simulation from the 'mj_model' attribute.")
    model = eval_env.mj_model
    data = mujoco.MjData(model)  # create mujoco data, which stores state like joint positions and velocities
  else:
    raise AttributeError("the evaluation environment does not have a 'mj_model' attribute for live visualization.")

  # disable automatic command randomization by setting a very high steps_until_next_cmd in the env state info;
  # this ensures our manual joystick commands are not overwritten by any internal randomness.
  if "steps_until_next_cmd" in state.info:
    state.info["steps_until_next_cmd"] = 9999999

    
    # --------------------------------------------------------------------------------
    # define a helper function: update_observation_with_command
    # this function takes in the current environment state (which contains various pieces of information, including the current "command")
    # and a new command (for example, a new velocity vector that we got from the keyboard).
    # it makes a copy of the state's info (a dictionary holding extra state data) and replaces the "command" with the new one.
    # then it re-computes the observation so that the policy (the neural network controlling the robot) sees this updated command.
    #
    # why do we need this? the observation that the policy uses is generated from both the raw sensor data and the info dictionary.
    # the command is a part of that info. if we want the policy to respond immediately to our keyboard inputs,
    # we must ensure that the observation includes the new command. this function helps enforce that.
  def update_observation_with_command(env_state, command):
    """update the state observation with the new command."""
    new_info = dict(env_state.info)  # copy the current info to avoid modifying original data directly
    new_info["command"] = command  # update the command in this copied info

    try:
        if hasattr(eval_env, "_get_obs"):
            # if the environment has a _get_obs method, re-compute the observation with the updated command
            updated_obs = eval_env._get_obs(env_state.data, new_info)
            return env_state.replace(obs=updated_obs, info=new_info)
        elif hasattr(eval_env, "get_obs"):
            # alternatively, use get_obs if _get_obs is not available
            updated_obs = eval_env.get_obs(env_state.data, new_info)
            return env_state.replace(obs=updated_obs, info=new_info)
        else:
            # if neither method is available, try to manually update the command inside the observation dictionary
            obs_dict = dict(env_state.obs)
            if "command" in obs_dict:
                obs_dict["command"] = command
                return env_state.replace(obs=obs_dict, info=new_info)
            else:
                print("warning: could not update observation with command.")
                return env_state.replace(info=new_info)
    except Exception as e:
        print(f"error updating observation: {e}")
        return env_state.replace(info=new_info)

  # --------------------------------------------------------------------------------
  # print instructions for the user about how to control the simulation
  print("\nstarting live mujoco visualization. use wasd keys to control the robot.")
  print("controls:")
  print("  w/s: forward/backward")
  print("  a/d: left/right lateral movement")
  print("  q/e: turn left/right")
  print("  space: reset all commands to zero")
  print("  z/x: decrease/increase command sensitivity")
  print("close the viewer window to end the session.\n")

  # launch the mujoco viewer in passive mode, which will update the simulation visualization,
  # and register the key_callback function to capture user input.
  with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    live_steps = 0
    try:
        # main simulation loop: while the viewer is open, repeatedly update state and render
        while viewer.is_running():
            # build the command vector from the global joystick variables
            command = jp.array([forward_vel, lateral_vel, turn_rate])
            # update the current environment state's observation with the new command so the policy sees it
            state = update_observation_with_command(state, command)

            # split rng for the policy inference; this ensures consistent randomness control
            act_rng, rng = jax.random.split(rng)
            # use the jit-compiled inference function to compute the action (motor_action) based on the current observation
            motor_action, _ = jit_inference_fn(state.obs, act_rng)

            # step the environment using the compiled step function, which applies the action to advance the simulation
            state = jit_step(state, motor_action)

            # synchronize the mujoco simulation's control signals with the new actions from brax state
            data.ctrl[:] = np.array(state.data.ctrl)
            # step the mujoco simulation by one time increment
            mujoco.mj_step(model, data)
            # update mujoco's simulation state (positions and velocities) if available,
            # ensuring the viewer displays the current state accurately
            if hasattr(state.data, "qpos") and hasattr(state.data, "qvel"):
                data.qpos[:] = np.array(state.data.qpos)
                data.qvel[:] = np.array(state.data.qvel)

            # refresh the viewer so the display updates with new simulation data
            viewer.sync()
            time.sleep(0.01)  # a brief sleep to control the simulation loop speed and reduce CPU load

            live_steps += 1
    except Exception as e:
        print(f"error during live visualization: {e}")

  print(f"live mujoco visualization ended after {live_steps} steps.")

if __name__ == "__main__":
  app.run(main)