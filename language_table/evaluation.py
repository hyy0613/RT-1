import imageio
from robotics_transformer.distribute_train import get_args,create_model,create_train_dataset
from tf_agents.specs import tensor_spec
import tensorflow as tf
import collections
from collections.abc import Sequence
import os
from absl import app
from absl import flags
from absl import logging
import jax
import numpy as np

from language_table.common import rt1_tokenizer
from language_table.environments import blocks
from language_table.environments import language_table
from language_table.environments.oracles import push_oracle_rrt_slowdown
from language_table.environments.rewards import block2absolutelocation
from language_table.environments.rewards import block2block
from language_table.environments.rewards import block2block_relative_location
from language_table.environments.rewards import block2relativelocation
from language_table.environments.rewards import separate_blocks
from language_table.eval import wrappers as env_wrappers
from language_table.train import policy as jax_policy
from ml_collections import config_flags

import tensorflow as tf
import tensorflow_hub as hub
from tf_agents.environments import gym_wrapper
from tf_agents.environments import wrappers as tfa_wrappers

_CONFIG = config_flags.DEFINE_config_file(
    "config","/adddisk1/huangyiyang/code/language_table/train/configs/language_table_sim_local.py", "Training configuration.", lock_config=True)
_WORKDIR = flags.DEFINE_string("workdir","/adddisk1/huangyiyang/eval", "working dir")

def get_ckpt_model():
    time_sequence_length = 6  # 常量，来自论文每次预测使用6张图片
    args = get_args()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    with tf.device('/gpu:1'):
        network = create_model(args)
        network_state = tensor_spec.sample_spec_nest(
            network.state_spec, outer_dims=[1])
        ckpt = tf.train.Checkpoint(step=tf.Variable(1),model=network)
        if tf.train.latest_checkpoint(args.loaded_checkpoints_dir):
            ckpt.restore(tf.train.latest_checkpoint(args.loaded_checkpoints_dir)).expect_partial()
            print("从 %s 恢复模型" % (args.loaded_checkpoints_dir))
        return ckpt.model,network_state

def evaluate_checkpoint(workdir, config):
  """Evaluates the given checkpoint and writes results to workdir."""
  video_dir = os.path.join(workdir, "videos")
  if not tf.io.gfile.exists(video_dir):
    tf.io.gfile.makedirs(video_dir)
  rewards = {
      "blocktoblock":
          block2block.BlockToBlockReward,
      "blocktoabsolutelocation":
          block2absolutelocation.BlockToAbsoluteLocationReward,
      "blocktoblockrelativelocation":
          block2block_relative_location.BlockToBlockRelativeLocationReward,
      "blocktorelativelocation":
          block2relativelocation.BlockToRelativeLocationReward,
      "separate":
          separate_blocks.SeparateBlocksReward,
  }

  num_evals_per_reward = 50
  max_episode_steps = 200

  policy = None
  model,network_state = get_ckpt_model()

  results = collections.defaultdict(lambda: 0)
  for reward_name, reward_factory in rewards.items():
    env = language_table.LanguageTable(
        block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
        reward_factory=reward_factory,
        seed=0)
    env = gym_wrapper.GymWrapper(env)
    env = env_wrappers.UseTokenWrapper(env)
    env = env_wrappers.CentralCropImageWrapper(
        env,
        target_width=config.data_target_width,
        target_height=config.data_target_height,
        random_crop_factor=config.random_crop_factor)
    env = tfa_wrappers.HistoryWrapper(
        env, history_length=config.sequence_length, tile_first_step_obs=True)

    if policy is None:
      policy = jax_policy.BCJaxPyPolicy(
          env.time_step_spec(),
          env.action_spec(),
          model=model,
          network_state=network_state,
          rng=jax.random.PRNGKey(0))

    for ep_num in range(num_evals_per_reward):
      # Reset env. Choose new init if oracle cannot find valid motion plan.
      # Get an oracle. We use this at the moment to decide whether an
      # environment initialization is valid. If oracle can motion plan,
      # init is valid.
      oracle_policy = push_oracle_rrt_slowdown.ObstacleOrientedPushOracleBoard2dRRT(
          env, use_ee_planner=True)
      plan_success = False
      while not plan_success:
        ts = env.reset()
        raw_state = env.compute_state()
        plan_success = oracle_policy.get_plan(raw_state)
        if not plan_success:
          logging.info(
              "Resetting environment because the "
              "initialization was invalid (could not find motion plan).")

      frames = [env.render()]

      episode_steps = 0
      while not ts.is_last():
        policy_step = policy.action(ts, ())
        ts = env.step(policy_step.action)
        frames.append(env.render())
        episode_steps += 1

        if episode_steps > max_episode_steps:
          break

      success_str = ""
      if env.succeeded:
        results[reward_name] += 1
        logging.info("Episode %d: success.", ep_num)
        success_str = "success"
      else:
        logging.info("Episode %d: failure.", ep_num)
        success_str = "failure"

      # Write out video of rollout.
      video_path = os.path.join(workdir, "videos/",
                                f"{reward_name}_{ep_num}_{success_str}.mp4")

      imageio.mimsave(video_path, frames, fps=10)

    print(results)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
      evaluate_checkpoint(
          workdir=_WORKDIR.value,
          config=_CONFIG.value,
      )
      
if __name__ == "__main__":
    app.run(main)




