import tensorflow_datasets as tfds
from language_table.environments import language_table
from language_table.environments.rewards import block2block
from language_table.environments import blocks
import imageio
import numpy as np
import matplotlib.pyplot as plt

def decode_inst(inst):
    return bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8")


env = language_table.LanguageTable(
      block_mode=blocks.LanguageTableBlockVariants.BLOCK_8,
      reward_factory=block2block.BlockToBlockReward,
      control_frequency=10.0,
  )
dataset_path = '/adddisk1/dataset/language-table-sim'
builder = tfds.builder_from_directory(dataset_path)
episode_ds = builder.as_dataset(split='train')
episode = next(iter(episode_ds.take(1)))
frames = []

action = []
target_translation = []
translation = []
action_visual = []
for step in episode['steps'].as_numpy_iterator():
    action.append(step['action'])
    target_translation.append(step['observation']['effector_target_translation'])
    translation.append(step['observation']['effector_translation'])
    frames.append(step['observation']['rgb'])

# 创建一个图表
plt.figure()

# 提取二维列表中的x和y坐标
target_x = [item[0] for item in target_translation]
target_y = [item[1] for item in target_translation]

translation_x = [item[0] for item in translation]
translation_y = [item[1] for item in translation]

# 绘制散点图
plt.scatter(target_x, target_y, label='target_translation')
plt.scatter(translation_x, translation_y, label='translation')

# 添加标题和标签
plt.xlabel('x')
plt.ylabel('Y')

# 添加图例
plt.legend()

# 显示图表
plt.show()

# env.reset()
# # fig, axes = plt.subplots(1, len(action), figsize=(15, 5))
# video_path ='./frames.mp4'
# video2_path = './action.mp4'
# for i in range(len(action)):
#     obs, _, _, _ = env.step(np.array(action[i]))
#     action_visual.append(obs['rgb'])
#
# imageio.mimsave(video_path, frames, fps=10)
# imageio.mimsave(video2_path,action_visual, fps=5)
