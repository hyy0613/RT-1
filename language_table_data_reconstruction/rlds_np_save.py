import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import io
import rlds
from tqdm import tqdm
import os


# 导入原始数据集
dataset_dirs = ['/adddisk1/dataset/language-table-sim/']
dataset_episode_num = 24870
builder = tfds.builder_from_directories(dataset_dirs)

ds = builder.as_dataset(
    split=f'train[{0}:{dataset_episode_num}]',
    decoders={"steps": {"observation": {"rgb": tfds.decode.SkipDecoding()}}},
    shuffle_files=False
)

from tqdm import tqdm
import os


# 将原本的数据存为np文件
def create_episode(path,raw_episode):
    episode = []
    for step in raw_episode[rlds.STEPS]:
        observation = step[rlds.OBSERVATION]
        observation_keys = observation.keys()
        step_keys = list(step.keys())
        step_keys.remove(rlds.OBSERVATION)
        step_dict = {}
        for k in step_keys:
            step_dict[k] = step[k].numpy()
        for k in observation_keys:
            step_dict[k] = observation[k].numpy()
        episode.append(step_dict)
    np.save(path, episode)

# 设置数据集中的episode数量

NUM_TRAIN = 20000
NUM_VAL = 3240
NUM_TEST = 1630

# 分别创建出对应的数据集

print("Generating train examples...")
os.makedirs('/adddisk1/huangyiyang/np_dataset/data/train', exist_ok=True)
cnt = 0
for element in tqdm(ds.take(NUM_TRAIN)):
    create_episode(f'/adddisk1/huangyiyang/np_dataset/data/train/episode_{cnt}.npy',element)
    cnt = cnt + 1

print("Generating val examples...")
os.makedirs('/adddisk1/huangyiyang/np_dataset/data/val', exist_ok=True)
cnt = 0
for element in tqdm(ds.skip(NUM_TRAIN).take(NUM_VAL)):
    create_episode(f'/adddisk1/huangyiyang/np_dataset/data/val/episode_{cnt}.npy', element)
    cnt = cnt + 1

print("Generating test examples...")
os.makedirs('/adddisk1/huangyiyang/np_dataset/data/test', exist_ok=True)
cnt = 0
for element in tqdm(ds.skip(NUM_TRAIN + NUM_VAL).take(NUM_TEST)):
    create_episode(f'/adddisk1/huangyiyang/np_dataset/data/test/episode_{cnt}.npy', element)
    cnt = cnt + 1







