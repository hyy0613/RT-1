"""读取RLDS数据集，详见：https://github.com/google-research/rlds  数据读取代码参考https://github.com/google-research/language-table"""

import dataclasses
import functools
from typing import Optional, Tuple
from clu import preprocess_spec
import jax
import rlds
import tensorflow as tf
import tensorflow_datasets as tfds
import tree

Features = preprocess_spec.Features


def create_datasets(
        rng,
        dataset_dirs,
        sequence_length,
        global_batch_size,
        target_width=320,
        target_height=180,
        random_crop_factor=None,
        cache=False,
        shuffle=True,
        shuffle_buffer_size=50_000,
        cache_dir=None,
        dataset_episode_num=10000
):
    """创建一个RLDS数据集."""

    builder = tfds.builder_from_directories(dataset_dirs)

    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.threading.private_threadpool_size = 48
    dataset_options.threading.max_intra_op_parallelism = 1

    ds = builder.as_dataset(
        split=f'train[{0}:{dataset_episode_num}]',
        decoders={"steps": {"observation": {"rgb": tfds.decode.SkipDecoding()}}},
        shuffle_files=True
    )

    def _pad_episode(episode, padding):
        first_item_tensor = episode["steps"].take(1).get_single_element()
        first_item_ds = tf.data.Dataset.from_tensors(first_item_tensor)

        first_item_mid_tensor = tf.nest.map_structure(
            tf.identity, first_item_tensor
        )
        first_item_mid_tensor[rlds.IS_FIRST] = False
        padding_ds = tf.data.Dataset.from_tensors(first_item_mid_tensor).repeat(
            padding
        )

        full_padding = rlds.transformations.concatenate(first_item_ds, padding_ds)
        episode["steps"] = rlds.transformations.concatenate(
            full_padding, episode["steps"].skip(1)
        )
        return episode

    ds = ds.map(
        functools.partial(_pad_episode, padding=sequence_length - 1),
        tf.data.AUTOTUNE,
    )

    def get_seqlen_pattern(step):
        return {
            rlds.OBSERVATION: tree.map_structure(
                lambda x: x[-sequence_length:], step[rlds.OBSERVATION]
            ),
            rlds.ACTION: tree.map_structure(
                lambda x: x[-sequence_length:], step[rlds.ACTION]
            ),
            rlds.IS_TERMINAL: tree.map_structure(
                lambda x: x[-sequence_length:], step[rlds.IS_TERMINAL]
            ),
        }

    ds = rlds.transformations.pattern_map_from_transform(
        episodes_dataset=ds,
        transform_fn=get_seqlen_pattern,
        respect_episode_boundaries=True,
    )

    if shuffle:
        shuffle_rng, rng = jax.random.split(rng)
        shuffle_rng = shuffle_rng[0]
        ds = ds.shuffle(shuffle_buffer_size, shuffle_rng)

    preprocessors = [
        DecodeAndRandomResizedCrop(
            random_crop_factor=random_crop_factor,
            resize_size=(target_height, target_width),
        ),
        TransformDict(),
    ]
    train_preprocess = preprocess_spec.PreprocessFn(
        preprocessors, only_jax_types=True
    )

    def _preprocess_fn(example_index, features):
        example_index = tf.cast(example_index, tf.int32)
        features[preprocess_spec.SEED_KEY] = (
            tf.random.experimental.stateless_fold_in(
                tf.cast(rng, tf.int64), example_index
            )
        )
        processed = train_preprocess(features)
        return processed

    ds = ds.enumerate().map(_preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(global_batch_size, drop_remainder=True)

    if cache:
        ds = ds.cache(cache_dir)

    return ds


@dataclasses.dataclass(frozen=True)
class DecodeAndRandomResizedCrop(preprocess_spec.RandomMapTransform):
    """解析图像，提取随机crop, resize并归一化"""

    random_crop_factor: Optional[float] = None
    resize_size: Tuple[int, int] = (180, 320)

    def _transform(self, features, seed):
        image = features["observation"]["rgb"]
        shape = tf.io.extract_jpeg_shape(image[0])
        raw_height, raw_width = shape[0], shape[1]
        raw_height = tf.cast(raw_height, tf.float32)
        raw_width = tf.cast(raw_width, tf.float32)

        if self.random_crop_factor is None:
            random_crop_factor = 1.0
            offset_width = 0
            offset_height = 0
            scaled_height = raw_height
            scaled_width = raw_width
        else:
            random_crop_factor = tf.constant(
                self.random_crop_factor, dtype=tf.float32
            )
            scaled_height = raw_height * random_crop_factor
            scaled_width = raw_width * random_crop_factor

            next_rng, rng = tf.unstack(tf.random.experimental.stateless_split(seed))
            offset_height = tf.random.stateless_uniform(
                shape=(),
                seed=next_rng,
                minval=0,
                maxval=tf.cast(raw_height - scaled_height, dtype=tf.int32),
                dtype=tf.int32,
            )

            next_rng, rng = tf.unstack(tf.random.experimental.stateless_split(rng))
            offset_width = tf.random.stateless_uniform(
                shape=(),
                seed=next_rng,
                minval=0,
                maxval=tf.cast(raw_width - scaled_width, dtype=tf.int32),
                dtype=tf.int32,
            )

        def apply_decode_and_crop(image):
            image = tf.image.decode_and_crop_jpeg(
                image,
                [
                    offset_height,
                    offset_width,
                    tf.cast(scaled_height, tf.int32),
                    tf.cast(scaled_width, tf.int32),
                ],
                channels=3,
            )
            return image

        image = tf.map_fn(apply_decode_and_crop, image, dtype=tf.uint8)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.image.resize(image, self.resize_size)
        features["observation"]["rgb"] = image
        return features


@dataclasses.dataclass(frozen=True)
class TransformDict(preprocess_spec.RandomMapTransform):
    """将数据存放字典格式转换成网络所需数据字典格式."""

    def _transform(self, features, seed):
        """Applies all distortions."""
        action_lable = {
            "terminate_episode": tf.one_hot(tf.cast(features["is_terminal"], dtype=tf.int32), depth=2, dtype=tf.int32),
            "effector_target_translation": features["observation"]["effector_target_translation"]}
        train_observation = {"image": features["observation"]["rgb"],
                             "natural_language_embedding": tf.cast(features['observation']['instruction'],dtype=tf.float32)}
        features = {"action_lable": action_lable, "train_observation": train_observation}
        return features
