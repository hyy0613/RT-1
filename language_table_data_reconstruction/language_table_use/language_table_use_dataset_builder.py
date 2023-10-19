from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class LanguageTableUse(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("/adddisk1/huangyiyang/save_model")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'rgb': tfds.features.Image(
                            shape=(360, 640, 3),
                            dtype=np.uint8,
                            doc='RGB observation.',
                        ),
                        'effector_target_translation': tfds.features.Tensor(
                            shape=(2,),
                            dtype=np.float32,
                            doc='robot effector target,like x,y in the 2-D dimension',
                        ),
                        'effector_translation': tfds.features.Tensor(
                            shape=(2,),
                            dtype=np.float32,
                            doc='robot effector state,like x,y in the 2-D dimension',
                        ),
                        'instruction': tfds.features.Tensor(
                            shape=(512,),
                            dtype=np.float32,
                            doc='universial sentence embedding instruction',
                        ),
                    }),
                    'action': tfds.features.Tensor(
                        shape=(2,),
                        dtype=np.float32,
                        doc='Robot action',
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='data/train/episode_*.npy'),
            'val': self._generate_examples(path='data/val/episode_*.npy'),
	    'test': self._generate_examples(path='data/test/episode_*.npy'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            with open(episode_path,'rb') as file:
                data = np.load(file, allow_pickle=True)     # this is a list of dicts in our case

            def decode_inst(inst):
                return bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8")

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i, step in enumerate(data):
                # compute Kona language embedding
                language_embedding = self._embed([decode_inst(np.array(step['instruction']))])[0].numpy()


                episode.append({
                    'observation': {
                        'rgb': step['rgb'],
                        'effector_target_translation': step['effector_target_translation'],
                        'effector_translation': step['effector_translation'],
                        'instruction': language_embedding,
                    },
                    'action': step['action'],
                    'reward': step['reward'],
                    'is_first': step['is_first'],
                    'is_last': step['is_last'],
                    'is_terminal': step['is_terminal'],
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(path)

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

