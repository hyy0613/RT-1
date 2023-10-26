# Tutorial

In order to use this part of the code, you need to follow these steps

1. Modify the dataset location in the `rlds_np_save`,and run this program then you will get the folder named `data`

   ```bash
   python rlds_np_save
   ```

   The data directory is going to look like this:

   ```python
   data
   ├── train
   │   ├── episode_0
   │   ├── episode_1
   │   ├── ...
   ├── val
   │   ├── episode_100
   │   ├── episode_101
   │   ├── ...
   ├── test
   │   ├── episode_201
   │   ├── episode_202
   │   ├── ...
   
   ```

2. Note that next you need to place the `data` folder under the `language_table_use` directory, or you can specify the path directly in `rlds_np_save`.

   The language_table_use directory is going to look like this:

   ```python
   language_table_use
   ├── data
   ├── language_table_use_dataset_builder.py
   ├── ...
   ```

3. Download the [Universial Sentence Encoder](https://www.tensorflow.org/hub/tutorials/semantic_similarity_with_tf_hub_universal_encoder) from tensorflow
4.  Next, create the `conda` environment and build the dataset using the `tfds` command in the `language_table_use` directory
    ```bash
    # create the environment
    conda env create -f environment_ubuntu.yml
    # activate the conda environment
    conda activate rlds_env
    # tfds build
    tfds build --data_dir 'Fill in the data final storage path here'
    ```

   ## Acknowledgements 

   This part  modify and use code from the [rlds_dataset_builder](https://github.com/kpertsch/rlds_dataset_builder) repository by [kpertsch](https://github.com/kpertsch/rlds_dataset_builder/commits?author=kpertsch)，You can go to the repository for more details.

   We would like to express our gratitude to the original author for their valuable work.
