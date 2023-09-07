# RT-1
This is the completion of google's rt-1 project code and can run directly.

You can view the google source code here: [robotics_transformer](https://github.com/google-research/robotics_transformer)

## Features

* Film efficient net based image tokenizer backbone
* Token learner based compression of input tokens
* Transformer for end to end robotic control
* Testing utilities

## Getting Started
### Downloading the dataset
**RT-1** dataset: [robotics_transformer_dataset](https://console.cloud.google.com/storage/browser/gresearch/rt-1-data-release;tab=objects?prefix=&forceOnObjectsSortingFiltering=false)

**Language-table** dataset: [language_table_dataset](https://github.com/google-research/language-table)

We are now using the **languaget_table** datasetloader, the **RT-1** dataloader will subsequent upload.

Both datasets are in [RLDS](https://arxiv.org/abs/2111.02767) format

### Installation
Clone the repo
```bash
git clone https://github.com/YiyangHuang-work/RT-1
# clone the repo rt-1 used
git clone https://github.com/google-research/tensor2robot

# Install protoc and compile the protobufs.
pip install protobuf
cd tensor2robot/proto
protoc -I=./ --python_out=`pwd` tensor2robot/t2r.proto

# Optional: Create a conda env,you can also follow google's instructions for configuration
cd ../..
conda env create -f RT-1/rt_environment.yaml

# Run distributed code
python -m robotics_transformer.distribute_train
```
### Using trained checkpoints
Checkpoints are included in trained_checkpoints/ folder for three models:
1. [RT-1 trained on 700 tasks](trained_checkpoints/rt1main)
2. [RT-1 jointly trained on EDR and Kuka data](trained_checkpoints/rt1multirobot)
3. [RT-1 jointly trained on sim and real data](trained_checkpoints/rt1simreal)

They are tensorflow SavedModel files. Instructions on usage can be found [here](https://www.tensorflow.org/guide/saved_model)

## Future Releases

The current repository includes an initial set of libraries for early adoption.
More components may come in future releases.

## License

The Robotics Transformer library is licensed under the terms of the Apache
license.

## Code specification

* 此项目代码来源于google的[robotics_transformer](https://github.com/google-research/robotics_transformer)论文代码

* 训练前请修改`distribute_train.py`中的参数，`tensorflow`版本`2.12`或`2.13`目前测试均可运行
## 近期将要完成工作
* 添加Universal Sentence Encoder模型数据集处理代码
  
* 添加**RT-1**数据导入代码，目前导入数据集为**language-table-dataset**

* 参考**language-table**代码添加仿真环境

* 添加项目代码详细tutorial
   
