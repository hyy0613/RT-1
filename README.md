# RT-1
This is the completion of google's rt-1 project code and can run directly.

You can view the google source code here: [robotics_transformer](https://github.com/google-research/robotics_transformer)
<p align="center">
<img width="715" alt="RT-1" src="https://github.com/YiyangHuang-work/RT-1/assets/75081077/14f44158-e264-447f-bfd4-c8dccd03abe2">
</p>

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

Before running, you need to resolve the tensorflow version compatibility issues in [tensor2robot](https://github.com/google-research/tensor2robot), see my answer [contrib_answer](https://github.com/google-research/robotics_transformer/issues/1#issuecomment-1673121690)
```bash
git clone https://github.com/YiyangHuang-work/RT-1
# clone the repo rt-1 used
cd RT-1
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

## 使用方法

1.下载**language-table**数据集，详见上文**Downloading the dataset**

2.下载**Universal Sentence Encoder**模型，对**instrucion**进行编码，原数据集为**UTF-8**编码格式

3.运行前需要解决`tensorflow`版本兼容性问题，参见我的回答[contrib_answer](https://github.com/google-research/robotics_transformer/issues/1#issuecomment-1673121690),运行**distribute_train.py**,保存模型

4.将模型预测输出的**action**，通过仿真环境中的**step**方法与环境连接，进行交互，仿真环境详见[language_table](https://github.com/google-research/language-table)。






## Code specification
<img width="715" alt="RT-1" src="https://github.com/YiyangHuang-work/RT-1/assets/75081077/e91556c4-4a12-4151-819a-1418cbed3080">

* 此项目代码来源于google的[robotics_transformer](https://github.com/google-research/robotics_transformer)论文代码

* 训练前请修改`distribute_train.py`中的参数，`tensorflow`版本`2.12`或`2.13`目前测试均可运行
## 近期将要完成工作
* add evaluation tutorial

## Special thanks
Special thanks to these people for their help in this project:[oym1994](https://github.com/oym1994),[kpertsch](https://github.com/kpertsch)
## Contact
The project will continue to improve and update, if you have any questions about the use of this project or suggestions for modification, please contact us by email 120l021822@stu.hit.edu.cn
   
