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
conda env create -f RT-1/rt_environment.yaml

# Run distributed code

```

### Running Tests

To run RT-1 tests, you can clone the git repo and run
[bazel](https://bazel.build/):

```bash
git clone https://github.com/google_research/robotics_transformer.git
cd robotics_transformer
bazel test ...
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

## 新增内容
本工程来自于google的robotics transformer工作，在此基础上增加了:
    
    1,读取rlds数据，以language table为例
    2,增加分布式训练代码
   
## 训练
    step1, 下载language_table数据, 见https://github.com/google-research/language-table
    step2, 下载Universal Sentence Encoder模型，将数据集中的文本(UTF-8)转化成USE embedding，依然是RLDS格式
    step3, 设置distribute_train.py里的变量
    step4, 开启训练 python distribute_train.py

## 未来工作
    1, 使用RT1开源的数据集进行训练，编写相应数据加载代码;
    2, 使用梯度累积，混合精度训练，增加单卡batch_size;
    3, 融合更多传感器信息，如深度等;
    抛砖引玉，欢迎大家提pull request

   
