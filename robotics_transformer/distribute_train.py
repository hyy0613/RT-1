"""
robotic transformer(https://github.com/google-research/robotics_transformer)的多节点分布式训练代码,
采用tensorflow2的distribute.MultiWorkerMirroredStrategy(https://www.tensorflow.org/api_docs/python/tf/distribute/MultiWorkerMirroredStrategy)进行分布式训练，使用加载rlds(https://github.com/google-research/rlds)数据的方式进行数据的读取
使用方法：
    python distribute_worker_train.py --args = param, 其中args见代码中的get_args()
"""

import os
from robotics_transformer import transformer_network
from tensor2robot.utils import tensorspec_utils
from tf_agents.specs import tensor_spec
import time
from robotics_transformer.data_loader import rlds_dataset_loader
import tensorflow as tf
import jax
import argparse
import json

'''
Description:
    设置分布式训练参数
Parameters:
    None
Return:
    args:包含训练需要的各种参数，参数详情请见代码
'''
def get_args():
    parser = argparse.ArgumentParser(description='获得分布式训练参数')
    parser.add_argument('--single_gpu_batch_size', '-s', help='batch size for single gpu', default=8, type=int)
    parser.add_argument('--training_epoch', '-te', help='training epoch', default=100, type=int)  # 训练epoch
    parser.add_argument('--log_step', '-ls', help='log step', default=10, type=int)
    parser.add_argument('--dataset_dirs', '-d', help='dataset path', default="/adddisk1/dataset/language-table-sim/")
    parser.add_argument('--learning_rate', '-lr', help='learning rate', default=0.00001, type=float)  # 学习率
    parser.add_argument('--vocab_size', '-vs', help='vocab size for discretization', default=256, type=int)  # 离散词典大小
    parser.add_argument('--dataset_episode_num', '-den', help='训练数据量', default=100, type=int)
    parser.add_argument('--loaded_checkpoints_dir', '-lcd', help='模型加载目录', default="~/", type=str)
    parser.add_argument('--save_model', '-sm', help='save model', default=True)
    parser.add_argument('--model_save_epoch', '-mse', help='save model at every num epoch', default=10, type=int)
    parser.add_argument('--checkpoints_saved_dir', '-csd', help='模型保存目录', default="/adddisk1/huangyiyang/model/", type=str)
    args = parser.parse_args()
    return args


time_sequence_length = 6  # 常量，来自论文每次预测使用6张图片


def create_train_dataset(args, global_batch_size):
    '''创建数据集'''
    dataset_dirs = args.dataset_dirs.split("+") # 可以包含多个数据集，输入时用+分割

    workdir = "~/"
    sequence_length = time_sequence_length # 每次图片张数
    data_target_width = 456 # 输入图像的宽度
    data_target_height = 256 # 输入图像的高度
    random_crop_factor = 0.95
    replay_capacity = 5_000
    seed = 42
    rng = jax.random.PRNGKey(seed) #jax可以理解为针对硬件的numpy等的加速包，详见项目pdf
    rng, data_rng = jax.random.split(rng)
    data_rng = jax.random.fold_in(data_rng, jax.process_index())

    '''导入rlds类型的数据集，rlds数据集为google论文中给定的数据格式，详见项目pdf'''
    train_ds = rlds_dataset_loader.create_datasets(
        data_rng,
        dataset_dirs=dataset_dirs,
        sequence_length=sequence_length,
        global_batch_size=global_batch_size,
        target_width=data_target_width,
        target_height=data_target_height,
        random_crop_factor=random_crop_factor,
        cache=False,
        shuffle=True,
        shuffle_buffer_size=replay_capacity,
        cache_dir=workdir,
        dataset_episode_num=args.dataset_episode_num
    )

    return train_ds


def create_model(args):
    '''创建模型'''
    data_target_width = 456
    data_target_height = 256

    state_spec = tensorspec_utils.TensorSpecStruct() #tensorflow的扩展数据类型，详见项目pdf,此处为

    state_spec.image = tensor_spec.BoundedTensorSpec([data_target_height, data_target_width, 3],
                                                     dtype=tf.float32,
                                                     name='image',
                                                     minimum=0.,
                                                     maximum=1.)
    state_spec.natural_language_embedding = tensor_spec.TensorSpec(
        shape=[512], dtype=tf.float32, name='natural_language_embedding')

    action_spec = tensorspec_utils.TensorSpecStruct()

    action_spec.terminate_episode = tensor_spec.BoundedTensorSpec(
        (2,), dtype=tf.int32, minimum=0, maximum=1, name='terminate_episode')

    action_spec.effector_target_translation = tensor_spec.BoundedTensorSpec(
        (2,), dtype=tf.float32, minimum=-0.3, maximum=0.6, name='effector_target_translation')

    network = transformer_network.TransformerNetwork(
        input_tensor_spec=state_spec,
        output_tensor_spec=action_spec,
        vocab_size=int(args.vocab_size),
        token_embedding_size=512,
        num_layers=8,
        layer_size=128,
        num_heads=8,
        feed_forward_size=512,
        dropout_rate=0.1,
        time_sequence_length=time_sequence_length,
        crop_size=236,
        use_token_learner=True,
        action_order=['terminate_episode', 'effector_target_translation'])
    return network

''' 设置 TF_CONFIG 环境变量,为每个集群设置对应的文件'''
def set_env():
    '''设置分布式训练环境变量，详见项目pdf'''
    worker_idx = 0
    swarm = []
    swarm.append("10.20.0.253" + ":" + "10023")
    cluster = {'worker': swarm}
    type = "worker"
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': cluster,
        'task': {'type': type, 'index': worker_idx}
    })


if __name__ == '__main__':
    os.environ.pop('TF_CONFIG', None) # 清除TF_CONFIG

    args = get_args()

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for k in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[k], True)
    else:
        print("GPU数量不够")
        exit("异常退出")

   # set_env()

   # options = tf.distribute.experimental.CommunicationOptions(
   #     implementation=tf.distribute.experimental.CommunicationImplementation.NCCL
   # )
   # mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy(
   #     communication_options=options)
    mirrored_strategy = tf.distribute.MirroredStrategy()
    global_batch_size = args.single_gpu_batch_size * mirrored_strategy.num_replicas_in_sync

    global_learning_rate = args.learning_rate * global_batch_size

    with mirrored_strategy.scope():
        print("begin")
        network = create_model(args)
        network.create_variables()
        print("network")
        dataset_dirs = args.dataset_dirs
        train_ds = create_train_dataset(args, global_batch_size)
        print("dataset")
        dist_dataset = mirrored_strategy.experimental_distribute_dataset(train_ds)
        print("dist_dataset")
        network_state = tensor_spec.sample_spec_nest(
            network.state_spec, outer_dims=[args.single_gpu_batch_size])
        optimizer = tf.keras.optimizers.Adam(learning_rate=global_learning_rate)

        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=network)
       # if tf.train.latest_checkpoint(args.loaded_checkpoints_dir):
           # ckpt.restore(tf.train.latest_checkpoint(args.loaded_checkpoints_dir))
           # print("从 %s 恢复模型" % (args.loaded_checkpoints_dir))

        current_step = ckpt.step.numpy()
        print("开始训练")
        T1 = time.time()


        @tf.function
        def train_one_step(model, observation_batch, label_batch, network_state, optimizer):
            '''单步训练'''
            with tf.GradientTape() as tape:
                model.set_actions(label_batch)
                model(observation_batch, step_type=None, network_state=network_state, training=True)
                loss = tf.reduce_mean(model.get_actor_loss())
                gradients = tape.gradient(loss, model.trainable_variables,
                                          unconnected_gradients=tf.UnconnectedGradients.ZERO)
                optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))
                logging_info = model.get_aux_info()
                return loss, logging_info


        action_order = network._action_tokenizer.action_order
        for epoch in range(1, args.training_epoch):
            total_loss = 0.0
            step = 0
            T1 = time.time()

            for data in dist_dataset:
                train_observation = data["train_observation"]
                train_labels = data["action_lable"]
                per_replica_losses, logging_info = mirrored_strategy.run(
                    train_one_step, args=(network, train_observation, train_labels, network_state, optimizer))
                step = step + 1
                mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
                total_loss = total_loss + mean_loss
                ckpt.step.assign_add(1)
                print('训练1个step')
                print(total_loss)

            T2 = time.time()
            print('训练1个epoch 总耗时: ', ((T2 - T1)))

            if epoch % args.model_save_epoch == 0 and args.save_model:
                checkpoint_prefix = os.path.join(args.checkpoints_saved_dir, "ckpt")
                ckpt.save(checkpoint_prefix)
                print("模型保存位置：  %s !" % (checkpoint_prefix))

print("正常退出!")
