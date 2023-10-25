import tensorflow_hub as hub
import tensorflow as tf


def tokenize_text(text):
  """Tokenizes the input text given a tokenizer."""
  with tf.device('/gpu:2'):
      embed = hub.load("/adddisk1/huangyiyang/save_model")
      tokens = embed([text])[0].numpy()
      del embed # 通过删除变量来清除模型
      tf.keras.backend.clear_session()  # 清除TensorFlow会话
      return tokens
