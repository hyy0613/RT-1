import tensorflow_hub as hub


def tokenize_text(text):
  """Tokenizes the input text given a tokenizer."""
  embed = hub.load("/adddisk1/huangyiyang/save_model")
  tokens = embed(text)
  return tokens
