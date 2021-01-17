import transformers
import json
from datasets import load_dataset
from collections import namedtuple
from train import train

def convert(dictionary):
    return namedtuple('GenericDict', 
                       dictionary.keys())(**dictionary)

def main():
  with open('config.json') as f:
    data = json.load(f)
  config = convert(data)
  dataset = load_dataset("squad")
  train(config, dataset)


if __name__ == '__main__':
  main()