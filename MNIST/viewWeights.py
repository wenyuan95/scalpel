import torch
import argparse
import matplotlib.pyplot as plt

def get_cmd_opts():
  parser = argparse.ArgumentParser(description='PyTorch model viewer')
  parser.add_argument('--model', action='store', default=None,
            help='path to model file')

  return parser.parse_args()


def get_layer_names(model):
  return [name for name in model['state_dict'].keys()]


def extract_weights(model, layer, weight='kernel'):
  layer_list = get_layer_names(model)
  for layer_name in layer_list:
    if layer in layer_name and weight in layer_name:
      return model['state_dict'][layer_name]
  
  raise Exception('Only kernel, bias, alpha, beta  value can be extracted from the model.')

def main():
  args = get_cmd_opts()

  model = torch.load(args.model)

  weight_map = extract_weights(model, 'conv1', 'weight')
  print weight_map.squeeze().numpy().shape
  y = torch.zeros(20,25)
  y = weight_map.view(20,25)
  print y.shape
  plt.matshow(y.numpy().transpose())

  plt.show()

if __name__ == "__main__":
  main()




