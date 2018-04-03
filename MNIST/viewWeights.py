import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np

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


def view(model, layer, weight, mode='raw'):
  weight_map = extract_weights(model, layer, weight)
  if model == 'pruned':
    alpha = extract_weights(model, layer, weight)

  output_dim = weight_map.shape[0]
  weight_in_2D = weight_map.view(output_dim,-1)
  fig, ax = plt.subplots()
  ax.imshow(weight_in_2D.numpy())
  if model == 'pruned':
    x = np.arange(output_dim)
  ax.autoscale(False)
  alpha_value = extract_weights(model, layer, 'alpha')
  rows = alpha_value.size()[1]
  mat = torch.FloatTensor(int(torch.sum(alpha_value)), int(weight_in_2D.shape[1]))
  i = 0
  for row in range(rows):
    if alpha_value.squeeze()[row] == 1:
      mat[i,:] = weight_in_2D[row,:]
      i+=1
  ax.imshow(extract_weights(model, layer, 'alpha').numpy().transpose())
  plt.matshow(mat)
  plt.show()
  print mat

def main():
  args = get_cmd_opts()

  model = torch.load(args.model)

#  weight_map = extract_weights(model, 'conv1', 'weight')
#  alpha = extract_weights(model, 'conv1', 'alpha')

  view(model, 'ip1', 'weight')

if __name__ == "__main__":
  main()




