import torch
import argparse

def getCmdOpts():
  parser = argparse.ArgumentParser(description='PyTorch model viewer')
  parser.add_argument('--model', action='store', default=None,
            help='path to model file')

  return parser.parse_args()


def extract_weights(model, layer, weight='kernel'):
  if weight == 'kernel':
    return model['state_dict'][layer+'.weight']
  elif weight == 'bias':
    return model['state_dict'][layer+'.bias']
  elif weight == 'alpha':
    return model['state_dict']['mask_'+layer+'.alpha']
  elif weight == 'beta':
    return model['state_dict']['mask_'+layer+'.beta']
  else:
    raise Exception('Only kernel, bias, alpha, beta  value can be extracted from the model.')

def main():
  args = getCmdOpts()

  model = torch.load(args.model)

  alpha_v = extract_weights(model, 'ip1', 'alpha')
  beta_v = extract_weights(model, 'ip1', 'beta')

  print type(alpha_v)
  print alpha_v.size()[0], alpha_v.size()[1]
  print alpha_v.size(), beta_v.size()
  for i in range(alpha_v.size()[0]):
    for j in range(alpha_v.size()[1]):
      print alpha_v[i][j], '-'*3, beta_v[i][j]

  print extract_weights(model, 'conv1', 'kernel').shape
  


if __name__ == "__main__":
  main()




