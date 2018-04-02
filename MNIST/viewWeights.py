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

  print extract_weights(model, 'conv2', 'alpha').shape



if __name__ == "__main__":
  main()




