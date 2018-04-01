import torch
import argparse

def getCmdOpts():
  parser = argparse.ArgumentParser(description='PyTorch model viewer')
  parser.add_argument('--model', action='store', default=None,
            help='path to model file')

  return parser.parse_args()


def main():
  args = getCmdOpts()

  model = torch.load(args.model)
  #acc = model['acc']
  #print acc
  #print model['state_dict']


if __name__ == "__main__":
  main()




