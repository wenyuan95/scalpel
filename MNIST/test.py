import torch
import numpy as np
import matplotlib.pyplot as mp
import argparse

#model_path='saved_models/LeNet_300_100.best_origin.pth.tar'

#model = torch.load(model_path)

#for k in model['state_dict'].keys():
  #  print k

def get_cmd_opts():
  parser = argparse.ArgumentParser(description='PyTorch model viewer')
  parser.add_argument('--model', action='store', default=None,
            help='path to model file')

  return parser.parse_args()

def get_layer_names(model):
  print [name for name in model['state_dict'].keys()]
  return [name for name in model['state_dict'].keys()]

def extract_weights(model, layer, weight='kernel'):
  layer_list = get_layer_names(model)
  for layer_name in layer_list:
    print layer_name
    if layer in layer_name and weight in layer_name:
      return model['state_dict'][layer_name].numpy()

  raise Exception('Only kernel, bias, alpha, beta  value can be extracted from the model.')


def calSim(mat):
  hamming_val = np.arange(len(mat) * len(mat), dtype=float).reshape(len(mat), len(mat))
  print len(mat)
  for i in range(len(mat)):
    for j in range(len(mat)):
      sum_ham = 0
      new_list = []
      new_list = map(lambda x, y: x-y, mat[i], mat[j])
      for t in new_list:
        if t != 0:
          sum_ham += 1
      hamming_val[i][j] = sum_ham
      if i == j:
        hamming_val[i][j] = np.inf
  return hamming_val
      

def calSim_by_col(mat):
  mat_t = mat.transpose()
  hamming_val = np.arange(len(mat) * len(mat), dtype=float).reshape(len(mat), len(mat))
  for i in range(len(mat)):
    for j in range(len(mat)):
      sum_ham = 0
      new_list = []
      new_list = map(lambda x, y: x-y, mat_t[i], mat_t[j])
      for t in new_list:
        if t != 0:
          sum_ham += 1
      hamming_val[i][j] = sum_ham
      if i == j:
        hamming_val[i][j] = np.inf
  return hamming_val


def cast_2_zero(mat, threshold):
  new_mat = np.arange(len(mat) * len(mat), dtype=float).reshape(len(mat), len(mat))
  for row in range(len(mat)):
    for col in range(len(mat)):
      if abs(mat[row][col]) < threshold:
        new_mat[row][col] = 0
      else:
#        new_mat[row][col] = mat[row][col]
        new_mat[row][col] = 1
  return new_mat

def reform_mat(mat, ham_mat):
  new_mat = np.copy(mat)
  curr = 0
  new_mat[0, :] = mat[0,:]
  #print new_mat
  record = set()
  index_mat = np.argsort(ham_mat)
  for i in range(1, len(index_mat)):
    for idx in index_mat[i]:
      if idx not in record:
        new_mat[i,:] = mat[idx, :]
        record.add(idx)
        break 
    
  print '===',record
  return new_mat

#def reform_mat_by_col(mat, ham_mat):
#  new_mat = np.copy(mat)
#  curr = 0
#  new_mat[:,0] = mat[:, 0]
#  record = set()
#  index_mat = np.argsort(ham_mat)
#  for i in range(1, len(index_mat)):
#    for idx in index_mat[:,i]: #TODO


def main():
  Size = 55
  Threshold = 0.13

  np.set_printoptions(precision = 2)
  matA = np.random.rand(Size,Size)

  args = get_cmd_opts()
  model = torch.load(args.model)
  matA = extract_weights(model, 'ip2', 'weight')

  print matA
  print
  matA = cast_2_zero(matA, Threshold)
  print matA
  print
  print "start..."
#  print calSim(matA)
  print "ending ..." 

  print 
#  print np.argsort(calSim(matA))
  #print reform_mat(matA, calSim(matA))
  transformed_mat = reform_mat(matA, calSim(matA))
  print matA
  print
  print transformed_mat
  transform_col = reform_mat(matA.transpose(), calSim_by_col(matA))
  transform_row_col = reform_mat(transformed_mat.transpose(), calSim_by_col(transformed_mat.transpose()))

  mp.matshow(matA)
  mp.title("matA")
  mp.matshow(transformed_mat)
  mp.title('transformed_mat')
  mp.matshow(transform_col.transpose())
  mp.title('transform_mat_by_col')
  mp.matshow(transform_row_col)
  mp.title('row_col')
  mp.show()

if __name__ == '__main__':
  main()


