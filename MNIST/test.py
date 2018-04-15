import torch
import numpy as np
import matplotlib.pyplot as mp

#model_path='saved_models/LeNet_300_100.best_origin.pth.tar'

#model = torch.load(model_path)

#for k in model['state_dict'].keys():
  #  print k


def calSim(mat):
  hamming_val = np.arange(len(mat) * len(mat), dtype=float).reshape(len(mat), len(mat))
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
      

def cast_2_zero(mat, threshold):
  new_mat = np.arange(len(mat) * len(mat), dtype=float).reshape(len(mat), len(mat))
  for row in range(len(mat)):
    for col in range(len(mat)):
      if mat[row][col] < threshold:
        new_mat[row][col] = 0
      else:
        new_mat[row][col] = mat[row][col]
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


def main():
  Size = 55
  Threshold = 0.85
  np.set_printoptions(precision = 2)
  matA = np.random.rand(Size,Size)
  print matA
  print
  matA = cast_2_zero(matA, Threshold)
  print matA
  print
  print calSim(matA)

  print 
  print np.argsort(calSim(matA))
  #print reform_mat(matA, calSim(matA))
  transformed_mat = reform_mat(matA, calSim(matA))
  print matA
  print
  print transformed_mat

  mp.matshow(matA)
  mp.title("matA")
  mp.matshow(transformed_mat)
  mp.title('transformed_mat')
  mp.show()

if __name__ == '__main__':
  main()


