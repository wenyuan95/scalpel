import torch
import numpy as np
import matplotlib.pyplot as mp
import argparse
import scipy.sparse as sp



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
    if layer in layer_name and weight in layer_name:
      raw_tensor = model['state_dict'][layer_name]
      output_dim = raw_tensor.shape[0]
      mat = raw_tensor.view(output_dim, -1).numpy()
      return mat

  raise Exception('Only weight, bias, alpha, beta  value can be extracted from the model.')


#def calSim(mat):
#  rows = mat.shape[0]
#  hamming_val = np.arange(rows * rows, dtype=float).reshape(rows, rows)
#  for i in range(mat.shape[0]):
#    for j in range(mat.shape[0]):
#      sum_ham = 0
#      new_list = []
#      new_list = map(lambda x, y: x-y, mat[i], mat[j])
#      for t in new_list:
#        if t != 0:
#          sum_ham += 1
#      hamming_val[i][j] = sum_ham
#      if i == j:
#        hamming_val[i][j] = np.inf
#  return hamming_val
#      
#
#def calSim_by_col(mat):
#  mat_t = mat.transpose()
#  hamming_val = np.arange(mat.shape[1]*mat.shape[1], dtype=float).reshape(mat.shape[1], mat.shape[1])
#  for i in range(mat.shape[1]):
#    for j in range(mat.shape[1]):
#      sum_ham = 0
#      new_list = []
#      new_list = map(lambda x, y: x-y, mat_t[i], mat_t[j])
#      for t in new_list:
#        if t != 0:
#          sum_ham += 1
#      hamming_val[i][j] = sum_ham
#      if i == j:
#        hamming_val[i][j] = np.inf
#  return hamming_val

def calculate_similarity(mat, dim='row'):
  if dim == 'row':
    mat_ = mat.copy()
  elif dim == 'col':
    mat_ = mat.transpose()
  else:
    raise Exception('you can only calculate hamming distance along either "row" or "col"')

  lines = mat_.shape[0]
  hamming_matrix = np.arange(lines * lines, dtype=float).reshape(lines, lines)
  for i in range(lines):
    for j in range(lines):
      hamming_acc = 0
      line_hammings = []
      line_hammings = map(lambda x, y: x-y, mat_[i], mat_[j])
      for element in line_hammings:
        if element != 0:
          hamming_acc += 1
      hamming_matrix[i][j] = hamming_acc
      if i == j:
        hamming_matrix[i][j] = np.inf
  return hamming_matrix


def cast_2_zero(mat, threshold):
  new_mat = np.zeros(mat.shape)
  for row in range(mat.shape[0]):
    for col in range(mat.shape[1]):
      if abs(mat[row][col]) < threshold:
        new_mat[row][col] = 0
      else:
#        new_mat[row][col] = mat[row][col]
        new_mat[row][col] = 1

  new_mat2 = np.zeros(mat.shape)
  new_mat2 = np.piecewise(mat, [abs(mat)<threshold, abs(mat)>threshold], [0,1])
  for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
      assert new_mat[i][j] == new_mat2[i][j]

  return new_mat

def reform_mat(mat, ham_mat):
#  new_mat = np.zeros(mat.shape)
#  curr = 0
#  new_mat[0, :] = mat[0,:]
#  record = set()
#  record.add(0)
#  #np.set_printoptions(threshold=np.nan)
#  print ham_mat
#  index_mat = np.argsort(ham_mat, axis=1)
#  print '-'*35
#  print index_mat
#  for i in range(1, len(index_mat)):
#    for idx in index_mat[i]:
#      if idx not in record:
#        new_mat[i,:] = mat[idx, :]
#        record.add(idx)
#        break 

  visited = set()
  nnz = np.count_nonzero(mat, axis=1)
  first_row = np.argmin(nnz)
  visited.add(first_row)
  target_row = first_row
  new_mat = mat[first_row, :]
  hamming_distance_idx_lst = np.argsort(ham_mat, axis=1)
  while len(visited) < len(ham_mat):
    for closest_row in hamming_distance_idx_lst[target_row]:
      if closest_row not in visited:
        new_mat = np.vstack((new_mat, mat[closest_row, :]))
        visited.add(closest_row)
        target_row = closest_row
        break

  return new_mat

#def reform_mat_by_col(mat, ham_mat):
#  new_mat = np.copy(mat)
#  curr = 0
#  new_mat[:,0] = mat[:, 0]
#  record = set()
#  index_mat = np.argsort(ham_mat)
#  for i in range(1, len(index_mat)):
#    for idx in index_mat[:,i]: #TODO


def calculate_zero_box_num(mat, block_size=(2,2)):
    packed_weights = sp.bsr_matrix(mat, blocksize=block_size)
    sparsity_pct = 100 * (1 - (packed_weights.count_nonzero() / float(mat.size)))
    
    return packed_weights.nnz



def main():
  Size = 55
  Threshold = 0.13

  np.set_printoptions(precision = 2)
  matA = np.random.rand(Size,Size)

  args = get_cmd_opts()
  model = torch.load(args.model)
  matA = extract_weights(model, 'ip2', 'weight')

  matA = cast_2_zero(matA, Threshold)

  #####transformed_mat = reform_mat(matA, calSim(matA))
  transformed_mat = reform_mat(matA, calculate_similarity(matA))
  #####transform_col = reform_mat(matA.transpose(), calSim_by_col(matA))
  transform_col = reform_mat(matA.transpose(), calculate_similarity(matA, 'col'))
  #####transform_row_col = reform_mat(transformed_mat.transpose(), calSim_by_col(transformed_mat))
  transform_row_col = reform_mat(transformed_mat.transpose(), calculate_similarity(transformed_mat, 'col'))
  
  # first col then row
  transform_col_row = reform_mat(transform_col, calculate_similarity(transform_col))

  


#  print "matA: ", calculate_zero_box_num(matA,(1,5))
#  print "transformed_mat: ", calculate_zero_box_num(transformed_mat,(1,5))
#  print "transform_mat_by_col:", calculate_zero_box_num(transform_col.transpose(), (1,5))
#  print "row_col: ", calculate_zero_box_num(transform_row_col, (1,5))

  matA_nnz = np.count_nonzero(matA)
  permute_row_nnz = np.count_nonzero(transformed_mat)
  permute_col_nnz = np.count_nonzero(transform_col.transpose())
  row_col_nnz = np.count_nonzero(transform_row_col)
  col_row_nnz = np.count_nonzero(transform_col_row)

  matA_blk = calculate_zero_box_num(matA)
  permute_row_blk = calculate_zero_box_num(transformed_mat)
  permute_col_blk = calculate_zero_box_num(transform_col.transpose())
  row_col_blk = calculate_zero_box_num(transform_row_col)
  col_row_blk = calculate_zero_box_num(transform_col_row)

#  print 'original matrix', np.count_nonzero(matA), calculate_zero_box_num(matA,(2,2))
#  print 'permuting rows', np.count_nonzero(transformed_mat), calculate_zero_box_num(transformed_mat, (2,2))
#  print 'permuting cols', np.count_nonzero(transform_col.transpose()), calculate_zero_box_num(transform_col.transpose(), (2,2))
#  print 'rows then cols', np.count_nonzero(transform_row_col), calculate_zero_box_num(transform_row_col, (2,2))
#  print 'cols then rows', np.count_nonzero(transform_col_row), calculate_zero_box_num(transform_col_row, (2,2))

  print 'original matrix', matA_nnz, matA_blk
  print 'permuting rows ', permute_row_nnz, permute_row_blk, (matA_blk - permute_row_blk) * 100/float(matA_blk)
  print 'permuting cols ', permute_col_nnz, permute_col_blk, (matA_blk - permute_col_blk) * 100/float(matA_blk)
  print 'rows then cols ', row_col_nnz, row_col_blk, (matA_blk - row_col_blk) * 100/float(matA_blk)
  print 'cols then rows ', col_row_nnz, col_row_blk, (matA_blk - col_row_blk) * 100/float(matA_blk)


  mp.matshow(matA)
  mp.title("matA")
  mp.matshow(transformed_mat)
  mp.title('transformed_mat')
  mp.matshow(transform_col.transpose())
  mp.title('transform_mat_by_col')
  mp.matshow(transform_row_col.transpose())
  mp.title('row_col')
  mp.matshow(transform_col_row.transpose())
  mp.title('col_row')
  mp.show()

if __name__ == '__main__':
  main()


