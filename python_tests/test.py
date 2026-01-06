from pathlib import Path
import torch
from scipy import sparse
import numpy
import gala_model

DATA_PATH = Path("../Data/Cora")

def load_adj_matrix() -> sparse.csr_matrix:
  adj_src = numpy.load(
    DATA_PATH / "Adj_src.npy"
  )

  adj_dst = numpy.load(
    DATA_PATH / "Adj_dst.npy"
  )

  adj_row_ids = adj_src[2:]
  adj_col_ids = adj_dst[0:]
  data = [1] * len(adj_row_ids)

  return sparse.csr_matrix(
    (data, (adj_row_ids, adj_col_ids))
  )

def prepare_adj_matrix(m: sparse.csr_matrix):
  vals = torch.tensor(m.data, dtype=torch.float32)
  cols = torch.tensor(m.indices, dtype=torch.int32)
  offsets = torch.tensor(m.indptr, dtype=torch.int32)
  # Is this the schedule?
  return gala_model.prepare(
    m.shape[0],
    m.shape[1],
    m.size, 
    vals, 
    cols, 
    offsets
  )

breakpoint()
adj_mtx_sparse = load_adj_matrix()
g = prepare_adj_matrix(adj_mtx_sparse)

# # These are row major ..?
# input_emb = numpy.load(DATA_PATH / "Feat.npy")
# emb_size = input_emb.shape[1]

# labels = numpy.load(DATA_PATH / "Lab.npy")
# train_mask_load = numpy.load(DATA_PATH / "TnMsk.npy")
# valid_mask_load = numpy.load(DATA_PATH / "VlMsk.npy")
# test_mask_load = numpy.load(DATA_PATH / "TsMsk.npy")

# classes = labels.max()