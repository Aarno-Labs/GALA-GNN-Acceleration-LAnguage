from pathlib import Path
import torch
from scipy import sparse
import numpy
import gala_model

DATA_PATH = Path("../Data/Cora")
NFEATS=1433
LABS=7

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

def make(m: sparse.csr_matrix):
  vals = torch.tensor(m.data, dtype=torch.float32)
  cols = torch.tensor(m.indices, dtype=torch.int32)
  offsets = torch.tensor(m.indptr, dtype=torch.int32)
  # Is this the schedule?
  return gala_model.make(
    m.shape[0],
    m.shape[1],
    m.size, 
    vals, 
    cols, 
    offsets,
    None,
    NFEATS,
    32,
    LABS,
    LABS
  )

adj_mtx_sparse = load_adj_matrix()
nrows = adj_mtx_sparse.shape[0]
print(f"adj matrix shape: {adj_mtx_sparse.shape}")
m = make(adj_mtx_sparse)

# These are row major ..?
input_emb = torch.from_numpy(numpy.load(DATA_PATH / "Feat.npy"))
emb_size = input_emb.shape[1]

labels = torch.from_numpy(numpy.load(DATA_PATH / "Lab.npy"))
train_mask_load = torch.from_numpy(numpy.load(DATA_PATH / "TnMsk.npy")).type(dtype=torch.bool)
valid_mask_load = torch.from_numpy(numpy.load(DATA_PATH / "VlMsk.npy")).type(dtype=torch.bool)
test_mask_load  = torch.from_numpy(numpy.load(DATA_PATH / "TsMsk.npy")).type(dtype=torch.bool)

classes = labels.max()

# Move to device?
torch.cuda.set_device(0)
device = torch.device('cuda:0')
input_emb = input_emb.to(device=device)
labels = labels.to(device=device)
train_mask = train_mask_load.to(device=device)
valid_mask = valid_mask_load.to(device=device)
test_mask = test_mask_load.to(device=device)

m.to(device=device, dtype=None)

def check_against_test():
  torch.cuda.synchronize()
  prediction = m.forward(input_emb, 0, 1)[0]
  torch.cuda.synchronize()
  predict_validate = prediction[test_mask.reshape(nrows)]
  labels_validate = labels[test_mask.reshape(nrows)]
  criterion = torch.nn.CrossEntropyLoss()
  torch.cuda.synchronize()
  d_loss = criterion(predict_validate, labels_validate.reshape(labels_validate.shape[0]))
  torch.cuda.synchronize()
  print(d_loss)


where = -1
opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
t_reshape = train_mask.reshape(nrows)
labels_train = labels[t_reshape].reshape(140)
for epoch in range(500):
  opt.zero_grad()
  prediction = m.forward(input_emb, epoch, 1)[0]
  prediction_train = prediction[train_mask.reshape(nrows)]
  criterion = torch.nn.CrossEntropyLoss()
  d_loss = criterion(prediction_train, labels_train)
  prediction_train = None
  prediction = None
  d_loss.backward()
  opt.step()
  d_loss = None
  check_against_test()
