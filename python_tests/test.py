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
    1433,
    32,
    7,
    7
  )

adj_mtx_sparse = load_adj_matrix()
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
device = torch.device('cuda:0')
input_emb = input_emb.to(device=device)
labels = labels.to(device=device)
train_mask = train_mask_load.to(device=device)
valid_mask = valid_mask_load.to(device=device)
test_mask = test_mask_load.to(device=device)

m.to(device=device, dtype=None)

opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
for epoch in range(5000):
  torch.cuda.synchronize()
  opt.zero_grad()
  torch.cuda.synchronize()
  prediction = m.forward(input_emb, epoch, 1)[0]
  torch.cuda.synchronize()
  prediction_train = prediction[train_mask.reshape(2708)]
  labels_train = labels[train_mask.reshape(2708)]
  criterion = torch.nn.CrossEntropyLoss()
  d_loss = criterion(prediction_train, labels_train.reshape(140))
  torch.cuda.synchronize()
  d_loss.backward()
  torch.cuda.synchronize()
  opt.step()

prediction = m.forward(input_emb, 10000, 1)[0]
predict_validate = prediction[valid_mask.reshape(2708)]
labels_validate = labels[valid_mask.reshape(2708)]
criterion = torch.nn.CrossEntropyLoss()
d_loss = criterion(predict_validate, labels_validate.reshape(labels_validate.shape[0]))
print(d_loss)
