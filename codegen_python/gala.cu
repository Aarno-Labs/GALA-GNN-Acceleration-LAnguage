#include "gala.h"

#define CUDA_CHECK(func)\
  do {\
    cudaError_t status = (func);\
    if (status != cudaSuccess) {\
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,\
             cudaGetErrorString(status), status);\
      exit(EXIT_FAILURE);\
    }\
  } while (0)\

#define CUSPARSE_CHECK(func)\
  do {\
    cusparseStatus_t status = (func);\
    if (status != CUSPARSE_STATUS_SUCCESS) {\
      printf("CUSPARSE failed at line %d with error: %s (%d)\n", __LINE__,\
             cusparseGetErrorString(status), status);\
      exit(EXIT_FAILURE);\
    }\
  } while (0)
extern "C" __global__ void __launch_bounds__(256)
    default_function_kernel_sddvv_mult_undir(
        float *__restrict__ C,           // output
        int *__restrict__ J_indptr_data, // index pointer
        float *__restrict__ A,           // input A
        float *__restrict__ B,           // Input B
        int *__restrict__ J_indices_data, int nrows) {
                if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) { // This is fine
                    for (int j = (int)threadIdx.x; // Not fine. This should increase by 32
                         j <
                         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
                          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
                         j += 32) {
                        C[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] =
                            ((A[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))] *
                              B[(J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                                                    ((int)threadIdx.y))])])]));
                         }
                }
            }

extern "C" __global__ void __launch_bounds__(256)
aggregate_node_mul_sum_coarse2_kernel0(float *__restrict__ C,
                    int *__restrict__ J_indptr_data,
                                float *__restrict__ A, float *__restrict__ B,
                    int *__restrict__ J_indices_data, int nrows,
                    int dcols) {
if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    float local0 = C[(((((((int)blockIdx.x) * 8)+ ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 32)) + ((int)threadIdx.x)) + 0)];
        for (int j = 0;
             j <
             (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
              J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
             ++j) {
            local0 = local0 +A[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                    ((int)threadIdx.y))])] * (B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) + 
                                                         ((int)threadIdx.y))])] * 
                      dcols) + (((int)blockIdx.y) * 32)) + ((int)threadIdx.x) + 0)]);
             }

C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
(((int)blockIdx.y) * 32)) +
((int)threadIdx.x) + 0)] = local0;
   }
}

extern "C" __global__ void __launch_bounds__(256)
aggregate_node_mul_sum_coarse2_kernel1(float *__restrict__ C,
                    int *__restrict__ J_indptr_data,
                                float *__restrict__ A, float *__restrict__ B,
                    int *__restrict__ J_indices_data, int nrows,
                    int dcols) {
if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    float local0 = C[(((((((int)blockIdx.x) * 8)+ ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x)) + 0)];
    float local1 = C[(((((((int)blockIdx.x) * 8)+ ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x)) + 32)];
        for (int j = 0;
             j <
             (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
              J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
             ++j) {
            local0 = local0 +A[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                    ((int)threadIdx.y))])] * (B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) + 
                                                         ((int)threadIdx.y))])] * 
                      dcols) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x) + 0)]);
            local1 = local1 +A[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                    ((int)threadIdx.y))])] * (B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) + 
                                                         ((int)threadIdx.y))])] * 
                      dcols) + (((int)blockIdx.y) * 64)) + ((int)threadIdx.x) + 32)]);
             }

C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
(((int)blockIdx.y) * 64)) +
((int)threadIdx.x) + 0)] = local0;

C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
(((int)blockIdx.y) * 64)) +
((int)threadIdx.x) + 32)] = local1;
   }
}

extern "C" __global__ void __launch_bounds__(256)
aggregate_node_mul_sum_coarse2_kernel0_offset(float *__restrict__ C,
                    int *__restrict__ J_indptr_data,
                                float *__restrict__ A, float *__restrict__ B,
                    int *__restrict__ J_indices_data, int nrows,
                    int dcols, int offset) {
if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {
    float local0 = C[(((((((int)blockIdx.x) * 8)+ ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * 32)) + ((int)threadIdx.x)) + 0) + offset];
for (int j = 0;
 j <
 (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -
  J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);
 ++j) {
local0 = local0 +A[(j + J_indptr_data[((((int)blockIdx.x) * 8) +
                                    ((int)threadIdx.y))])] * (B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) + 
                                             ((int)threadIdx.y))])] * 
          dcols) + (((int)blockIdx.y) * 32)) + ((int)threadIdx.x) + 0) + offset]);
             }

C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +
(((int)blockIdx.y) * 32)) +
((int)threadIdx.x) + 0) + offset] = local0;
   }
}


torch::Tensor aggregate_edge_mul(torch::Tensor input_dense1,
                               torch::Tensor input_dense2,
                               torch::Tensor offset_graph,
                               torch::Tensor columns_graph,
                               torch::Tensor value_graph, torch::Tensor bounds,
                               int nrows,
                               int segments) {
  auto nvals = columns_graph.numel();
  auto full_iden = input_dense1.numel();
  auto dcols = full_iden / nrows;
  // // Dense
  // Input
  float *iden_ptr1 = input_dense1.data_ptr<float>();
  float *iden_ptr2 = input_dense2.data_ptr<float>();
  // Output
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat)
                     .requires_grad(true)
                     .device(torch::kCUDA, 0);
  auto output_sparse = torch::zeros({nvals}, options);
  float *oden_array = output_sparse.data_ptr<float>();
  // Sparse
  int *offset_ptr = offset_graph.data_ptr<int>();
  int *col_ptr = columns_graph.data_ptr<int>();
  float *val_ptr = value_graph.data_ptr<float>();
  int *bounds_ptr = bounds.data_ptr<int>();
  // cudaStream_t stream1;
  // cudaStreamCreate(&stream1);
  // dim3 gridDim(((int)(nrows - 1) / 8) + 1);
  // dim3 blockDim(32, 8);
  // default_function_kernel_sddvv_plus<<<gridDim, blockDim, 0, stream1>>>(
  //     oden_array, offset_ptr, iden_ptr1, iden_ptr2, col_ptr, nrows);
  std::vector<cudaStream_t> streams;
  for (int i = 0; i < segments; i++) {
    int i1 = i;
    int start_vals = bounds_ptr[i1 * 2];
    int end_vals = bounds_ptr[i1 * 2 + 1];
    int nvals = end_vals - start_vals;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    streams.push_back(stream);
    dim3 gridDim(((int)(nrows - 1) / 8) + 1);
    dim3 blockDim(32, 8);
    default_function_kernel_sddvv_mult_undir<<<gridDim, blockDim, 0,
                                               stream>>>(
        &oden_array[start_vals], &offset_ptr[i1 * (nrows + 1)], iden_ptr1,
        iden_ptr2, &col_ptr[start_vals], nrows);
  }
  for (auto& s : streams) { cudaStreamSynchronize(s); }
  for (auto& s : streams) { cudaStreamDestroy(s); }
  return output_sparse;
}

torch::Tensor aggregate_edge_mul_dir(torch::Tensor input_dense1,
                               torch::Tensor input_dense2,
                               torch::Tensor offset_graph,
                               torch::Tensor columns_graph,
                               torch::Tensor value_graph,
                               int nrows) {
  auto nvals = columns_graph.numel();
  auto full_iden = input_dense1.numel();
  auto dcols = full_iden / nrows;
  // // Dense
  // Input
  float *iden_ptr1 = input_dense1.data_ptr<float>();
  float *iden_ptr2 = input_dense2.data_ptr<float>();
  // Output
  auto options = torch::TensorOptions()
                     .dtype(torch::kFloat)
                     .requires_grad(true)
                     .device(torch::kCUDA, 0);
  auto output_sparse = torch::zeros({nvals}, options);
  float *oden_array = output_sparse.data_ptr<float>();
  // Sparse
  int *offset_ptr = offset_graph.data_ptr<int>();
  int *col_ptr = columns_graph.data_ptr<int>();
  float *val_ptr = value_graph.data_ptr<float>();
    cudaStream_t stream1, stream2, stream3;
      cudaStreamCreate(&stream1);
      dim3 gridDim(((int)(nrows - 1) / 8) + 1);
      dim3 blockDim(32, 8);
      default_function_kernel_sddvv_mult_undir<<<gridDim, blockDim, 0,
                                                 stream1>>>(
          oden_array, offset_ptr, iden_ptr1,
          iden_ptr2, col_ptr, nrows);
      cudaStreamSynchronize(stream1);
      cudaStreamDestroy(stream1);
  return output_sparse;
}

torch::Tensor aggregate_node_mul_sum_coarse2_call(torch::Tensor input_dense,
                   torch::Tensor offset_graph,
                   torch::Tensor columns_graph,
                   torch::Tensor value_graph
, torch::Tensor bounds,
 int nrows, int segments) {
auto nvals = columns_graph.numel();
auto full_iden = input_dense.numel();
auto dcols = full_iden / nrows;
// // Dense
// Input
float *iden_ptr = input_dense.data_ptr<float>();
// Output
auto options = torch::TensorOptions()
         .dtype(torch::kFloat)
         .requires_grad(true)
         .device(torch::kCUDA, 0);
auto output_dense = torch::zeros({nrows, dcols}, options);
float *oden_array = output_dense.data_ptr<float>();
// Sparse
int *offset_ptr = offset_graph.data_ptr<int>();
int *col_ptr = columns_graph.data_ptr<int>();
float *val_ptr = value_graph.data_ptr<float>();
std::vector<cudaStream_t> streams;
int *bounds_ptr = bounds.data_ptr<int>();
for (int i = 0; i < segments; i++) {
  int i1 = i;
  int start_vals = bounds_ptr[i1 * 2];cudaStream_t stream0, stream1, stream2;
  if ((int)dcols / 64) {
    cudaStreamCreate(&stream2);
    streams.push_back(stream2);
    dim3 gridDim(((int)(nrows - 1) / 8) + 1, (int)dcols / 64);
        dim3 blockDim(32, 8);
    aggregate_node_mul_sum_coarse2_kernel1<<<gridDim, blockDim, 0, stream2>>>(
        oden_array, &offset_ptr[i1 * (nrows + 1)],&val_ptr[start_vals],  iden_ptr, &col_ptr[start_vals], nrows, dcols);
  if ((dcols % 64 ) > 32) {
    cudaStreamCreate(&stream1);
    streams.push_back(stream1);
    dim3 gridDim(((int)(nrows - 1) / 8) + 1, 1);
    dim3 blockDim(32, 8);
    aggregate_node_mul_sum_coarse2_kernel0_offset<<<gridDim, blockDim, 0, stream1>>>(
        oden_array, &offset_ptr[i1 * (nrows + 1)],&val_ptr[start_vals],  iden_ptr, &col_ptr[start_vals], nrows, dcols, ((int)dcols /64) * 64);
  if ((dcols % 32 ) > 0) {
    cudaStreamCreate(&stream0);
    streams.push_back(stream0);
    dim3 gridDim(((int)(nrows - 1) / 8) + 1, 1);
    dim3 blockDim(dcols %32, 8);
    aggregate_node_mul_sum_coarse2_kernel0_offset<<<gridDim, blockDim, 0, stream0>>>(
        oden_array, &offset_ptr[i1 * (nrows + 1)],&val_ptr[start_vals],  iden_ptr, &col_ptr[start_vals], nrows, dcols, ((int)dcols /32) * 32);
  }
  }
else {
  if ((dcols % 64 ) > 0) {
    cudaStreamCreate(&stream0);
    streams.push_back(stream0);
    dim3 gridDim(((int)(nrows - 1) / 8) + 1, 1);
    dim3 blockDim(dcols %64, 8);
    aggregate_node_mul_sum_coarse2_kernel0_offset<<<gridDim, blockDim, 0, stream0>>>(
        oden_array, &offset_ptr[i1 * (nrows + 1)],&val_ptr[start_vals],  iden_ptr, &col_ptr[start_vals], nrows, dcols, ((int)dcols /64) * 64);
  }
}
  }
else {
  if ((int)dcols / 32) {
    cudaStreamCreate(&stream1);
    streams.push_back(stream1);
    dim3 gridDim(((int)(nrows - 1) / 8) + 1, (int)dcols / 32);
        dim3 blockDim(32, 8);
    aggregate_node_mul_sum_coarse2_kernel0<<<gridDim, blockDim, 0, stream1>>>(
        oden_array, &offset_ptr[i1 * (nrows + 1)],&val_ptr[start_vals],  iden_ptr, &col_ptr[start_vals], nrows, dcols);
  if ((dcols % 32 ) > 0) {
    cudaStreamCreate(&stream0);
    streams.push_back(stream0);
    dim3 gridDim(((int)(nrows - 1) / 8) + 1, 1);
    dim3 blockDim(dcols %32, 8);
    aggregate_node_mul_sum_coarse2_kernel0_offset<<<gridDim, blockDim, 0, stream0>>>(
        oden_array, &offset_ptr[i1 * (nrows + 1)],&val_ptr[start_vals],  iden_ptr, &col_ptr[start_vals], nrows, dcols, ((int)dcols /32) * 32);
  }
  }
else {
  if ((int)dcols) {
    cudaStreamCreate(&stream0);
    streams.push_back(stream0);
    dim3 gridDim(((int)(nrows - 1) / 8) + 1, 1);
        dim3 blockDim((int)dcols, 8);
    aggregate_node_mul_sum_coarse2_kernel0<<<gridDim, blockDim, 0, stream0>>>(
        oden_array, &offset_ptr[i1 * (nrows + 1)],&val_ptr[start_vals],  iden_ptr, &col_ptr[start_vals], nrows, dcols);
  }
}
}
}for (auto& s : streams) { cudaStreamSynchronize(s); }
for (auto& s : streams) { cudaStreamDestroy(s); }
return output_dense;
}
class aggregate_node_mul_sum_coarse2_AutoGrad : public torch::autograd::Function<aggregate_node_mul_sum_coarse2_AutoGrad> {
    public:
        static torch::Tensor forward(torch::autograd::AutogradContext *ctx,
                                     GALAGNN *gnn,
                                     torch::Tensor input_dense, int li) {
            torch::Tensor offset_graph = gnn->global_offset_graph[2 * li];
            torch::Tensor columns_graph = gnn->global_columns_graph[2 * li];
            torch::Tensor value_graph = gnn->global_value_graph[2 * li];
ctx->save_for_backward({gnn->global_offset_graph[2*li + 1], gnn->global_columns_graph[2*li + 1], gnn->global_value_graph[2*li + 1], gnn->global_bounds[2*li + 1]});ctx->saved_data["segments"] = gnn->global_segments[2*li + 1];ctx->saved_data["nrows"] = gnn->global_nrows;        torch::Tensor bounds = gnn->global_bounds[2 * li];
            int segments = gnn->global_segments[2 * li];
             return aggregate_node_mul_sum_coarse2_call(input_dense, offset_graph, columns_graph,
                                value_graph, bounds, gnn->global_nrows, segments);
    }
    
        static torch::autograd::tensor_list
        backward(torch::autograd::AutogradContext *ctx,
                 torch::autograd::tensor_list grad_outputs) {
            torch::Tensor input_dense = grad_outputs[0];
            auto saved = ctx->get_saved_variables();
            torch::Tensor offset_graph = saved[0];
            torch::Tensor columns_graph = saved[1];
            torch::Tensor value_graph = saved[2];
       int nrows = ctx->saved_data["nrows"].toInt();
        torch::Tensor bounds = saved[3];
            int segments = ctx->saved_data["segments"].toInt();
            return {torch::Tensor(), aggregate_node_mul_sum_coarse2_call(input_dense, offset_graph, columns_graph, value_graph, bounds, nrows, segments), torch::Tensor()};        }
    };
void GALAGNN::transform(SM& adj0, DB* train_mask){
  // Adj info
  iT nrows = adj0.nrows();
  iT ncols = adj0.ncols();
  nT nvals0 = adj0.nvals();
  global_nrows = nrows;
  std::vector<SM*> tiled_graph_tile;
  tiled_graph_tile.push_back(&adj0);
  torch::Tensor total_offsets_graph_tile;
  torch::Tensor total_cols_graph_tile;
  torch::Tensor total_vals_graph_tile;
  torch::Tensor total_bounds_graph_tile;
  std::vector<iT> tile_offsets_graph_tile = static_ord_col_breakpoints<SM>(&adj0, 100000.000000);
  iT segments_graph_tile = (tile_offsets_graph_tile.size()) - (1);
  total_offsets_graph_tile = torch::zeros(((1) + (adj0.nrows())) * (segments_graph_tile), options_int_tile);
  total_cols_graph_tile = torch::zeros(adj0.nvals(), options_int_tile);
  total_vals_graph_tile = torch::zeros(adj0.nvals(), options_float_tile);
  total_bounds_graph_tile = torch::zeros((2) * (segments_graph_tile), options_int_tile);
  ord_col_tiling_torch(tile_offsets_graph_tile, total_offsets_graph_tile, total_cols_graph_tile, total_vals_graph_tile, total_bounds_graph_tile, &adj0);
  iT* offset_ptr_graph_tile = total_offsets_graph_tile.data_ptr<iT>();
  iT* col_ptr_graph_tile = total_cols_graph_tile.data_ptr<iT>();
  vT* val_ptr_graph_tile = total_vals_graph_tile.data_ptr<vT>();
  global_segments.push_back(segments_graph_tile);
  global_bounds.push_back(total_bounds_graph_tile);
  global_segments.push_back(segments_graph_tile);
  global_bounds.push_back(total_bounds_graph_tile);
  // Allocate t_offsets0 on device
  int* dev_t_offsets0;
  CUDA_CHECK(cudaMalloc((void**)&dev_t_offsets0, ((nrows) + (1)) * (segments_graph_tile) * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dev_t_offsets0, offset_ptr_graph_tile, ((nrows) + (1)) * (segments_graph_tile) * sizeof(int), cudaMemcpyHostToDevice));
  torch::Tensor t_offsets0 = torch::from_blob(dev_t_offsets0, {((nrows) + (1)) * (segments_graph_tile)}, options_cu_int);
  // 
  // Allocate t_cols0 on device
  int* dev_t_cols0;
  CUDA_CHECK(cudaMalloc((void**)&dev_t_cols0, nvals0 * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(dev_t_cols0, col_ptr_graph_tile, nvals0 * sizeof(int), cudaMemcpyHostToDevice));
  torch::Tensor t_cols0 = torch::from_blob(dev_t_cols0, {nvals0}, options_cu_int);
  // 
  // Allocate t_vals0 on device
  float* dev_t_vals0;
  CUDA_CHECK(cudaMalloc((void**)&dev_t_vals0, nvals0 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dev_t_vals0, val_ptr_graph_tile, nvals0 * sizeof(float), cudaMemcpyHostToDevice));
  torch::Tensor t_vals0 = torch::from_blob(dev_t_vals0, {nvals0}, options_cu_float_ngrad);
  // 
  global_offset_graph.push_back(t_offsets0);
  global_columns_graph.push_back(t_cols0);
  global_value_graph.push_back(t_vals0);
  global_offset_graph.push_back(t_offsets0);
  global_columns_graph.push_back(t_cols0);
  global_value_graph.push_back(t_vals0);
}

void GALAGNN::inv(){
                  torch::Tensor ones_val = torch::ones({global_nrows, 1}, options_ones_val);
                torch::Tensor offset_graph_ones_val = global_offset_graph[2 * 0];
                torch::Tensor columns_graph_ones_val = global_columns_graph[2 * 0];
                torch::Tensor value_graph_ones_val = global_value_graph[2 * 0];
                torch::Tensor bounds_ones_val = global_bounds[2 * 0];
                int segments_ones_val = global_segments[2 * 0];
                torch::Tensor degrees_val = aggregate_node_mul_sum_coarse2_call(ones_val, offset_graph_ones_val, columns_graph_ones_val,
                                    value_graph_ones_val, bounds_ones_val, global_nrows, segments_ones_val);
                torch::Tensor norm_val = torch::pow(degrees_val, -0.500000).detach();

          torch::Tensor offset_graph_vals = global_offset_graph[2 * 0];
            torch::Tensor columns_graph_vals = global_columns_graph[2 * 0];
            torch::Tensor value_graph_vals = global_value_graph[2 * 0];
        torch::Tensor bounds_vals = global_bounds[2 * 0];
        int segments_vals = global_segments[2 * 0];
torch::Tensor val = aggregate_edge_mul(norm_val, norm_val, offset_graph_vals, columns_graph_vals, value_graph_vals, bounds_vals, global_nrows, segments_vals).detach();
  global_value_graph[2 * 0] = val;
}

std::vector<torch::Tensor> GALAGNN::forward(torch::Tensor t_iden, int ep, int mod_v){
      torch::Tensor res;
      torch::Tensor val;
  res = fc0->forward(t_iden);
      if (ep % mod_v == 0) {
      res = aggregate_node_mul_sum_coarse2_AutoGrad::apply(this, res, 0);
    } else {
      res = aggregate_node_mul_sum_coarse2_AutoGrad::apply(this, res, 0);
    }
          res = torch::relu(res);
  res = fc1->forward(res);
  res = fc2->forward(res);
  return {res};
}
std::shared_ptr<GALAGNN> make(int nrows, int ncols, int nvals, torch::Tensor val, torch::Tensor col, torch::Tensor offsets, std::optional<torch::Tensor> train_mask, int size0, int size1, int size2, int size3){
  SM* a = new SM;
  a->import_mtx(nrows, ncols, nvals, col.data_ptr<iT>(), val.data_ptr<vT>(), offsets.data_ptr<nT>(), CSRC_TYPE::CSR);
  return std::make_shared<GALAGNN>(*a, nullptr, size0, size1, size2, size3);
}
PYBIND11_MODULE(gala_model, m) {
  torch::python::bind_module<GALAGNN>(m, "GALAGNN")
    .def("forward", &GALAGNN::forward);

  m.def("make", &make);
}
