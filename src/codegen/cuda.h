//

//

#ifndef GNN_ACCELERATION_LANGUAGE_CUDA_H
#define GNN_ACCELERATION_LANGUAGE_CUDA_H

#include <unordered_set>
#include "common.h"

class CUDAGenerator : public CodeGenerator
{
public:
    CUDAGenerator(GALAContext* context, std::string& outputPath) : CodeGenerator(context, outputPath)
    {
    }

    void initCMake() override
    {
        std::string cmakeCudaBase = "cmake_minimum_required(VERSION 3.1 FATAL_ERROR)\n"
            "project(gala_cuda LANGUAGES CUDA CXX)\n"
            "set(CMAKE_CXX_COMPILER icpx)\n"
            "find_package(Torch REQUIRED)\n"
            "find_package(OpenMP)\n"
            "if (OPENMP_FOUND)\n"
            "    set(OpenMP_CXX_FLAGS \"-fopenmp\")\n"
            "    set(CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}\")\n"
            "    set(CMAKE_EXE_LINKER_FLAGS \"${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}\")\n"
            "else ()\n"
            "    message(FATAL_ERROR \"Need OpenMP\")\n"
            "endif ()\n"
            "include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})\n"
            "link_libraries(\"${TORCH_LIBRARIES}\" cudart cusparse)\n"
            "set(CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} -qopt-report=0  -march=native -xCORE-AVX512 -O3 -DICC -restrict\")\n"
            "set(CMAKE_EXE_LINKER_FLAGS \"${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed -lpthread -lm -ldl\")\n"
            "set(CMAKE_CXX_STANDARD_LIBRARIES \"${CMAKE_CXX_STANDARD_LIBRARIES} -lnuma\")\n"
            "if (CMAKE_CXX_COMPILER_ID STREQUAL GNU)\n"
            "# set(CMAKE_EXE_LINKER_FLAGS \"${CMAKE_EXE_LINKER_FLAGS} -lmkl_gnu_thread -lgomp\")\n"
            "    set(CMAKE_EXE_LINKER_FLAGS \"${CMAKE_EXE_LINKER_FLAGS} -lgomp\")\n"
            "elseif (CMAKE_CXX_COMPILER_ID STREQUAL IntelLLVM)\n"
            "    # set(CMAKE_EXE_LINKER_FLAGS \"${CMAKE_EXE_LINKER_FLAGS} -lmkl_intel_thread -liomp5\")\n"
            "    set(CMAKE_EXE_LINKER_FLAGS \"${CMAKE_EXE_LINKER_FLAGS} -liomp5\")\n"
            "endif ()\n"
            "include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})\n"
            "link_libraries(\"${TORCH_LIBRARIES}\" cudart cusparse)\n"
            "add_compile_options(-Xcompiler -fopenmp -march=native -O3)\n"
            "add_compile_definitions(GALA_TORCH)\n"
            "add_compile_definitions(GN_1)\n"
            "add_compile_definitions(PT_0)\n"
            "add_compile_definitions(ST_0)\n"
            "add_compile_definitions(A_ALLOC)";
        std::string cmakeExecutable = "add_executable(gala_model gala.cu)\n"
            "target_compile_features(gala_model PRIVATE cxx_std_14)";
        cmakeCode.addCode(cmakeCudaBase);
        cmakeCode.addCode(cmakeExecutable);
    }

    std::string coarsenedKernelCall(ComputeNode* cNode, int cFact, int prevLayer, int weighted = true, int skip = 0)
    {
        std::string res = "";
        bool isKernelSample = hasCOpt(cNode, SAMPLE_COPT) || hasCOpt(cNode, SAMPLE_DYNAMIC_COPT);

        // Add check for the next coarsening
        // Top layer just divide
        if (prevLayer == -1)
        {
            if (cFact != 0){
                res += "  if ((int)dcols / " + std::to_string(32 * (cFact)) + ") {\n";
            } else {
                res += "  if ((int)dcols) {\n";
            }
        } else
        {
            // Sub-layers - modulo division by the higher layer. Then, see if any remaining
            res += "  if ((dcols % " + std::to_string(32 * (cFact + 1 + skip)) + " ) > "
            + std::to_string(32 * (cFact)) + ") {\n";
        }

        res += "    cudaStreamCreate(&stream" + std::to_string(cFact) + ");\n";
        res += "    streams.push_back(stream" + std::to_string(cFact) + ");\n";

        if (prevLayer == -1)
        {
            if (cFact != 0){
                res += "    dim3 gridDim(((int)(nrows - 1) / 8) + 1, (int)dcols / "
                + std::to_string(32 * (cFact)) + ");\n\
        dim3 blockDim(32, 8);\n";
            } else {
                res += "    dim3 gridDim(((int)(nrows - 1) / 8) + 1, 1);\n\
        dim3 blockDim((int)dcols, 8);\n";
            }
        } else
        {
            res += "    dim3 gridDim(((int)(nrows - 1) / 8) + 1, 1);\n";
            if (cFact != 0)
            {
                res += "    dim3 blockDim(32, 8);\n";
            } else
            {
                res += "    dim3 blockDim(dcols %" + std::to_string(32 * (cFact + 1 + skip)) + ", 8);\n";
            }
        }

        std::string weightedStr = weighted ? "&val_ptr[start_vals], " : "";

        if (prevLayer == -1)
        {
            if (cFact != 0)
            {
                res += "    " + getKernelName(cNode) + "_kernel" + std::to_string(cFact - 1) + "<<<gridDim, blockDim, 0, stream"
                + std::to_string(cFact) + ">>>(\n\
        oden_array, &offset_ptr[i1 * (nrows + 1)]," + (weightedStr);
            } else
            {

                res += "    " + getKernelName(cNode) + "_kernel0<<<gridDim, blockDim, 0, stream0>>>(\n\
        oden_array, &offset_ptr[i1 * (nrows + 1)]," + (weightedStr);
            }
            res += " iden_ptr, &col_ptr[start_vals], nrows, dcols";
            if (isKernelSample){
                res += ", global_ra, global_rb";
            }
            res += ");\n";
                // TODO Continue from here to add a,b
        } else
        {
            if (cFact != 0)
            {
                res += "    " + getKernelName(cNode) + "_kernel" + std::to_string(cFact - 1) + "_offset<<<gridDim, blockDim, 0, stream"
                + std::to_string(cFact) + ">>>(\n\
        oden_array, &offset_ptr[i1 * (nrows + 1)]," + (weightedStr) +" iden_ptr, &col_ptr[start_vals], nrows, dcols, ((int)dcols /"
                + std::to_string(32 * (cFact + 1 + skip)) + ") * " + std::to_string(32 * (cFact + 1 + skip));
            } else
            {
                res += "    " + getKernelName(cNode) + "_kernel0_offset<<<gridDim, blockDim, 0, stream0>>>(\n\
        oden_array, &offset_ptr[i1 * (nrows + 1)]," + (weightedStr) + " iden_ptr, &col_ptr[start_vals], nrows, dcols, ((int)dcols /"
                + std::to_string(32 * (cFact + 1 + skip)) + ") * " + std::to_string(32 * (cFact + 1 + skip));
            }
            if (isKernelSample){
                res += ", global_ra, global_rb";
            }
            res += ");\n";
        }
        // Remainder
        if (cFact != 0)
        {
            res += coarsenedKernelCall(cNode, cFact - 1, cFact, weighted);
        }
        // else if (prevLayer == -1)
        // {
        //     res += coarsenedKernelCall(cNode, cFact, cFact, weighted);
        // }
        res += "  }\n";

        // This should be the path if no computation was done earlier
        if (cFact != 0 && prevLayer == -1)
        {
            res += "else {\n";
            res += coarsenedKernelCall(cNode, cFact - 1, -1, weighted);
            res += "}\n";
        } else if (cFact != 0 )
        {
            res += "else {\n";
            // std::cout << "ss:" <<  cFact - 1 << std::endl;
            res += coarsenedKernelCall(cNode, cFact - 1, 0, weighted, skip + 1);
            res += "}\n";
        }
        return res;
    }

    void generateCudaCodeForCNode(ComputeNode* cNode)
    {
        if (cNode->getOpType() == AGGREGATE_NODE)
        {
            
            // Get the input
            auto graphInput = cNode->getInput(1);
            auto graphInfo =  graphInput->getDataInfo();

            // Unweighted (Col tile or undirected is not necessary at the moment)
            bool isWeighted = true;
            if (!graphInfo->getWeighted())
            {
                isWeighted = false;
            }

            int maxCoarsening = 1;
            for (auto opt: *cNode->getOpts())
            {
                if (opt.first == COARSE_COPT)
                {
                    maxCoarsening = (int)opt.second;
                }
            }

            bool isColTile = hasDOpt(cNode->getInput(1), COL_TILE_DOPT);
            bool isKernelSample = hasCOpt(cNode, SAMPLE_COPT) || hasCOpt(cNode, SAMPLE_DYNAMIC_COPT);
            int nsamples;
            if (isKernelSample)
            {
                for (auto opt: *cNode->getOpts())
                {
                    if ((opt.first == SAMPLE_COPT) || (opt.first == SAMPLE_DYNAMIC_COPT))
                    {
                        nsamples = (int)opt.second;
                    }
                }
            }


            // std::cout << "CC:" << maxCoarsening << " " << isWeighted << " " << !(isColTile) << " " << !(isKernelSample) << std::endl;
            if (maxCoarsening == 1 && isWeighted && !(isColTile) && !(isKernelSample))
            {
                std::string aggrKernelCall = "torch::Tensor " + getKernelName(cNode) + "_call(torch::Tensor input_dense,\n\
                                 torch::Tensor offset_graph,\n\
                                 torch::Tensor columns_graph,\n\
                                 torch::Tensor value_graph) {\n\
  auto nrows = offset_graph.numel() - 1;\n\
  auto nvals = columns_graph.numel();\n\
  auto full_iden = input_dense.numel();\n\
  auto dcols = full_iden / nrows;\n\
  float *iden_ptr = input_dense.data_ptr<float>();\n\
  // Output\n\
  auto options = torch::TensorOptions()\n\
                     .dtype(torch::kFloat)\n\
                     .requires_grad(true)\n\
                     .device(torch::kCUDA, 0);\n\
  auto output_dense = torch::zeros({nrows, dcols}, options);\n\
  float *oden_array = output_dense.data_ptr<float>();\n\
  int *offset_ptr = offset_graph.data_ptr<int>();\n\
  int *col_ptr = columns_graph.data_ptr<int>();\n\
  float *val_ptr = value_graph.data_ptr<float>();\n\
\n\
  float alpha = 1.0f;\n\
  float beta = 1.0f;\n\
\n\
  // Create the sparse / dense objects\n\
  cusparseHandle_t handle = NULL;\n\
  cusparseSpMatDescr_t matA;\n\
  cusparseDnMatDescr_t matB, matC;\n\
  void *dBuffer = NULL;\n\
  size_t bufferSize = 0;\n\
\n\
  CUSPARSE_CHECK(cusparseCreate(&handle));\n\
  CUSPARSE_CHECK(cusparseCreateCsr(&matA, nrows, nrows, nvals, offset_ptr,\n\
                                   col_ptr, val_ptr, CUSPARSE_INDEX_32I,\n\
                                   CUSPARSE_INDEX_32I, // Need to change these\n\
                                   CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));\n\
  // Create dense matrix B\n\
  CUSPARSE_CHECK(cusparseCreateDnMat(&matB, nrows, dcols, dcols, iden_ptr,\n\
                                     CUDA_R_32F,\n\
                                     CUSPARSE_ORDER_ROW)); // changed\n\
  // Create dense matrix C\n\
  CUSPARSE_CHECK(cusparseCreateDnMat(&matC, nrows, dcols, dcols, oden_array,\n\
                                     CUDA_R_32F,\n\
                                     CUSPARSE_ORDER_ROW)); // changed\n\
\n\
  // allocate an external buffer if needed\n\
  CUSPARSE_CHECK(cusparseSpMM_bufferSize(\n\
      handle, CUSPARSE_OPERATION_NON_TRANSPOSE,\n\
      CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,\n\
      CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, &bufferSize));\n\
  CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));\n\
\n\
  CUSPARSE_CHECK(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,\n\
                              CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,\n\
                              matB, &beta, matC, CUDA_R_32F,\n\
                              CUSPARSE_SPMM_CSR_ALG2, dBuffer));\n\
\n\
  CUSPARSE_CHECK(cusparseDestroySpMat(matA));\n\
  CUSPARSE_CHECK(cusparseDestroyDnMat(matB));\n\
  CUSPARSE_CHECK(cusparseDestroyDnMat(matC));\n\
  CUSPARSE_CHECK(cusparseDestroy(handle));\n\
  CUDA_CHECK(cudaFree(dBuffer));\n\
\n\
  return output_dense;\n\
}\n";
                // Adding the kernel call and setting the name
                kernelCallCode.addCode(aggrKernelCall);
                cNode->setKernelName("gather_forward");
            } else
            {
                std::string kernelCodeStr = "";
            // TODO eventually change the 32, 8 sizes based on the configurations of the CIR
            //  The 64 here needs to be changed into something else if the blocksize.y is changed
            // TODO add the semiring selection here
            for (int cFact = 0; cFact < maxCoarsening; cFact++)
            {
                kernelCodeStr += "extern \"C\" __global__ void __launch_bounds__(256)\n"
                + getKernelName(cNode) + "_kernel" + std::to_string(cFact) + "(float *__restrict__ C,\n\
                    int *__restrict__ J_indptr_data,\n";

                if (isWeighted)
                {
                    kernelCodeStr += "                                float *__restrict__ A, ";
                }

                kernelCodeStr += "float *__restrict__ B,\n\
                    int *__restrict__ J_indices_data, int nrows,\n\
                    int dcols";
                if (isKernelSample){
                    kernelCodeStr += ", int ra, int rb";
                }
                kernelCodeStr += ") {\n\
if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {\n";

                // The local register storage
                for (int j = 0; j <= cFact; j++)
                {
                    kernelCodeStr += "    float local" + std::to_string(j) + " = C[(((((((int)blockIdx.x) * 8)\
+ ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * " + std::to_string(32 * (cFact + 1)) + ")) + ((int)threadIdx.x)) + " + std::to_string(32 * j) + ")];\n";
                }

                if (isKernelSample) {
                    kernelCodeStr += "\
        int jmax =\n\
        (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -\n\
         J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);\n\
    if (jmax > 0) {\n\
      for (int ji = 0; ji < " + std::to_string(nsamples) + "; ++ji) {\n\
        int j = (ra * ji + rb) % jmax;\n";
                } else {
                    kernelCodeStr += "\
        for (int j = 0;\n\
             j <\n\
             (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -\n\
              J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);\n\
             ++j) {\n";
                }

                            for (int j = 0; j <= cFact; j++)
                            {
                                kernelCodeStr += "\
            local" + std::to_string(j) + " = local" + std::to_string(j) + " +";

                                if (isWeighted)
                                {
                                    kernelCodeStr += "A[(j + J_indptr_data[((((int)blockIdx.x) * 8) +\n\
                                    ((int)threadIdx.y))])] * ";
                                }
                                kernelCodeStr += "(B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) + \n\
                                                         ((int)threadIdx.y))])] * \n\
                      dcols) + (((int)blockIdx.y) * " + std::to_string(32 * (cFact + 1)) + ")) + ((int)threadIdx.x) + " + std::to_string(32 * j) + ")]);\n";
                            }

                kernelCodeStr += "             }\n";
                for (int j = 0; j <= cFact; j++)
                {
                kernelCodeStr += "\n\
C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +\n\
(((int)blockIdx.y) * " + std::to_string(32 * (cFact + 1)) + ")) +\n\
((int)threadIdx.x) + " + std::to_string(32 * j) + ")] = local" + std::to_string(j) + ";\n";
                }

                if (isKernelSample){
                    kernelCodeStr += "     }\n";
                }
                kernelCodeStr += "   }\n}\n\n";
            }

            // Offsets
            for (int cFact = 0; cFact < maxCoarsening - 1; cFact++)
            {
                kernelCodeStr += "extern \"C\" __global__ void __launch_bounds__(256)\n"
                + getKernelName(cNode) + "_kernel" + std::to_string(cFact) + "_offset(float *__restrict__ C,\n\
                    int *__restrict__ J_indptr_data,\n";

                if (isWeighted)
                {
                    kernelCodeStr += "                                float *__restrict__ A, ";
                }

                kernelCodeStr += "float *__restrict__ B,\n\
                    int *__restrict__ J_indices_data, int nrows,\n\
                    int dcols, int offset";
                if (isKernelSample){
                    kernelCodeStr += ", int ra, int rb";
                }
                kernelCodeStr += ") {\n\
if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) {\n";

                // The local register storage
                for (int j = 0; j <= cFact; j++)
                {
                    kernelCodeStr += "    float local" + std::to_string(j) + " = C[(((((((int)blockIdx.x) * 8)\
+ ((int)threadIdx.y)) * dcols + (((int)blockIdx.y) * " + std::to_string(32 * (cFact + 1)) + ")) + ((int)threadIdx.x)) + "
                    + std::to_string(32 * j) + ") + offset];\n";
                }

                if (isKernelSample){
                    kernelCodeStr += "\
        int jmax =\n\
        (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -\n\
         J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);\n\
    if (jmax > 0) {\n\
      for (int ji = 0; ji < " + std::to_string(nsamples) + "; ++ji) {\n\
        int j = (ra * ji + rb) % jmax;\n";
                } else {
                    kernelCodeStr += "\
for (int j = 0;\n\
 j <\n\
 (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -\n\
  J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);\n\
 ++j) {\n";
                }

                for (int j = 0; j <= cFact; j++)
                {
                    kernelCodeStr += "\
local" + std::to_string(j) + " = local" + std::to_string(j) + " +";

                    if (isWeighted)
                    {
                        kernelCodeStr += "A[(j + J_indptr_data[((((int)blockIdx.x) * 8) +\n\
                                    ((int)threadIdx.y))])] * ";
                    }
                    kernelCodeStr += "(B[(((J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) + \n\
                                             ((int)threadIdx.y))])] * \n\
          dcols) + (((int)blockIdx.y) * " + std::to_string(32 * (cFact + 1)) + ")) + ((int)threadIdx.x) + "
                    + std::to_string(32 * j) + ") + offset]);\n";
                }

                kernelCodeStr += "             }\n";
                for (int j = 0; j <= cFact; j++)
                {
                kernelCodeStr += "\n\
C[((((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols +\n\
(((int)blockIdx.y) * " + std::to_string(32 * (cFact + 1)) + ")) +\n\
((int)threadIdx.x) + " + std::to_string(32 * j) + ") + offset] = local" + std::to_string(j) + ";\n";
                }

                if (isKernelSample){
                    kernelCodeStr += "     }\n";
                }

                kernelCodeStr += "   }\n}\n\n";
            }

            kernelCode.addCode(kernelCodeStr);

            // This is the kernel call
            std::string aggrKernelCall = ""
            "torch::Tensor " + getKernelName(cNode) + "_call(torch::Tensor input_dense,\n\
                   torch::Tensor offset_graph,\n\
                   torch::Tensor columns_graph,\n\
                   torch::Tensor value_graph\n";
            if (isColTile)
            {
                aggrKernelCall += ", torch::Tensor bounds,\n int nrows, int segments";
            }
            aggrKernelCall += ") {\n\
auto nvals = columns_graph.numel();\n\
auto full_iden = input_dense.numel();\n\
auto dcols = full_iden / nrows;\n\
// // Dense\n\
// Input\n\
float *iden_ptr = input_dense.data_ptr<float>();\n\
// Output\n\
auto options = torch::TensorOptions()\n\
         .dtype(torch::kFloat)\n\
         .requires_grad(true)\n\
         .device(torch::kCUDA, 0);\n\
auto output_dense = torch::zeros({nrows, dcols}, options);\n\
float *oden_array = output_dense.data_ptr<float>();\n\
// Sparse\n\
int *offset_ptr = offset_graph.data_ptr<int>();\n\
int *col_ptr = columns_graph.data_ptr<int>();\n\
float *val_ptr = value_graph.data_ptr<float>();\n";

            aggrKernelCall += "std::vector<cudaStream_t> streams;\n";
            if (isColTile)
            {
                aggrKernelCall += "int *bounds_ptr = bounds.data_ptr<int>();\n\
for (int i = 0; i < segments; i++) {\n\
  int i1 = i;\n\
  int start_vals = bounds_ptr[i1 * 2];";
            } else
            {
                aggrKernelCall += "int i1 = 0;\n\
int start_vals = 0;";
            }

            aggrKernelCall += "cudaStream_t ";
            for (int cFact = 0; cFact < maxCoarsening + 1; cFact++)
            {
                aggrKernelCall += "stream" + std::to_string(cFact);
                if (cFact < maxCoarsening)
                {
                    aggrKernelCall += ", ";
                }
            }
            aggrKernelCall += ";\n";

            aggrKernelCall += coarsenedKernelCall(cNode, maxCoarsening, -1, isWeighted);
            if (isColTile)
            {
                aggrKernelCall += "}";
            }
            aggrKernelCall += "for (auto& s : streams) { cudaStreamSynchronize(s); }\n\
for (auto& s : streams) { cudaStreamDestroy(s); }\n\
return output_dense;\n\
}";
            // Adding the kernel call and setting the name
            kernelCallCode.addCode(aggrKernelCall);
            cNode->setKernelName("gather_forward");
            }
        } else if (cNode->getOp() == NON_LNR_OP_SOFTMAX) {
            std::string kernelCodeStr = "extern \"C\" __global__ void __launch_bounds__(256)\n\
default_function_kernel_spmm_backward_sddmm_32_nln(\n\
    float *__restrict__ C, // Output dense\n\
    int *__restrict__ J_indptr_data,\n\
    float *__restrict__ A, // Input values\n\
    int *__restrict__ J_indices_data, int nrows) {\n\
  if (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < nrows) {\n\
    float local_C = 1e-12;\n\
    for (int j = 0;\n\
         j <\n\
         (J_indptr_data[(((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) + 1)] -\n\
          J_indptr_data[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);\n\
         ++j) {\n\
      local_C = (local_C + (A[(j + J_indptr_data[((((int)blockIdx.x) * 32) +\n\
                                                  ((int)threadIdx.x))])]));\n\
    }\n\
    C[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] =\n\
        C[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + local_C;\n\
  }\n\
}\n\
extern \"C\" __global__ void __launch_bounds__(256)\n\
default_function_kernel_softmax_sddvv_undir(\n\
    float *__restrict__ C,           // output (values)\n\
    int *__restrict__ J_indptr_data, // index pointer\n\
    float *__restrict__ A,           // input A\n\
    int *__restrict__ J_indices_data, int nrows) {\n\
    if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) { // This is fine\n\
        for (int j = (int)threadIdx.x; // Not fine. This should increase by 32\n\
             j <\n\
             (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -\n\
              J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);\n\
             j += 32) {\n\
            C[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] =\n\
                C[(j +\n\
                   J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] *\n\
                (A[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);\n\
             }\n\
    }\n\
}\n\
extern \"C\" __global__ void __launch_bounds__(256)\n\
default_function_kernel_mult_sddvv_undir(\n\
    float *__restrict__ C,           // output (values)\n\
    int *__restrict__ J_indptr_data, // index pointer\n\
    float *__restrict__ A,           // input A\n\
    int *__restrict__ J_indices_data, int nrows) {\n\
    if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) { // This is fine\n\
        for (int j = (int)threadIdx.x; // Not fine. This should increase by 32\n\
             j <\n\
             (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -\n\
              J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);\n\
             j += 32) {\n\
            C[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] =\n\
                C[(j +\n\
                   J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] *\n\
                (A[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);\n\
             }\n\
    }\n\
}";
            kernelCode.addCode(kernelCodeStr);

            std::string kernelCallCodeStr = "torch::Tensor node_spmv_backward_of_sddmm_nln(torch::Tensor offset_graph,\n\
                                          torch::Tensor columns_graph,\n\
                                          torch::Tensor value_graph,\n\
                                          torch::Tensor bounds, int nrows,\n\
                                          int segments) {\n\
  // Output\n\
  auto options = torch::TensorOptions()\n\
                     .dtype(torch::kFloat)\n\
                     .requires_grad(true)\n\
                     .device(torch::kCUDA, 0);\n\
  auto output_dense = torch::zeros({nrows, 1}, options);\n\
  float *oden_array = output_dense.data_ptr<float>();\n\
\n\
  // Sparse\n\
  int *offset_ptr = offset_graph.data_ptr<int>();\n\
  int *col_ptr = columns_graph.data_ptr<int>();\n\
  float *val_ptr = value_graph.data_ptr<float>();\n\
  int *bounds_ptr = bounds.data_ptr<int>();\n\
  std::vector<cudaStream_t> streams;\n\
\n\
  for (int i = 0; i < segments; i++) {\n\
    int i1 = i;\n\
    int start_vals = bounds_ptr[i1 * 2];\n\
\n\
    cudaStream_t stream;\n\
    cudaStreamCreate(&stream);\n\
    streams.push_back(stream);\n\
    dim3 gridDim_rem(((int)(nrows - 1) / 32) + 1);\n\
    dim3 blockDim_rem(32);\n\
    default_function_kernel_spmm_backward_sddmm_32_nln<<<gridDim_rem, blockDim_rem,\n\
                                                     0, stream>>>(\n\
        oden_array, &offset_ptr[i1 * (nrows + 1)], &val_ptr[start_vals],\n\
        &col_ptr[start_vals], nrows);\n\
  }\n\
  for (auto& s : streams) { cudaStreamSynchronize(s); }\n\
  for (auto& s : streams) { cudaStreamDestroy(s); }\n\
\n\
  return output_dense;\n\
}\n\
torch::Tensor inplace_softmax_sddvv(torch::Tensor row_val,\n\
                                    torch::Tensor offset_graph,\n\
                                    torch::Tensor columns_graph,\n\
                                    torch::Tensor value_graph,\n\
                                    torch::Tensor bounds, int nrows,\n\
                                    int segments) {\n\
    float *row_val_ptr = row_val.data_ptr<float>();\n\
    // Sparse\n\
    int *offset_ptr = offset_graph.data_ptr<int>();\n\
    int *col_ptr = columns_graph.data_ptr<int>();\n\
    float *val_ptr = value_graph.data_ptr<float>();\n\
    int *bounds_ptr = bounds.data_ptr<int>();\n\
    std::vector<cudaStream_t> streams;\n\
    for (int i = 0; i < segments; i++) {\n\
        int i1 = i;\n\
        int start_vals = bounds_ptr[i1 * 2];\n\
        // int end_vals = bounds_ptr[i1 * 2 + 1];\n\
        // int nvals = end_vals - start_vals;\n\
        cudaStream_t stream;\n\
        cudaStreamCreate(&stream);\n\
        streams.push_back(stream);\n\
        dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1);\n\
        dim3 blockDim_rem(32, 8);\n\
        default_function_kernel_softmax_sddvv_undir<<<gridDim_rem, blockDim_rem, 0,\n\
                                                      stream>>>(\n\
            &val_ptr[start_vals], &offset_ptr[i1 * (nrows + 1)], row_val_ptr,\n\
            &col_ptr[start_vals], nrows);\n\
    }\n\
    for (auto& s : streams) { cudaStreamSynchronize(s); }\n\
    for (auto& s : streams) { cudaStreamDestroy(s); }\n\
    return value_graph;\n\
}\n\
torch::Tensor inplace_softmax_sddvv_mult(torch::Tensor row_val,\n\
                                        torch::Tensor offset_graph,\n\
                                        torch::Tensor columns_graph,\n\
                                        torch::Tensor value_graph,\n\
                                        torch::Tensor bounds, int nrows,\n\
                                        int segments) {\n\
    float *row_val_ptr = row_val.data_ptr<float>();\n\
    // Sparse\n\
    int *offset_ptr = offset_graph.data_ptr<int>();\n\
    int *col_ptr = columns_graph.data_ptr<int>();\n\
    float *val_ptr = value_graph.data_ptr<float>();\n\
    int *bounds_ptr = bounds.data_ptr<int>();\n\
    std::vector<cudaStream_t> streams;\n\
    for (int i = 0; i < segments; i++) {\n\
        int i1 = i;\n\
        int start_vals = bounds_ptr[i1 * 2];\n\
        // int end_vals = bounds_ptr[i1 * 2 + 1];\n\
        // int nvals = end_vals - start_vals;\n\
        cudaStream_t stream;\n\
        cudaStreamCreate(&stream);\n\
        streams.push_back(stream);\n\
        dim3 gridDim_rem(((int)(nrows - 1) / 8) + 1);\n\
        dim3 blockDim_rem(32, 8);\n\
        default_function_kernel_mult_sddvv_undir<<<gridDim_rem, blockDim_rem, 0,\n\
                                                   stream>>>(\n\
            &val_ptr[start_vals], &offset_ptr[i1 * (nrows + 1)], row_val_ptr,\n\
            &col_ptr[start_vals], nrows);\n\
    }\n\
    for (auto& s : streams) { cudaStreamSynchronize(s); }\n\
    for (auto& s : streams) { cudaStreamDestroy(s); }\n\
    return value_graph;\n\
}";
            kernelCallCode.addCode(kernelCallCodeStr);
        } else if (cNode->getOp() == AGGREGATE_EDGE_SUM_OP) {
            std::string kernelCodeStr = "extern \"C\" __global__ void __launch_bounds__(256)\n\
    default_function_kernel_spmm_backward_sddmm_32_eaggr(\n\
        float *__restrict__ C, // Output dense\n\
        int *__restrict__ J_indptr_data,\n\
        float *__restrict__ A, // Input values\n\
        int *__restrict__ J_indices_data, int nrows) {\n\
  if (((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) < nrows) {\n\
    float local_C = 1e-12;\n\
    for (int j = 0;\n\
         j <\n\
         (J_indptr_data[(((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) + 1)] -\n\
          J_indptr_data[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))]);\n\
         ++j) {\n\
      local_C = (local_C + (A[(j + J_indptr_data[((((int)blockIdx.x) * 32) +\n\
                                                  ((int)threadIdx.x))])]));\n\
    }\n\
    C[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] =\n\
        C[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] + local_C;\n\
  }\n\
}\n\
extern \"C\" __global__ void __launch_bounds__(256)\n\
default_function_kernel_sddvv_plus_undir(\n\
    float *__restrict__ C,           // output\n\
    int *__restrict__ J_indptr_data, // index pointer\n\
    float *__restrict__ A,           // input A\n\
    float *__restrict__ B,           // Input B\n\
    int *__restrict__ J_indices_data, int nrows) {\n\
    if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) { // This is fine\n\
        for (int j = (int)threadIdx.x; // Not fine. This should increase by 32\n\
             j <\n\
             (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -\n\
              J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);\n\
             j += 32) {\n\
            C[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] =\n\
                (A[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))] +\n\
                 B[(J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +\n\
                                                       ((int)threadIdx.y))])])]);\n\
             }\n\
    }\n\
}\n\
extern \"C\" __global__ void __launch_bounds__(256)\n\
default_function_kernel_sddmm_mult_undir_shared(\n\
    float *__restrict__ C,           // output\n\
    int *__restrict__ J_indptr_data, // index pointer\n\
    float *__restrict__ A,           // input A\n\
    float *__restrict__ B,           // Input B\n\
    int *__restrict__ J_indices_data, int nrows, int dcols) {\n\
    extern __shared__ float shared_mem[];\n\
    if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) { // This is fine\n\
        for (int k = threadIdx.x; k < dcols; k += 32) {\n\
            if (k < dcols) {\n\
                shared_mem[k] =\n\
                    A[((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) * dcols + k];\n\
            }\n\
        }\n\
        __syncthreads();\n\
        for (int j = (int)threadIdx.x; // Not fine. This should increase by 32\n\
             j <\n\
             (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -\n\
              J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);\n\
             j += 32) {\n\
            float local_C = 0;\n\
            for (int k = 0; k < dcols; k++) {\n\
                local_C =\n\
                    local_C +\n\
                    ((shared_mem[k] *\n\
                      B[(J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +\n\
                                                            ((int)threadIdx.y))])]) *\n\
                            dcols +\n\
                        k]));\n\
            }\n\
            C[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] =\n\
                local_C;\n\
             }\n\
    }\n\
}\n";
            kernelCode.addCode(kernelCodeStr);

            std::string kernelCallCodeStr = "torch::Tensor node_spmv_backward_of_sddmm_eaggr(torch::Tensor offset_graph,\n\
                                          torch::Tensor columns_graph,\n\
                                          torch::Tensor value_graph,\n\
                                          torch::Tensor bounds, int nrows,\n\
                                          int segments) {\n\
  // Output\n\
  auto options = torch::TensorOptions()\n\
                     .dtype(torch::kFloat)\n\
                     .requires_grad(true)\n\
                     .device(torch::kCUDA, 0);\n\
  auto output_dense = torch::zeros({nrows, 1}, options);\n\
  float *oden_array = output_dense.data_ptr<float>();\n\
\n\
  // Sparse\n\
  int *offset_ptr = offset_graph.data_ptr<int>();\n\
  int *col_ptr = columns_graph.data_ptr<int>();\n\
  float *val_ptr = value_graph.data_ptr<float>();\n\
  int *bounds_ptr = bounds.data_ptr<int>();\n\
  std::vector<cudaStream_t> streams;\n\
\n\
  for (int i = 0; i < segments; i++) {\n\
    int i1 = i;\n\
    int start_vals = bounds_ptr[i1 * 2];\n\
\n\
    cudaStream_t stream;\n\
    cudaStreamCreate(&stream);\n\
    streams.push_back(stream);\n\
    dim3 gridDim_rem(((int)(nrows - 1) / 32) + 1);\n\
    dim3 blockDim_rem(32);\n\
    default_function_kernel_spmm_backward_sddmm_32_eaggr<<<gridDim_rem, blockDim_rem,\n\
                                                     0, stream>>>(\n\
        oden_array, &offset_ptr[i1 * (nrows + 1)], &val_ptr[start_vals],\n\
        &col_ptr[start_vals], nrows);\n\
  }\n\
  for (auto& s : streams) { cudaStreamSynchronize(s); }\n\
  for (auto& s : streams) { cudaStreamDestroy(s); }\n\
\n\
  return output_dense;\n\
}\n\
torch::Tensor edge_sddvv(torch::Tensor input_dense1, torch::Tensor input_dense2,\n\
torch::Tensor offset_graph,\n\
torch::Tensor columns_graph, torch::Tensor value_graph,\n\
torch::Tensor bounds, int nrows, int segments) {\n\
    auto nvals = columns_graph.numel();\n\
    // // Dense\n\
    // Input\n\
    float *iden_ptr1 = input_dense1.data_ptr<float>();\n\
    float *iden_ptr2 = input_dense2.data_ptr<float>();\n\
    // Output\n\
    auto options = torch::TensorOptions()\n\
                       .dtype(torch::kFloat)\n\
                       .requires_grad(true)\n\
                       .device(torch::kCUDA, 0);\n\
    auto output_sparse = torch::zeros({nvals}, options);\n\
    float *oden_array = output_sparse.data_ptr<float>();\n\
    // Sparse\n\
    int *offset_ptr = offset_graph.data_ptr<int>();\n\
    int *col_ptr = columns_graph.data_ptr<int>();\n\
    float *val_ptr = value_graph.data_ptr<float>();\n\
    int *bounds_ptr = bounds.data_ptr<int>();\n\
    std::vector<cudaStream_t> streams;\n\
    for (int i = 0; i < segments; i++) {\n\
        int i1 = i;\n\
        int start_vals = bounds_ptr[i1 * 2];\n\
        cudaStream_t stream;\n\
        cudaStreamCreate(&stream);\n\
        streams.push_back(stream);\n\
        dim3 gridDim(((int)(nrows - 1) / 8) + 1);\n\
        dim3 blockDim(32, 8);\n\
        default_function_kernel_sddvv_plus_undir<<<gridDim, blockDim, 0,\n\
                                                   stream>>>(\n\
            &oden_array[start_vals], &offset_ptr[i1 * (nrows + 1)], iden_ptr1,\n\
            iden_ptr2, &col_ptr[start_vals], nrows);\n\
    }\n\
    for (auto& s : streams) { cudaStreamSynchronize(s); }\n\
    for (auto& s : streams) { cudaStreamDestroy(s); }\n\
    return output_sparse;\n\
}\n\
torch::Tensor edge_sddmm(torch::Tensor input_dense1, torch::Tensor input_dense2,\n\
torch::Tensor offset_graph,\n\
torch::Tensor columns_graph, torch::Tensor value_graph,\n\
torch::Tensor bounds, int nrows, int segments) {\n\
    auto nvals = columns_graph.numel();\n\
    auto full_iden = input_dense1.numel();\n\
    auto dcols = full_iden / nrows;\n\
    // // Dense\n\
    // Input\n\
    float *iden_ptr1 = input_dense1.data_ptr<float>();\n\
    float *iden_ptr2 = input_dense2.data_ptr<float>();\n\
    // Output\n\
    auto options = torch::TensorOptions()\n\
                       .dtype(torch::kFloat)\n\
                       .requires_grad(true)\n\
                       .device(torch::kCUDA, 0);\n\
    auto output_sparse = torch::zeros({nvals}, options);\n\
    float *oden_array = output_sparse.data_ptr<float>();\n\
    // Sparse\n\
    int *offset_ptr = offset_graph.data_ptr<int>();\n\
    int *col_ptr = columns_graph.data_ptr<int>();\n\
    float *val_ptr = value_graph.data_ptr<float>();\n\
    int *bounds_ptr = bounds.data_ptr<int>();\n\
    std::vector<cudaStream_t> streams;\n\
    for (int i = 0; i < segments; i++) {\n\
        int i1 = i;\n\
        int start_vals = bounds_ptr[i1 * 2];\n\
        cudaStream_t stream;\n\
        cudaStreamCreate(&stream);\n\
        streams.push_back(stream);\n\
        dim3 gridDim(((int)(nrows - 1) / 8) + 1);\n\
        dim3 blockDim(32, 8);\n\
        int shared_memory_size = dcols * sizeof(float);\n\
        default_function_kernel_sddmm_mult_undir_shared<<<\n\
            gridDim, blockDim, shared_memory_size, stream>>>(\n\
            &oden_array[start_vals], &offset_ptr[i1 * (nrows + 1)], iden_ptr1,\n\
            iden_ptr2, &col_ptr[start_vals], nrows, dcols);\n\
    }\n\
    for (auto& s : streams) { cudaStreamSynchronize(s); }\n\
    for (auto& s : streams) { cudaStreamDestroy(s); }\n\
    return output_sparse;\n\
}\n";
            kernelCallCode.addCode(kernelCallCodeStr);
        } else if (cNode->getOp() == AGGREGATE_EDGE_MUL_OP) {
            std::string kernelCodeStr = "extern \"C\" __global__ void __launch_bounds__(256)\n\
    default_function_kernel_sddvv_mult_undir(\n\
        float *__restrict__ C,           // output\n\
        int *__restrict__ J_indptr_data, // index pointer\n\
        float *__restrict__ A,           // input A\n\
        float *__restrict__ B,           // Input B\n\
        int *__restrict__ J_indices_data, int nrows) {\n\
                if (((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) < nrows) { // This is fine\n\
                    for (int j = (int)threadIdx.x; // Not fine. This should increase by 32\n\
                         j <\n\
                         (J_indptr_data[(((((int)blockIdx.x) * 8) + ((int)threadIdx.y)) + 1)] -\n\
                          J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))]);\n\
                         j += 32) {\n\
                        C[(j + J_indptr_data[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))])] =\n\
                            ((A[((((int)blockIdx.x) * 8) + ((int)threadIdx.y))] *\n\
                              B[(J_indices_data[(j + J_indptr_data[((((int)blockIdx.x) * 8) +\n\
                                                                    ((int)threadIdx.y))])])]));\n\
                         }\n\
                }\n\
            }\n";
            kernelCode.addCode(kernelCodeStr);

            std::string kernelCallCodeStr = "torch::Tensor aggregate_edge_mul(torch::Tensor input_dense1,\n\
                               torch::Tensor input_dense2,\n\
                               torch::Tensor offset_graph,\n\
                               torch::Tensor columns_graph,\n\
                               torch::Tensor value_graph, torch::Tensor bounds,\n\
                               int nrows,\n\
                               int segments) {\n\
  auto nvals = columns_graph.numel();\n\
  auto full_iden = input_dense1.numel();\n\
  auto dcols = full_iden / nrows;\n\
  // // Dense\n\
  // Input\n\
  float *iden_ptr1 = input_dense1.data_ptr<float>();\n\
  float *iden_ptr2 = input_dense2.data_ptr<float>();\n\
  // Output\n\
  auto options = torch::TensorOptions()\n\
                     .dtype(torch::kFloat)\n\
                     .requires_grad(true)\n\
                     .device(torch::kCUDA, 0);\n\
  auto output_sparse = torch::zeros({nvals}, options);\n\
  float *oden_array = output_sparse.data_ptr<float>();\n\
  // Sparse\n\
  int *offset_ptr = offset_graph.data_ptr<int>();\n\
  int *col_ptr = columns_graph.data_ptr<int>();\n\
  float *val_ptr = value_graph.data_ptr<float>();\n\
  int *bounds_ptr = bounds.data_ptr<int>();\n\
  // cudaStream_t stream1;\n\
  // cudaStreamCreate(&stream1);\n\
  // dim3 gridDim(((int)(nrows - 1) / 8) + 1);\n\
  // dim3 blockDim(32, 8);\n\
  // default_function_kernel_sddvv_plus<<<gridDim, blockDim, 0, stream1>>>(\n\
  //     oden_array, offset_ptr, iden_ptr1, iden_ptr2, col_ptr, nrows);\n\
  std::vector<cudaStream_t> streams;\n\
  for (int i = 0; i < segments; i++) {\n\
    int i1 = i;\n\
    int start_vals = bounds_ptr[i1 * 2];\n\
    int end_vals = bounds_ptr[i1 * 2 + 1];\n\
    int nvals = end_vals - start_vals;\n\
    cudaStream_t stream;\n\
    cudaStreamCreate(&stream);\n\
    streams.push_back(stream);\n\
    dim3 gridDim(((int)(nrows - 1) / 8) + 1);\n\
    dim3 blockDim(32, 8);\n\
    default_function_kernel_sddvv_mult_undir<<<gridDim, blockDim, 0,\n\
                                               stream>>>(\n\
        &oden_array[start_vals], &offset_ptr[i1 * (nrows + 1)], iden_ptr1,\n\
        iden_ptr2, &col_ptr[start_vals], nrows);\n\
  }\n\
  for (auto& s : streams) { cudaStreamSynchronize(s); }\n\
  for (auto& s : streams) { cudaStreamDestroy(s); }\n\
  return output_sparse;\n\
}\n";
            kernelCallCode.addCode(kernelCallCodeStr);
            std::string kernelCallCodeStr2 = "torch::Tensor aggregate_edge_mul_dir(torch::Tensor input_dense1,\n\
                               torch::Tensor input_dense2,\n\
                               torch::Tensor offset_graph,\n\
                               torch::Tensor columns_graph,\n\
                               torch::Tensor value_graph,\n\
                               int nrows) {\n\
  auto nvals = columns_graph.numel();\n\
  auto full_iden = input_dense1.numel();\n\
  auto dcols = full_iden / nrows;\n\
  // // Dense\n\
  // Input\n\
  float *iden_ptr1 = input_dense1.data_ptr<float>();\n\
  float *iden_ptr2 = input_dense2.data_ptr<float>();\n\
  // Output\n\
  auto options = torch::TensorOptions()\n\
                     .dtype(torch::kFloat)\n\
                     .requires_grad(true)\n\
                     .device(torch::kCUDA, 0);\n\
  auto output_sparse = torch::zeros({nvals}, options);\n\
  float *oden_array = output_sparse.data_ptr<float>();\n\
  // Sparse\n\
  int *offset_ptr = offset_graph.data_ptr<int>();\n\
  int *col_ptr = columns_graph.data_ptr<int>();\n\
  float *val_ptr = value_graph.data_ptr<float>();\n\
    cudaStream_t stream1, stream2, stream3;\n\
      cudaStreamCreate(&stream1);\n\
      dim3 gridDim(((int)(nrows - 1) / 8) + 1);\n\
      dim3 blockDim(32, 8);\n\
      default_function_kernel_sddvv_mult_undir<<<gridDim, blockDim, 0,\n\
                                                 stream1>>>(\n\
          oden_array, offset_ptr, iden_ptr1,\n\
          iden_ptr2, col_ptr, nrows);\n\
      cudaStreamSynchronize(stream1);\n\
      cudaStreamDestroy(stream1);\n\
  return output_sparse;\n\
}\n";
            kernelCallCode.addCode(kernelCallCodeStr2);
        }
    }

    void initKernels(std::vector<CIRNode*>& program) override
    {
        std::string importBase = "#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.\n"
            "#include <cusparse.h>\n"
            "#include <torch/script.h>\n"
            "#include <cmath>\n"
            "#include <iostream>\n"
            "#include <parallel/algorithm>\n"
            "#include <vector>\n"
            "#include <bits/stdc++.h>\n"
            "#include <omp.h>\n"
            "#include <stdlib.h>\n"
            "#include <torch/torch.h>\n"
            "#include \"../src/formats/csrc_matrix.h\"\n"
            "#include \"../src/formats/dense_matrix.h\"\n"
            "#include \"../src/ops/aggregators.h\"\n"
            "#include \"../src/ops/tiling.h\"\n"
            "#include \"../src/utils/mtx_io.h\"\n"
            "#include \"../include/common.h\"\n";
        importCode.addCode(importBase);


        std::string cudaInitFunctions = "\n"
"#define CUDA_CHECK(func)\\\n\
  do {\\\n\
    cudaError_t status = (func);\\\n\
    if (status != cudaSuccess) {\\\n\
      printf(\"CUDA API failed at line %d with error: %s (%d)\\n\", __LINE__,\\\n\
             cudaGetErrorString(status), status);\\\n\
      exit(EXIT_FAILURE);\\\n\
    }\\\n\
  } while (0)\\\n\
\n\
#define CUSPARSE_CHECK(func)\\\n\
  do {\\\n\
    cusparseStatus_t status = (func);\\\n\
    if (status != CUSPARSE_STATUS_SUCCESS) {\\\n\
      printf(\"CUSPARSE failed at line %d with error: %s (%d)\\n\", __LINE__,\\\n\
             cusparseGetErrorString(status), status);\\\n\
      exit(EXIT_FAILURE);\\\n\
    }\\\n\
  } while (0)";

        if (GALAFEContext::print_memory)
        {
            cudaInitFunctions += "\n\
int printMemoryUsage() {\n\
                size_t freeMem, totalMem;\n\
                cudaMemGetInfo(&freeMem, &totalMem);\n\
                return (int)((totalMem - freeMem) / (1024 * 1024));\n\
            }\n";
        }
        kernelCode.addCode(cudaInitFunctions);

        std::unordered_set<std::string> encountedOps;
        for (int i = 0; i < program.size(); i++)
        {
            // std::cout << "Works0.0" << std::endl;
            CIRNode* outNode = program[i];
            // if (outNode == nullptr){
            //     std::cout << "This is null" << std::endl;
            // }
            // std::cout << "Works0.0.0" << std::endl;
            auto oNode = dynamic_cast<ComputeNode*>(outNode);
            // std::cout << "Works0.1" << oNode->getOp() << std::endl;
            if (oNode)
            {
                // std::cout << "Works0.2.0" << std::endl;
                auto cNode = dynamic_cast<ComputeNode*>(outNode);
                // std::cout << "Works0.2" << std::endl;
                std::string kernelName = getKernelName(cNode);
                // std::cout << kernelName << std::endl;

                if (encountedOps.find(kernelName) == encountedOps.end())
                {
                    generateCudaCodeForCNode(cNode);
                    encountedOps.insert(kernelName);
                }
            } else {
                // std::cout << "Works0.3.0" << std::endl;
                auto loopNode = dynamic_cast<TrainingLoopNode*>(outNode);
                for (int ix = 0; ix < loopNode->getLoopNodeNum(); ix++)
                {
                    CIRNode* inNode = loopNode->getNode(ix);
                    auto cNode = dynamic_cast<ComputeNode*>(inNode);
                    std::string kernelName = getKernelName(cNode);
                    if (encountedOps.find(kernelName) == encountedOps.end())
                    {
                        generateCudaCodeForCNode(cNode);
                        encountedOps.insert(kernelName);
                    }
                }
            }
        }
    }

    std::tuple<std::string, std::string, std::string>
    generateCudaTransfer(bool isColTile, int indexData, std::string dataName, std::string suffix)
    {
        std::string t_offsets, t_cols, t_vals;
        if (isColTile)
        {
            t_offsets = tensorFromBlob(model.getTransform()->getCode(),
                                       "int",
                                       "t_offsets"+std::to_string(indexData)+suffix,
                                       "offset_ptr_"+dataName+suffix,
                                       { Code::binOp("*", Code::binOp("+", "nrows", "1"), "segments_"+dataName+suffix) },
                                       "options_cu_int");
            t_cols = tensorFromBlob(model.getTransform()->getCode(),
                                    "int",
                                    "t_cols"+std::to_string(indexData)+suffix,
                                    "col_ptr_"+dataName+suffix,
                                    { "nvals"+std::to_string(indexData) },
                                    "options_cu_int");
            t_vals = tensorFromBlob(model.getTransform()->getCode(),
                                    "float",
                                    "t_vals"+std::to_string(indexData)+suffix,
                                    "val_ptr_"+dataName+suffix,
                                    { "nvals"+std::to_string(indexData) },
                                    "options_cu_float_ngrad");
        } else
        {
            t_offsets = tensorFromBlob(model.getTransform()->getCode(),
                                       "int",
                                       "t_offsets"+std::to_string(indexData)+suffix,
                                       Code::callMethod("adj"+std::to_string(indexData)+suffix, "offset_ptr"),
                                       { Code::binOp("+", "nrows", "1") },
                                       "options_cu_int");
            t_cols = tensorFromBlob(model.getTransform()->getCode(),
                                       "int",
                                       "t_cols"+std::to_string(indexData)+suffix,
                                       Code::callMethod("adj"+std::to_string(indexData)+suffix, "ids_ptr"),
                                       { "nvals"+std::to_string(indexData) },
                                       "options_cu_int");
            t_vals = tensorFromBlob(model.getTransform()->getCode(),
                                       "float",
                                       "t_vals"+std::to_string(indexData)+suffix,
                                       Code::callMethod("adj"+std::to_string(indexData)+suffix, "vals_ptr"),
                                       { "nvals"+std::to_string(indexData) },
                                       "options_cu_float_ngrad");
        }
        model.getTransform()->getCode()->expr(Code::callMethod("global_offset_graph", "push_back", { t_offsets}));
        model.getTransform()->getCode()->expr(Code::callMethod("global_columns_graph", "push_back", { t_cols }));
        model.getTransform()->getCode()->expr(Code::callMethod("global_value_graph", "push_back", { t_vals }));

        return std::tuple(t_offsets, t_cols, t_vals);
    }

    void generateCudaTransferCodeForUniqueInput(ComputeNode* cNode,
        std::unordered_set<std::string> &encounteredStrings, bool &defaultLoaded)
    {
        // TODO need to the same for the backward pass' data
        // Add BOTH precode and postcode
        for (int inpI = 0; inpI < cNode->getNumInputs(); inpI++)
        {
            auto inputData = cNode->getInput(inpI);

            if (!defaultLoaded)
            {
                auto inputInfo =  inputData->getDataInfo();

                // std::cout << inputData->getName() << " :a: " << inputInfo->getDefaultName()  << " " << inputInfo->getDefaultIndex() << " " << inputInfo->getIndex() << std::endl;

                if (!(inputInfo->getIndex() <= 0)){
                    std::string dataName;
                    if (inputInfo->getDefaultName() == ""){
                        dataName = inputData->getName();
                    } else {
                        dataName = inputInfo->getDefaultName();
                    }
                    // std::cout << inputData->getName() << " :das" << std::endl;

                    if (encounteredStrings.find(dataName) == encounteredStrings.end())
                    {
                        if (inputInfo->getFormat() == CSR_STYPE)
                        {
                            // std::cout << inputData->getName() << " cas" << std::endl;

                            defaultLoaded = true;

                            int indexData = inputInfo->getDefaultIndex();
                            encounteredStrings.insert(dataName);

                            //std::cout << dataName << " " << indexData << std::endl;

                            bool isColTile = hasDOpt(inputData, COL_TILE_DOPT);
                            std::string t_offsets, t_cols, t_vals;
                            std::tie(t_offsets, t_cols, t_vals) = generateCudaTransfer(isColTile, indexData, dataName, "");

                            // These are graphs for backprop
                            if (!inputInfo->getDefaultDirected())
                            {
                                model.getTransform()->getCode()->expr(Code::callMethod("global_offset_graph", "push_back", { t_offsets}));
                                model.getTransform()->getCode()->expr(Code::callMethod("global_columns_graph", "push_back", { t_cols }));
                                model.getTransform()->getCode()->expr(Code::callMethod("global_value_graph", "push_back", { t_vals }));
                            } else
                            {
                                generateCudaTransfer(isColTile, indexData, dataName, "_b");
                            }
                        }
                    }
                }
            }

            // Check if the string has been encountered before
            if (encounteredStrings.find(inputData->getName()) == encounteredStrings.end()) {

                // For now only generate the transfer code for CSR type graphs
                auto inputInfo =  inputData->getDataInfo();
                if (inputInfo->getFormat() == CSR_STYPE && !inputInfo->getDerived())
                {
                    // TODO Check if this is a dependant of an exising graph
                    int indexData = (int)encounteredStrings.size();
                    if (inputInfo->getIndex() != -1)
                    {
                        indexData = inputInfo->getIndex();
                    }
                    encounteredStrings.insert(inputData->getName());

                    // TODO Temp fix
                    if (inputData->getName() == "attn" || inputData->getName() == "val")
                    {
                        continue;
                    }

                    inputInfo->setIndex(indexData);

                    bool isColTile = hasDOpt(inputData, COL_TILE_DOPT);
                    std::string t_offsets, t_cols, t_vals;
                    std::tie(t_offsets, t_cols, t_vals) = generateCudaTransfer(isColTile, indexData, inputData->getName(), "");

                    // These are graphs for backprop
                    if (!inputInfo->getDefaultDirected())
                    {
                        model.getTransform()->getCode()->expr(Code::callMethod("global_offset_graph", "push_back", { t_offsets}));
                        model.getTransform()->getCode()->expr(Code::callMethod("global_columns_graph", "push_back", { t_cols }));
                        model.getTransform()->getCode()->expr(Code::callMethod("global_value_graph", "push_back", { t_vals }));
                    } else
                    {
                        generateCudaTransfer(isColTile, indexData, inputData->getName(), "_b");
                    }
                }
            }
        }
    }

    // You don't need transfer operations if it is produced by some computation operation
    void cudaTransfer(std::vector<CIRNode*>& program)
    {
        std::unordered_set<std::string> encounteredStrings;
        bool defaultLoaded = false;
        // Iterate through the entire program
        for (int i = 0; i < program.size(); i++)
        {
            CIRNode* outNode = program[i];
            auto oNode = dynamic_cast<ComputeNode*>(outNode);
            if (oNode)
            {
                generateCudaTransferCodeForUniqueInput(oNode, encounteredStrings, defaultLoaded);
            } else
            {
                auto loopNode = dynamic_cast<TrainingLoopNode*>(outNode);
                for (int ix = 0; ix < loopNode->getLoopNodeNum(); ix++)
                {
                    CIRNode* inNode = loopNode->getNode(ix);
                    auto cNode = dynamic_cast<ComputeNode*>(inNode);
                    generateCudaTransferCodeForUniqueInput(cNode, encounteredStrings, defaultLoaded);
                }
            }
        }
    }

    void dataPrep(std::vector<CIRNode*>& program) override
    {
        // TODO make the transfer based on the data and the transformations applied
        // Add the graph parts to a vector
        mainBuilder.getCode()->declare_cstr_init("torch::Device", "device", "torch::kCUDA");
        importCode.declare("const torch::TensorOptions", "options_cu_int", "torch::TensorOptions().dtype(torch::kInt).requires_grad(false).device(torch::kCUDA, 0)");
        importCode.declare("const torch::TensorOptions", "options_cu_float_grad", "torch::TensorOptions().dtype(torch::kFloat).requires_grad(true).device(torch::kCUDA, 0)");
        importCode.declare("const torch::TensorOptions", "options_cu_float_ngrad", "torch::TensorOptions().dtype(torch::kFloat).requires_grad(false).device(torch::kCUDA, 0)");
        importCode.declare("const torch::TensorOptions", "options_cu_bool", "torch::TensorOptions().dtype(torch::kBool).requires_grad(false).device(torch::kCUDA, 0)");
        importCode.declare("const torch::TensorOptions", "options_cu_long", "torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA, 0)");

        // For now there's no slicing for a dense input so just use a hard-coded code-generation.
        auto t_iden       = tensorFromBlob(mainBuilder.getCode(), "float", "t_iden", Code::callMethod("input_emb", "vals_ptr"), { "nrows", "emb_size" }, "options_cu_float_grad");
        auto t_labs       = tensorFromBlob(mainBuilder.getCode(), "long", "t_labs", Code::callMethod("labels", "vals_ptr"), { "nrows" }, "options_cu_long");
        auto t_train_mask = tensorFromBlob(mainBuilder.getCode(), "bool", "t_train_mask", Code::callMethod("train_mask", "vals_ptr"), { "nrows" }, "options_cu_bool");
        auto t_valid_mask = tensorFromBlob(mainBuilder.getCode(), "bool", "t_valid_mask", Code::callMethod("valid_mask", "vals_ptr"), { "nrows" }, "options_cu_bool");
        auto t_test_mask  = tensorFromBlob(mainBuilder.getCode(), "bool", "t_test_mask", Code::callMethod("test_mask", "vals_ptr"), { "nrows" }, "options_cu_bool");

        cudaTransfer(program);

        // std::string cleanCuda = " CUDA_CHECK(cudaFree(dB));";
        // postCode.addCode(cleanCuda);
    }
private:
    static std::string tensorFromBlob(Code *builder,
                                        std::string type,
                                        std::string var,
                                        std::string blob,
                                        std::vector<std::string> dim,
                                        std::string options)
    {
        // TODO: It should be possible to simply construct a tensor and send it to device
        // with Tensor::to(), but in quick tests this seemed to result in worse
        // inference performance
        builder->comment("Allocate " + var + " on device");
        auto devPtr = builder->declare(type + "*", "dev_" + var);
        auto size = Code::intersperse(" * ", dim) + " * sizeof(" + type + ")";
        builder->expr(Code::callFn("CUDA_CHECK", { Code::callFn("cudaMalloc", {"(void**)&"+devPtr, size }) }));
        builder->expr(Code::callFn("CUDA_CHECK", { Code::callFn("cudaMemcpy", {devPtr, blob, size, "cudaMemcpyHostToDevice"}) }));
        auto from_blob = Code::callFn("torch::from_blob", { devPtr, Code::vec(dim), options });
        auto tensor = builder->declare("torch::Tensor", var, from_blob);
        builder->comment("");

        return tensor;
    }
};

#endif //GNN_ACCELERATION_LANGUAGE_CUDA_H
