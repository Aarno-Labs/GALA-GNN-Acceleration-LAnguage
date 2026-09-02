#ifndef GNN_ACCELERATION_LANGUAGE_CPU_H
#define GNN_ACCELERATION_LANGUAGE_CPU_H

// CPU (x86-64 / OpenMP / AVX2) code-generation backend.
//
// Everything device-neutral (dense LibTorch ops, autograd class templates,
// training loop, loss, evaluator, host-side preprocessing and column tiling)
// is emitted by CodeGenerator; this backend supplies the build recipe, the
// sparse kernels, and the data placement. Generated programs are compiled
// with -march=x86-64-v3 (AVX2/FMA/BMI2) so the same binary runs on Intel
// hybrid parts (which lack AVX-512) and AMD hosts alike.

#include "common.h"
#include "cuda.h"
#include <memory>

class CPUGenerator : public CodeGenerator
{
public:
    CPUGenerator(GALAContext* context, std::string& outputPath, std::string dataRoot)
        : CodeGenerator(context, outputPath, dataRoot) {}

    // ---- backend hooks ----
    std::string deviceOpt() override { return ""; }
    std::string syncCall() override { return ""; }
    std::string modelFileName() override { return "gala.cpp"; }

    std::string transposePermToDevice(const std::string& name, const std::string& nvalsExpr) override
    {
        // perm_data_<name> is a block-scoped std::vector<int>: copy it into a tensor.
        return "    torch::Tensor t_perm_" + name + " = torch::from_blob(perm_data_" + name + ".data(),\n"
               "        {(int64_t)" + nvalsExpr + "}, torch::TensorOptions().dtype(torch::kInt)).clone();\n";
    }

    // The tiled CSR is already held in host torch tensors (total_*_<name>); the untiled
    // CSR lives in the CSRCMatrix adj<index>. Either way, no copies are needed.
    std::string graphTransferCode(int index, const std::string& name, bool isColTile, bool directed) override
    {
        std::string N = std::to_string(index);
        auto one = [&](const std::string& sfx) {
            std::string c;
            if (isColTile)
            {
                c += "  torch::Tensor t_offsets" + N + sfx + " = total_offsets_" + name + sfx + ";\n"
                     "  torch::Tensor t_cols" + N + sfx + " = total_cols_" + name + sfx + ";\n"
                     "  torch::Tensor t_vals" + N + sfx + " = total_vals_" + name + sfx + ";\n";
            } else
            {
                c += "  torch::Tensor t_offsets" + N + sfx + " = torch::from_blob(adj" + N + sfx + ".offset_ptr(), {nrows + 1}, options_cu_int);\n"
                     "  torch::Tensor t_cols" + N + sfx + " = torch::from_blob(adj" + N + sfx + ".ids_ptr(), {nvals" + N + "}, options_cu_int);\n"
                     "  torch::Tensor t_vals" + N + sfx + " = torch::from_blob(adj" + N + sfx + ".vals_ptr(), {nvals" + N + "}, options_cu_float_ngrad);\n";
            }
            c += "  global_offset_graph.push_back(t_offsets" + N + sfx + ");\n"
                 "  global_columns_graph.push_back(t_cols" + N + sfx + ");\n"
                 "  global_value_graph.push_back(t_vals" + N + sfx + ");\n";
            return c;
        };
        std::string code = one("");
        if (!directed)
        {
            code += "  global_offset_graph.push_back(t_offsets" + N + ");\n"
                    "  global_columns_graph.push_back(t_cols" + N + ");\n"
                    "  global_value_graph.push_back(t_vals" + N + ");\n";
        } else
        {
            code += one("_b");
        }
        return code;
    }

    void initCMake() override
    {
        std::string cmakeBase = "cmake_minimum_required(VERSION 3.18 FATAL_ERROR)\n"
            "set(CMAKE_CXX_COMPILER /usr/bin/g++ CACHE FILEPATH \"GALA host compiler\")\n"
            "project(gala_cpu LANGUAGES CXX)\n"
            "set(CMAKE_CXX_STANDARD 17)\n"
            "find_package(Torch REQUIRED)\n"
            "find_package(OpenMP REQUIRED)\n"
            "if (NOT DEFINED GALA)\n"
            "  set(GALA \"${PROJECT_SOURCE_DIR}/..\")\n"
            "endif()\n"
            "message(\"GALA Sources at ${GALA}\")\n"
            "include_directories(\"${GALA}/include\")\n"
            "link_libraries(\"${TORCH_LIBRARIES}\" OpenMP::OpenMP_CXX)\n"
            // x86-64-v3 = AVX2/FMA/BMI2: the portable common denominator of the Intel
            // evaluation CPU (no AVX-512) and AMD development hosts. GALA_CPU_MARCH can
            // override it (e.g. native) for builds done on the target machine.
            "if (NOT DEFINED GALA_CPU_MARCH)\n"
            "  set(GALA_CPU_MARCH x86-64-v3)\n"
            "endif()\n"
            "set(CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -O3 -march=${GALA_CPU_MARCH} -fopenmp -DICC\")\n"
            "set(CMAKE_EXE_LINKER_FLAGS \"${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed -lpthread -lm -ldl -lgomp\")\n"
            "set(CMAKE_CXX_STANDARD_LIBRARIES \"${CMAKE_CXX_STANDARD_LIBRARIES} -lnuma\")\n"
            "add_compile_definitions(GALA_TORCH)\n"
            "add_compile_definitions(GALA_CPU)\n"
            "add_compile_definitions(GN_1)\n"
            "add_compile_definitions(PT_0)\n"
            "add_compile_definitions(ST_0)\n"
            "add_compile_definitions(A_ALLOC)";
        std::string cmakeExecutable = "add_executable(gala_model gala.cpp)\n"
            "target_compile_features(gala_model PRIVATE cxx_std_17)";
        cmakeCode.addCode(cmakeBase);
        cmakeCode.addCode(cmakeExecutable);
    }

    void initKernels(std::vector<CIRNode*>& program) override
    {
        std::string importBase = "#include <torch/script.h>\n"
            "#include <cmath>\n"
            "#include <iostream>\n"
            "#include <parallel/algorithm>\n"
            "#include <vector>\n"
            "#include <bits/stdc++.h>\n"
            "#include <omp.h>\n"
            "#include <stdlib.h>\n"
            "#include <torch/torch.h>\n"
            "#include <formats/csrc_matrix.h>\n"
            "#include <formats/dense_matrix.h>\n"
            "#include <ops/aggregators.h>\n"
            "#include <ops/tiling.h>\n"
            "#include <utils/mtx_io.h>\n"
            "#include <tests/common.h>\n"
            "#include <codegen/evaluator.h>\n";
        importCode.addCode(importBase);

        if (GALAFEContext::print_memory)
        {
            std::string memFn = "\nint printMemoryUsage() {\n"
                "  long pages = 0, rss = 0;\n"
                "  FILE *f = fopen(\"/proc/self/statm\", \"r\");\n"
                "  if (f) { if (fscanf(f, \"%ld %ld\", &pages, &rss) != 2) rss = 0; fclose(f); }\n"
                "  return (int)(rss * (sysconf(_SC_PAGESIZE) / 1024) / 1024);\n"
                "}\n";
            kernelCode.addCode(memFn);
        }

        std::unordered_set<std::string> encountedOps;
        auto visit = [&](ComputeNode* cNode) {
            std::string kernelName = getKernelName(cNode);
            if (encountedOps.find(kernelName) == encountedOps.end())
            {
                generateCPUCodeForCNode(cNode);
                encountedOps.insert(kernelName);
            }
        };
        for (int i = 0; i < program.size(); i++)
        {
            CIRNode* outNode = program[i];
            auto oNode = dynamic_cast<ComputeNode*>(outNode);
            if (oNode)
            {
                visit(oNode);
            } else
            {
                auto loopNode = dynamic_cast<TrainingLoopNode*>(outNode);
                for (int ix = 0; ix < loopNode->getLoopNodeNum(); ix++)
                {
                    visit(dynamic_cast<ComputeNode*>(loopNode->getNode(ix)));
                }
            }
        }
    }

    // out[i] = sum over segments of (1e-12 + sum of edge values in row i), matching the
    // CUDA row-sum kernels' per-segment seeding.
    std::string rowSumKernel(const std::string& name)
    {
        return "torch::Tensor " + name + "(torch::Tensor offset_graph,\n"
            "                                          torch::Tensor columns_graph,\n"
            "                                          torch::Tensor value_graph,\n"
            "                                          torch::Tensor bounds, int nrows,\n"
            "                                          int segments) {\n"
            "  auto options = torch::TensorOptions().dtype(torch::kFloat).requires_grad(true);\n"
            "  auto output_dense = torch::zeros({nrows, 1}, options);\n"
            "  float *oden_array = output_dense.data_ptr<float>();\n"
            "  const int *offset_ptr = offset_graph.data_ptr<int>();\n"
            "  const float *val_ptr = value_graph.data_ptr<float>();\n"
            "  const int *bounds_ptr = bounds.data_ptr<int>();\n"
            "  for (int t = 0; t < segments; t++) {\n"
            "    const int *off = offset_ptr + (int64_t)t * (nrows + 1);\n"
            "    const float *vals = val_ptr + bounds_ptr[2 * t];\n"
            "#pragma omp parallel for schedule(dynamic, 256)\n"
            "    for (int i = 0; i < nrows; i++) {\n"
            "      float acc = 1e-12f;\n"
            "      for (int e = off[i]; e < off[i + 1]; e++) acc += vals[e];\n"
            "      oden_array[i] += acc;\n"
            "    }\n"
            "  }\n"
            "  return output_dense;\n"
            "}\n";
    }

    // In place: vals[e] *= row_val[row(e)]; returns value_graph.
    std::string rowScaleKernel(const std::string& name)
    {
        return "torch::Tensor " + name + "(torch::Tensor row_val,\n"
            "                                    torch::Tensor offset_graph,\n"
            "                                    torch::Tensor columns_graph,\n"
            "                                    torch::Tensor value_graph,\n"
            "                                    torch::Tensor bounds, int nrows,\n"
            "                                    int segments) {\n"
            "  torch::Tensor r_c = row_val.contiguous();\n"
            "  const float *row_val_ptr = r_c.data_ptr<float>();\n"
            "  const int *offset_ptr = offset_graph.data_ptr<int>();\n"
            "  float *val_ptr = value_graph.data_ptr<float>();\n"
            "  const int *bounds_ptr = bounds.data_ptr<int>();\n"
            "  for (int t = 0; t < segments; t++) {\n"
            "    const int *off = offset_ptr + (int64_t)t * (nrows + 1);\n"
            "    float *vals = val_ptr + bounds_ptr[2 * t];\n"
            "#pragma omp parallel for schedule(dynamic, 256)\n"
            "    for (int i = 0; i < nrows; i++) {\n"
            "      const float r = row_val_ptr[i];\n"
            "      for (int e = off[i]; e < off[i + 1]; e++) vals[e] *= r;\n"
            "    }\n"
            "  }\n"
            "  return value_graph;\n"
            "}\n";
    }

    // Sparse kernels. Each host wrapper keeps the exact name and signature the
    // device-neutral autograd templates in common.h call.
    void generateCPUCodeForCNode(ComputeNode* cNode)
    {
        if (cNode->getOpType() == AGGREGATE_NODE)
        {
            auto graphInfo = cNode->getInput(1)->getDataInfo();
            bool isWeighted = graphInfo->getWeighted();
            bool isColTile = hasDOpt(cNode->getInput(1), COL_TILE_DOPT);

            // SpMM: out[i,:] += sum_{e in row i} val[e] * X[col[e],:] over every column
            // segment of the tiled CSR (segment t's offsets start at offset_ptr[t*(nrows+1)]
            // and its column/value entries at bounds_ptr[2t]). Rows are independent, so
            // the row loop is parallel with no atomics; the feature loop vectorizes.
            std::string call = "torch::Tensor " + getKernelName(cNode) + "_call(torch::Tensor input_dense,\n"
                "                   torch::Tensor offset_graph,\n"
                "                   torch::Tensor columns_graph,\n"
                "                   torch::Tensor value_graph";
            if (isColTile) call += ",\n                   torch::Tensor bounds,\n                   int segments";
            call += ") {\n"
                "  const int64_t nrows = global_nrows;\n"
                "  torch::Tensor input_c = input_dense.contiguous();\n"
                "  const int64_t dcols = input_c.numel() / nrows;\n"
                "  auto options = torch::TensorOptions().dtype(torch::kFloat).requires_grad(true);\n"
                "  auto output_dense = torch::zeros({nrows, dcols}, options);\n"
                "  const float *iden_ptr = input_c.data_ptr<float>();\n"
                "  float *oden_array = output_dense.data_ptr<float>();\n"
                "  const int *offset_ptr = offset_graph.data_ptr<int>();\n"
                "  const int *col_ptr = columns_graph.data_ptr<int>();\n"
                "  const float *val_ptr = value_graph.data_ptr<float>();\n";
            if (isColTile)
            {
                call += "  const int *bounds_ptr = bounds.data_ptr<int>();\n"
                        "  for (int t = 0; t < segments; t++) {\n"
                        "    const int *off = offset_ptr + (int64_t)t * (nrows + 1);\n"
                        "    const int *cols = col_ptr + bounds_ptr[2 * t];\n"
                        "    const float *vals = val_ptr + bounds_ptr[2 * t];\n";
            } else
            {
                call += "  {\n"
                        "    const int *off = offset_ptr;\n"
                        "    const int *cols = col_ptr;\n"
                        "    const float *vals = val_ptr;\n";
            }
            call += "#pragma omp parallel for schedule(dynamic, 64)\n"
                    "    for (int64_t i = 0; i < nrows; i++) {\n"
                    "      float *o = oden_array + i * dcols;\n"
                    "      for (int e = off[i]; e < off[i + 1]; e++) {\n"
                    "        const float *xr = iden_ptr + (int64_t)cols[e] * dcols;\n";
            if (isWeighted)
            {
                call += "        const float v = vals[e];\n"
                        "#pragma omp simd\n"
                        "        for (int64_t k = 0; k < dcols; k++) o[k] += v * xr[k];\n";
            } else
            {
                call += "        (void)vals;\n"
                        "#pragma omp simd\n"
                        "        for (int64_t k = 0; k < dcols; k++) o[k] += xr[k];\n";
            }
            call += "      }\n"
                    "    }\n"
                    "  }\n"
                    "  return output_dense;\n"
                    "}\n";
            kernelCallCode.addCode(call);
            cNode->setKernelName("gather_forward");
        } else if (cNode->getOp() == AGGREGATE_EDGE_MUL_OP)
        {
            // Per-edge product of two per-node scalars: out[e] = A[row(e)] * B[col(e)]
            // (degree-normalization folding). Segment t's edges live at
            // bounds[2t] + offset_ptr[t*(nrows+1) + i .. i+1).
            std::string body =
                "  const int64_t nrows = global_nrows;\n"
                "  const int64_t nvals = columns_graph.numel();\n"
                "  torch::Tensor a_c = input_dense1.contiguous();\n"
                "  torch::Tensor b_c = input_dense2.contiguous();\n"
                "  const float *iden_ptr1 = a_c.data_ptr<float>();\n"
                "  const float *iden_ptr2 = b_c.data_ptr<float>();\n"
                "  auto options = torch::TensorOptions().dtype(torch::kFloat).requires_grad(true);\n"
                "  auto output_sparse = torch::zeros({nvals}, options);\n"
                "  float *oden_array = output_sparse.data_ptr<float>();\n"
                "  const int *offset_ptr = offset_graph.data_ptr<int>();\n"
                "  const int *col_ptr = columns_graph.data_ptr<int>();\n";
            std::string loop =
                "#pragma omp parallel for schedule(dynamic, 256)\n"
                "    for (int64_t i = 0; i < nrows; i++) {\n"
                "      const float a = iden_ptr1[i];\n"
                "      for (int e = off[i]; e < off[i + 1]; e++) {\n"
                "        out[e] = a * iden_ptr2[cols[e]];\n"
                "      }\n"
                "    }\n";
            std::string tiled = "torch::Tensor aggregate_edge_mul(torch::Tensor input_dense1,\n"
                "                               torch::Tensor input_dense2,\n"
                "                               torch::Tensor offset_graph,\n"
                "                               torch::Tensor columns_graph,\n"
                "                               torch::Tensor value_graph, torch::Tensor bounds,\n"
                "                               int segments) {\n" + body +
                "  const int *bounds_ptr = bounds.data_ptr<int>();\n"
                "  for (int t = 0; t < segments; t++) {\n"
                "    const int *off = offset_ptr + (int64_t)t * (nrows + 1);\n"
                "    const int *cols = col_ptr + bounds_ptr[2 * t];\n"
                "    float *out = oden_array + bounds_ptr[2 * t];\n" + loop +
                "  }\n"
                "  return output_sparse;\n"
                "}\n";
            std::string untiled = "torch::Tensor aggregate_edge_mul_dir(torch::Tensor input_dense1,\n"
                "                               torch::Tensor input_dense2,\n"
                "                               torch::Tensor offset_graph,\n"
                "                               torch::Tensor columns_graph,\n"
                "                               torch::Tensor value_graph) {\n" + body +
                "  {\n"
                "    const int *off = offset_ptr;\n"
                "    const int *cols = col_ptr;\n"
                "    float *out = oden_array;\n" + loop +
                "  }\n"
                "  return output_sparse;\n"
                "}\n";
            kernelCallCode.addCode(tiled);
            kernelCallCode.addCode(untiled);
        } else if (cNode->getOp() == NON_LNR_OP_SOFTMAX)
        {
            // Edge softmax support: per-row max stabilization (in place), row sums of
            // edge values (seeded 1e-12 per segment, as the CUDA kernels do), and
            // in-place per-row scaling of edge values.
            std::string code =
                "void sparse_softmax_stabilize(torch::Tensor offset_graph,\n"
                "                                          torch::Tensor columns_graph,\n"
                "                                          torch::Tensor value_graph,\n"
                "                                          torch::Tensor bounds, int nrows,\n"
                "                                          int segments) {\n"
                "  std::vector<float> row_max((size_t)nrows, -1e30f);\n"
                "  const int *offset_ptr = offset_graph.data_ptr<int>();\n"
                "  float *val_ptr = value_graph.data_ptr<float>();\n"
                "  const int *bounds_ptr = bounds.data_ptr<int>();\n"
                "  for (int t = 0; t < segments; t++) {\n"
                "    const int *off = offset_ptr + (int64_t)t * (nrows + 1);\n"
                "    const float *vals = val_ptr + bounds_ptr[2 * t];\n"
                "#pragma omp parallel for schedule(dynamic, 256)\n"
                "    for (int i = 0; i < nrows; i++) {\n"
                "      float m = row_max[i];\n"
                "      for (int e = off[i]; e < off[i + 1]; e++) m = std::max(m, vals[e]);\n"
                "      row_max[i] = m;\n"
                "    }\n"
                "  }\n"
                "  for (int t = 0; t < segments; t++) {\n"
                "    const int *off = offset_ptr + (int64_t)t * (nrows + 1);\n"
                "    float *vals = val_ptr + bounds_ptr[2 * t];\n"
                "#pragma omp parallel for schedule(dynamic, 256)\n"
                "    for (int i = 0; i < nrows; i++) {\n"
                "      const float m = row_max[i];\n"
                "      for (int e = off[i]; e < off[i + 1]; e++) vals[e] -= m;\n"
                "    }\n"
                "  }\n"
                "}\n"
                + rowSumKernel("node_spmv_backward_of_sddmm_nln")
                + rowScaleKernel("inplace_softmax_sddvv")
                + rowScaleKernel("inplace_softmax_sddvv_mult");
            kernelCallCode.addCode(code);
        } else if (cNode->getOp() == AGGREGATE_EDGE_SUM_OP)
        {
            // Attention logits and their backward: per-edge sum of two per-node scalars,
            // per-edge dot product of two node feature rows, and row sums.
            std::string code = rowSumKernel("node_spmv_backward_of_sddmm_eaggr") +
                "torch::Tensor edge_sddvv(torch::Tensor input_dense1, torch::Tensor input_dense2,\n"
                "torch::Tensor offset_graph,\n"
                "torch::Tensor columns_graph, torch::Tensor value_graph,\n"
                "torch::Tensor bounds, int nrows, int segments) {\n"
                "  const int64_t nvals = columns_graph.numel();\n"
                "  torch::Tensor a_c = input_dense1.contiguous();\n"
                "  torch::Tensor b_c = input_dense2.contiguous();\n"
                "  const float *iden_ptr1 = a_c.data_ptr<float>();\n"
                "  const float *iden_ptr2 = b_c.data_ptr<float>();\n"
                "  auto options = torch::TensorOptions().dtype(torch::kFloat).requires_grad(true);\n"
                "  auto output_sparse = torch::zeros({nvals}, options);\n"
                "  float *oden_array = output_sparse.data_ptr<float>();\n"
                "  const int *offset_ptr = offset_graph.data_ptr<int>();\n"
                "  const int *col_ptr = columns_graph.data_ptr<int>();\n"
                "  const int *bounds_ptr = bounds.data_ptr<int>();\n"
                "  for (int t = 0; t < segments; t++) {\n"
                "    const int *off = offset_ptr + (int64_t)t * (nrows + 1);\n"
                "    const int *cols = col_ptr + bounds_ptr[2 * t];\n"
                "    float *out = oden_array + bounds_ptr[2 * t];\n"
                "#pragma omp parallel for schedule(dynamic, 256)\n"
                "    for (int i = 0; i < nrows; i++) {\n"
                "      const float a = iden_ptr1[i];\n"
                "      for (int e = off[i]; e < off[i + 1]; e++) out[e] = a + iden_ptr2[cols[e]];\n"
                "    }\n"
                "  }\n"
                "  return output_sparse;\n"
                "}\n"
                "torch::Tensor edge_sddmm(torch::Tensor input_dense1, torch::Tensor input_dense2,\n"
                "torch::Tensor offset_graph,\n"
                "torch::Tensor columns_graph, torch::Tensor value_graph,\n"
                "torch::Tensor bounds, int nrows, int segments) {\n"
                "  const int64_t nvals = columns_graph.numel();\n"
                "  torch::Tensor a_c = input_dense1.contiguous();\n"
                "  torch::Tensor b_c = input_dense2.contiguous();\n"
                "  const int64_t dcols = a_c.numel() / nrows;\n"
                "  const float *iden_ptr1 = a_c.data_ptr<float>();\n"
                "  const float *iden_ptr2 = b_c.data_ptr<float>();\n"
                "  auto options = torch::TensorOptions().dtype(torch::kFloat).requires_grad(true);\n"
                "  auto output_sparse = torch::zeros({nvals}, options);\n"
                "  float *oden_array = output_sparse.data_ptr<float>();\n"
                "  const int *offset_ptr = offset_graph.data_ptr<int>();\n"
                "  const int *col_ptr = columns_graph.data_ptr<int>();\n"
                "  const int *bounds_ptr = bounds.data_ptr<int>();\n"
                "  for (int t = 0; t < segments; t++) {\n"
                "    const int *off = offset_ptr + (int64_t)t * (nrows + 1);\n"
                "    const int *cols = col_ptr + bounds_ptr[2 * t];\n"
                "    float *out = oden_array + bounds_ptr[2 * t];\n"
                "#pragma omp parallel for schedule(dynamic, 64)\n"
                "    for (int i = 0; i < nrows; i++) {\n"
                "      const float *ai = iden_ptr1 + i * dcols;\n"
                "      for (int e = off[i]; e < off[i + 1]; e++) {\n"
                "        const float *bj = iden_ptr2 + (int64_t)cols[e] * dcols;\n"
                "        float acc = 0.f;\n"
                "#pragma omp simd reduction(+:acc)\n"
                "        for (int64_t k = 0; k < dcols; k++) acc += ai[k] * bj[k];\n"
                "        out[e] = acc;\n"
                "      }\n"
                "    }\n"
                "  }\n"
                "  return output_sparse;\n"
                "}\n";
            kernelCallCode.addCode(code);
        }
    }

    void dataPrep(std::vector<CIRNode*>& program) override
    {
        // Same option names the device-neutral emission expects, minus any device clause.
        std::string torchTypesStr = "  torch::Device device(torch::kCPU);\n\
  auto options_cu_int = torch::TensorOptions()\n\
                            .dtype(torch::kInt)\n\
                            .requires_grad(false);\n\
  auto options_cu_float_grad = torch::TensorOptions()\n\
                                   .dtype(torch::kFloat)\n\
                                   .requires_grad(true);\n\
  auto options_cu_float_ngrad = torch::TensorOptions()\n\
                                    .dtype(torch::kFloat)\n\
                                    .requires_grad(false);\n\
  auto options_cu_bool = torch::TensorOptions()\n\
                             .dtype(torch::kBool)\n\
                             .requires_grad(false);\n\
  auto options_cu_long =\n\
      torch::TensorOptions().dtype(torch::kLong);\n";
        preCode.addCode(torchTypesStr);

        // Features, labels and masks are wrapped in place; the host matrices outlive
        // the training loop (they are declared in the same main() scope).
        std::string labelMaskStr = "  torch::Tensor t_iden =\n\
      torch::from_blob(input_emb.vals_ptr(), {nrows, emb_size}, options_cu_float_grad);\n\
  torch::Tensor t_labs = torch::from_blob(labels.vals_ptr(), {nrows}, options_cu_long);\n\
\n\
  torch::Tensor t_train_mask =\n\
      torch::from_blob(train_mask.vals_ptr(), {nrows}, options_cu_bool);\n\
  torch::Tensor t_valid_mask =\n\
      torch::from_blob(valid_mask.vals_ptr(), {nrows}, options_cu_bool);\n\
  torch::Tensor t_test_mask =\n\
      torch::from_blob(test_mask.vals_ptr(), {nrows}, options_cu_bool);";
        preCode.addCode(labelMaskStr);
        transferGraphs(program);
    }
};

// Select the backend named by the schedule's target() directive (or --target).
inline std::unique_ptr<CodeGenerator> makeCodeGenerator(const std::string& target, std::string& outputPath, std::string dataRoot)
{
    if (target == "cpu")
    {
        auto ctx = new GALAContext(CPU_DEVICE, SINGLE_NODE_SINGLE);
        return std::make_unique<CPUGenerator>(ctx, outputPath, dataRoot);
    }
    if (target != "cuda" && target != "gpu")
    {
        std::cerr << "Unknown target '" << target << "'; expected cuda|cpu" << std::endl;
        exit(1);
    }
    auto ctx = new GALAContext(GPU_DEVICE, SINGLE_NODE_SINGLE);
    return std::make_unique<CUDAGenerator>(ctx, outputPath, dataRoot);
}

#endif // GNN_ACCELERATION_LANGUAGE_CPU_H
