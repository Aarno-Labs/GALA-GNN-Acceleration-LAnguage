//
//
#ifndef GNN_ACCELERATION_LANGUAGE_COMMON_H
#define GNN_ACCELERATION_LANGUAGE_COMMON_H

#include <string>
#include <string>
#include <vector>
#include <map>
#include "../ir/compute.h"
#include "../frontend/context.h"
#include <filesystem>
#include <fstream>
#include <iostream>

// Specify the target device (Use to create the type of CodeGen instance)
enum Device
{
    CPU_DEVICE,
    GPU_DEVICE
};

// The distrubuted environment the GNN will operate on
enum Environment
{
    SINGLE_NODE_SINGLE, // Single node with a single CPU/GPU
    SINGLE_NODE_MULTI, // Single node with multiple CPUs/GPUs (Like the A100x4 Server)
    MULTI_NODE_SINGLE, // Distributed multi-node setting, each with a single CPU/GPU
    MULTI_NODE_MULTI // Distributed multi-node setting, each with multiple CPUs/GPUs
};

// Context of the execution used by GALA
class GALAContext
{
private:
    Device device;
    Environment env;

public:
    GALAContext(Device dev, Environment env)
    {
        this->device = dev;
        this->env = env;
    }

    // Get the target device
    Device getDevice()
    {
        return this->device;
    }

    // Get the environment in terms of distribution
    Environment getEnv()
    {
        return this->env;
    }
};

// TODO -- For now both CmakeCode and KernelCode have the same operations
//  Separate this out. CMake is common for all, but the kernel code would also have
//  a PyTorch linking component
class Code
{
private:
    std::vector<std::string> codeLines;

public:
    Code()
    {
    };

    int getNum()
    {
        return this->codeLines.size();
    }

    void addCode(std::string& newCode)
    {
        this->codeLines.push_back(newCode);
    }

    std::string* atLine(int ix)
    {
        return &(this->codeLines.at(ix));
    }
};

class CMakeCode : public Code
{
};

class TargetCode : public Code
{
};

class KernelCode : public TargetCode
{
private:
    std::string name;
    std::vector<std::string> kernelCall;

public:
    KernelCode(std::string& name): TargetCode()
    {
        this->name = name;
    };

    std::string* getName()
    {
        return &this->name;
    }

    // Kernel call
    int getNumCall()
    {
        return this->kernelCall.size();
    }

    void addCallCode(std::string& newCode)
    {
        this->kernelCall.push_back(newCode);
    }

    std::string* atCallLine(int ix)
    {
        return &(this->kernelCall.at(ix));
    }

    void clearCall()
    {
        this->kernelCall.clear();
    }
};

// TODO break this down into multiple parts??
// Model code
class Model
{
private:
    // ID for the model
    std::string modelName;

    // These are in the main funciton
    // Model component definition (The init of a Torch model)
    Code modelPreCall;
    Code modelCall;
    Code modelPostCall;
    // Model Training Invariant
    Code modelInv;
    // Code for the model use
    Code modelTraining;
    Code modelValidation;
    Code modelTesting;

    // Code for the model defition
    // This is for the
    Code modelDef;
    // The code in the initialization section of the model + training invariant code
    Code modelInit;
    Code modelInitCall;
    // Model forward
    Code modelForward;
    Code modelForwardCallPre;
    Code modelForwardCallInternal;
    Code modelForwardCallPost;
    uint forwardTensorArguments;
    std::vector<std::string> forwardTensorArgNames;


public:
    Model() : forwardTensorArguments(0)
    {
        std::string defaultName = "gnn";
        this->modelName = defaultName;
        forwardTensorArgNames.push_back("t_iden");
    }

    Model(std::string& name): forwardTensorArguments(0)
    {
        this->modelName = name;
        forwardTensorArgNames.push_back("t_iden");
    }

    void incForwardTensorArgs()
    {
        forwardTensorArguments++;
    }

    uint numForwardTensorArgs()
    {
        return forwardTensorArguments;
    }

    void addForwardTensorArgName(const std::string& name)
    {
        forwardTensorArgNames.push_back(name);
    }

    const std::vector<std::string>& getForwardTensorArgNames() const
    {
        return forwardTensorArgNames;
    }

    // TODO this is at the code generation phase so you don't need to clear / remove stuff
    //  Those should already have been decided in previous passes
    std::string* getName()
    {
        return &this->modelName;
    }

    // Def
    Code* getDef()
    {
        return &this->modelDef;
    }
    Code* getForwardCallPre()
    {
        return &this->modelForwardCallPre;
    }
    Code* getForwardCallInternal()
    {
        return &this->modelForwardCallInternal;
    }
    Code* getForwardCallPost()
    {
        return &this->modelForwardCallPost;
    }

    // Init
    Code* getInit()
    {
        return &this->modelInit;
    }
    Code* getInitCall()
    {
        return &this->modelInitCall;
    }

    // Use
    Code* getPreCall()
    {
        return &this->modelPreCall;
    }
    Code* getCall()
    {
        return &this->modelCall;
    }
    Code* getPostCall()
    {
        return &this->modelPostCall;
    }
    // Invariant
    Code* getInv()
    {
        return &this->modelInv;
    }

    // Forward
    Code* getForward()
    {
        return &this->modelForward;
    }

    // Training
    Code* getTraining()
    {
        return &this->modelTraining;
    }

    // Validation
    Code* getValidation()
    {
        return &this->modelValidation;
    }

    // Testing
    Code* getTesting()
    {
        return &this->modelTesting;
    }
};

// TODO You're generating code for Python. So you NEED to identify where to make any indents
class CodeGenerator
{
private:
    GALAContext* context;

protected:
    Code cmakeCode;
    Code importCode; // Imports
    Code kernelCode; // Kernels
    Code kernelCallCode; // Kernel calls
    Code autoGradCode; // Autograd code
    Code preCode; // Just one for now. Preprocessing should be done first and THEN the trasfers.
    Model model; // TODO: Assume a single model for now
    Code postCode; // Cleanup code?

    std::vector<std::string> generatedFunctions;

    std::ofstream outStreamModel;
    std::ofstream outStreamCMake;
    std::string dataRoot;
    std::string outputPath;

public:
    CodeGenerator(GALAContext* context, std::string& outputPath, std::string dataRoot)
    {
        this->context = context;
        this->dataRoot = dataRoot;
        this->outputPath = outputPath;
        if (!this->dataRoot.empty() && this->dataRoot.back() == '/')
            this->dataRoot.pop_back();
        // Streams are opened in writeCode(): the model file name is backend-specific
        // (virtual), and virtual dispatch is not available in a base constructor.
    }


    std::string processDims(int val)
    {
        if (val < 0)
        {
            if (val == -1)
            {
                return "global_nrows";
            } else if (val == -2)
            {
                return "global_emb_size";
            } else if (val == -3)
            {
                return "global_classes";
            } else
            {
                // TODO
                return "ERROR!!!";
            }
        } else
        {
            return std::to_string(val);
        }
    }

    // Move to data node 
    bool hasDOpt(DataNode* dNode, DataOptimization op)
    {
        auto opts = dNode->getDataInfo()->getOpts();
        for (int ix = 0; ix < opts->size(); ix++)
        {
            auto opt = opts->at(ix);
            if (opt.first == op)
            {
                return true;
            }
        }
        return false;
    }

    // TODO Move to compute node
    bool hasCOpt(ComputeNode* cNode, CompOptimization op)
    {
        for (int ix = 0; ix < cNode->getNumOpts(); ix++)
        {
            auto opt = cNode->getOpt(ix);
            if (opt->first == op)
            {
                return true;
            }
        }
        return false;
    }

    std::string getKernelName(ComputeNode* cNode)
    {
        std::string kernelName = "";
        if (cNode->getOp() == AGGREGATE_MUL_SUM_OP)
        {
            kernelName += "aggregate_node_mul_sum";
        } else if (cNode->getOp() == AGGREGATE_MUL_SUM_DIRECT)
        {
            kernelName += "aggregate_node_mul_sum_direct";
        } else if (cNode->getOp() == NON_LNR_OP_SOFTMAX)
        {
            kernelName += "non_lnr_op_softmax";
        } else if (cNode->getOp() == AGGREGATE_EDGE_MUL_OP)
        {
            kernelName += "aggregate_edge_mul";
        } else if (cNode->getOp() == AGGREGATE_EDGE_SUM_OP)
        {
            kernelName += "aggregate_edge_sum";
        } else if (cNode->getOp() == LOAD_OP)
        {
            kernelName += "load_op";
        } else if (cNode->getOp() == ONES_OP)
        {
            kernelName += "ones_op";
        } else
        {
            kernelName += kernelName + " is unsupported";
        }
        // TODO add other kernel optimizations
        for (std::pair<CompOptimization, float> optPair: *cNode->getOpts())
        {
            if (optPair.first == COARSE_COPT)
            {
                kernelName += "_coarse" + std::to_string((int)(optPair.second));
            }
        }
        return kernelName;
    }

    std::string generateEvaluatorTestCall()
    {
        std::string result = "    evaluator.test<";
        // Generate template arguments (torch::Tensor for each tensor arg)
        auto& argNames = model.getForwardTensorArgNames();
        for (size_t i = 0; i < argNames.size(); ++i)
        {
            if (i > 0) result += ", ";
            result += "torch::Tensor";
        }
        result += ">(net.get(), &GALAGNN::forward, ";
        // Generate actual arguments (the tensor arg names)
        for (size_t i = 0; i < argNames.size(); ++i)
        {
            if (i > 0) result += ", ";
            result += argNames[i];
        }
        // Add the remaining fixed arguments
        result += ", epoch, mod_v, t_labs, t_train_mask, t_test_mask, t_valid_mask, train_acc, test_acc, val_acc);";
        return result;
    }

    std::string generateOutputString(ComputeNode* cNode, bool outOfLoop)
    {
        for (int ix = 0; ix < cNode->getNumInputs(); ix++)
        {
            if (cNode->getOutput(0)->getName() == cNode->getInput(ix)->getName())
            {
                if (cNode->getOp() == FFN_OP)
                {

                }
                return cNode->getInput(ix)->getName();
            }
        }
         if (outOfLoop)
         {
             return "torch::Tensor " + cNode->getOutput(0)->getName();
         } else
         {
             return cNode->getOutput(0)->getName();
         }

    }

    std::string generateTransformation(DataNode* srcNode, std::vector<TransformEdge*>& transforms)
    {
        std::string resString = "";
        for (int ix = 0; ix < transforms.size(); ix++)
        {
            auto transform = transforms[ix];
            // std::cout << "aa: " << transform->getNode1()->getName() << " " << transform->getNode2()->getName() << " " << transform->getNumTransformations() << std::endl;
            if (transform->getNode1()->getName() == srcNode->getName())
            {
                auto dNode = transform->getNode2();
                for (int tix = 0; tix < transform->getNumTransformations(); tix++)
                {
                    auto tr = transform->getTransformation(tix);
                    if (tr->getTransformation() == COL_TILE_DOPT)
                    {
                        resString +=  "  std::vector<SM *> tiled_" + dNode->getName() +";\n\
      tiled_" + dNode->getName() + ".push_back(&" + srcNode->getName() + ");\n\
      torch::Tensor total_offsets_" + dNode->getName() + ";\n\
      torch::Tensor total_cols_" + dNode->getName() + ";\n\
      torch::Tensor total_vals_" + dNode->getName() + ";\n\
      torch::Tensor total_bounds_" + dNode->getName() + ";\n\
      std::vector<iT> tile_offsets_" + dNode->getName() + " =\n\
        static_ord_col_breakpoints<SM>(&" + srcNode->getName() + ", " + tr->getParam(0) +");\n\
      iT segments_" + dNode->getName() + " = tile_offsets_" + dNode->getName() + ".size() - 1;\n\
      total_offsets_" + dNode->getName() + " = torch::zeros({(" + srcNode->getName() + ".nrows() + 1) * (segments_"
                        + dNode->getName() + ")}, options_int_tile);\n\
      total_cols_" + dNode->getName() + " = torch::zeros({" + srcNode->getName() + ".nvals()}, options_int_tile);\n\
      total_vals_" + dNode->getName() + " = torch::zeros({" + srcNode->getName() + ".nvals()}, options_float_tile);\n\
      total_bounds_" + dNode->getName() + " = torch::zeros({2 * (segments_" + dNode->getName() + ")}, options_int_tile);\n\
      ord_col_tiling_torch(tile_offsets_" + dNode->getName() + ", total_offsets_" + dNode->getName() +
                            ", total_cols_" + dNode->getName() + ", total_vals_" + dNode->getName() + ",\n\
        total_bounds_" + dNode->getName() + ", &" + srcNode->getName() + ");\n\
      iT *offset_ptr_" + dNode->getName() + " = total_offsets_" + dNode->getName() + ".data_ptr<iT>();\n\
      iT *col_ptr_" + dNode->getName() + " = total_cols_" + dNode->getName() + ".data_ptr<iT>();\n\
      vT *val_ptr_" + dNode->getName() + " = total_vals_" + dNode->getName() + ".data_ptr<vT>();\n";

                        resString += "  global_segments.push_back(segments_" + dNode->getName() + ");\n";
                        resString += "  global_bounds.push_back(total_bounds_" + dNode->getName() + ");\n";
                        // std::cout << transform->getNode1()->getName() << " " << transform->getNode2()->getName() << std::endl;
                        // std::cout << transform->getNode1()->getDataInfo()->getDirected() << " " << transform->getNode2()->getDataInfo()->getDirected() << std::endl;
                        if (!dNode->getDataInfo()->getDirected())
                        {
                            resString += "  global_segments.push_back(segments_" + dNode->getName() + ");\n";
                            resString += "  global_bounds.push_back(total_bounds_" + dNode->getName() + ");\n";
                            // Undirected: perm maps each (j,i) position to (i,j) position in the same tiled CSR.
                            resString += "  {\n";
                            resString += "    std::vector<int> perm_data_" + dNode->getName() + " = compute_transpose_perm<iT>(\n";
                            resString += "        offset_ptr_" + dNode->getName() + ", col_ptr_" + dNode->getName() + ",\n";
                            resString += "        total_bounds_" + dNode->getName() + ".data_ptr<iT>(), segments_" + dNode->getName() + ",\n";
                            resString += "        offset_ptr_" + dNode->getName() + ", col_ptr_" + dNode->getName() + ",\n";
                            resString += "        total_bounds_" + dNode->getName() + ".data_ptr<iT>(), segments_" + dNode->getName() + ",\n";
                            resString += "        (int)" + srcNode->getName() + ".nrows(), (int)" + srcNode->getName() + ".nvals());\n";
                            resString += transposePermToDevice(dNode->getName(), srcNode->getName() + ".nvals()");
                            resString += "    global_transpose_perm.push_back(t_perm_" + dNode->getName() + ");\n";
                            resString += "  }\n";
                        } else
                        {
                            std::string tilingParam;
                            if (tr->getNumParam() > 1)
                            {
                                tilingParam = tr->getParam(1);
                            } else
                            {
                                tilingParam = tr->getParam(0);
                            }
                            // The directed tiling below references <srcNode>_b (the reverse
                            // graph). Subgraphs receive theirs from getMaskSubgraphs, but the
                            // top-level loaded graph (adj0) has no transpose yet, so build it.
                            if (srcNode->getName() == "adj0")
                            {
                                resString += "  SM " + srcNode->getName() + "_b;\n\
      buildTranspose(&" + srcNode->getName() + ", &" + srcNode->getName() + "_b);\n";
                            }
                            resString +=  "  std::vector<SM *> tiled_" + dNode->getName() +"_b;\n\
      tiled_" + dNode->getName() + "_b.push_back(&" + srcNode->getName() + "_b);\n\
      torch::Tensor total_offsets_" + dNode->getName() + "_b;\n\
      torch::Tensor total_cols_" + dNode->getName() + "_b;\n\
      torch::Tensor total_vals_" + dNode->getName() + "_b;\n\
      torch::Tensor total_bounds_" + dNode->getName() + "_b;\n\
      std::vector<iT> tile_offsets_" + dNode->getName() + "_b =\n\
        static_ord_col_breakpoints<SM>(&" + srcNode->getName() + "_b, " + tilingParam +");\n\
      iT segments_" + dNode->getName() + "_b = tile_offsets_" + dNode->getName() + "_b.size() - 1;\n\
      total_offsets_" + dNode->getName() + "_b = torch::zeros({(" + srcNode->getName() + "_b.nrows() + 1) * (segments_"
                        + dNode->getName() + "_b)}, options_int_tile);\n\
      total_cols_" + dNode->getName() + "_b = torch::zeros({" + srcNode->getName() + "_b.nvals()}, options_int_tile);\n\
      total_vals_" + dNode->getName() + "_b = torch::zeros({" + srcNode->getName() + "_b.nvals()}, options_float_tile);\n\
      total_bounds_" + dNode->getName() + "_b = torch::zeros({2 * (segments_" + dNode->getName() + "_b)}, options_int_tile);\n\
      ord_col_tiling_torch(tile_offsets_" + dNode->getName() + "_b, total_offsets_" + dNode->getName() +
                            "_b, total_cols_" + dNode->getName() + "_b, total_vals_" + dNode->getName() + "_b,\n\
        total_bounds_" + dNode->getName() + "_b, &" + srcNode->getName() + "_b);\n\
      iT *offset_ptr_" + dNode->getName() + "_b = total_offsets_" + dNode->getName() + "_b.data_ptr<iT>();\n\
      iT *col_ptr_" + dNode->getName() + "_b = total_cols_" + dNode->getName() + "_b.data_ptr<iT>();\n\
      vT *val_ptr_" + dNode->getName() + "_b = total_vals_" + dNode->getName() + "_b.data_ptr<vT>();\n";

                            resString += "  global_segments.push_back(segments_" + dNode->getName() + "_b);\n";
                            resString += "  global_bounds.push_back(total_bounds_" + dNode->getName() + "_b);\n";
                            // Directed: perm maps tiled-backward positions to tiled-forward positions.
                            resString += "  {\n";
                            resString += "    std::vector<int> perm_data_" + dNode->getName() + " = compute_transpose_perm<iT>(\n";
                            resString += "        offset_ptr_" + dNode->getName() + ", col_ptr_" + dNode->getName() + ",\n";
                            resString += "        total_bounds_" + dNode->getName() + ".data_ptr<iT>(), segments_" + dNode->getName() + ",\n";
                            resString += "        offset_ptr_" + dNode->getName() + "_b, col_ptr_" + dNode->getName() + "_b,\n";
                            resString += "        total_bounds_" + dNode->getName() + "_b.data_ptr<iT>(), segments_" + dNode->getName() + "_b,\n";
                            resString += "        (int)" + srcNode->getName() + "_b.nrows(), (int)" + srcNode->getName() + "_b.nvals());\n";
                            resString += transposePermToDevice(dNode->getName(), srcNode->getName() + "_b.nvals()");
                            resString += "    global_transpose_perm.push_back(t_perm_" + dNode->getName() + ");\n";
                            resString += "  }\n";
                        }
                    } else if (tr->getTransformation() == SUBGRAPH_DOPT)
                    {
                        if (tr->getNumParam() == 2)
                        {
                            resString +=  " std::vector<SM *> forward_adj;\n\
      std::vector<SM *> backward_adj;\n\
      getMaskSubgraphs(&adj0, &train_mask, " + tr->getParam(1) + ", forward_adj, backward_adj);\n";
                            for (int i = 0; i < std::stoi(tr->getParam(1)); i++)
                            {
                                int iy = std::stoi(tr->getParam(1)) - (i + 1);
                                int iz = i + 1;
                                resString += "  SM adj" + std::to_string(iz) + " = *forward_adj[" + std::to_string(iy) +"];\n\
      SM adj" + std::to_string(iz) + "_b = *backward_adj[" + std::to_string(iy) +"];\n\
      nT nvals" + std::to_string(iz) + " = adj" + std::to_string(iz) + ".nvals();\n";
                            }
                        }
                        resString += generateTransformation(dNode, transforms);
                    } else if (tr->getTransformation() == SAMPLE_DOPT)
                    {
                        resString += "inplace_sample_graph_ab(&" + srcNode->getName() + ", " +  tr->getParam(0) + ", 5, 7);\n\
nvals0 = adj0.nvals();\n";
                        resString += generateTransformation(dNode, transforms);
                    }

                }
            }
        }
        return resString;

    }

    void generateOpCode(ComputeNode* cNode, int& fcCount, int& fcEdgeCount, int& fcSelfCount,int& epCount, bool outOfLoop, bool& hasFFNEdgeUpdate, bool& hasEdgeMulAggr,
        std::unordered_set<std::string> &encounteredAutograds,
        std::vector<int> &inputSizes,
        std::vector<TransformEdge*>& transforms)
    {
        if (cNode->getOp() == LOAD_OP)
        {
            // TODO assume a single load for now and make the index for it 0

            for (int oI = 0; oI < cNode->getNumOutputs(); oI++)
            {
                auto currentInfo = cNode->getOutput(oI)->getDataInfo();
                if (currentInfo->getFormat() == CSR_STYPE)
                {
                    currentInfo->setIndex(0);
                }
            }

            filesystem::path data_path = GALAFEContext::data_root / cNode->getParam(0);
            // This doesn't need to change
            std::string fileLoadCode;
            if (GALAFEContext::use_long)
            {
                fileLoadCode = "    SM adj0;\n\
    std::string filename = \"" + this->dataRoot + "/" + cNode->getParam(0) + "/\";\n\
    readSM_npy32<SM>(filename, &adj0);\n\
\n\
    // Adj info\n\
    int64_t nrows = (int64_t)adj0.nrows();\n\
    global_nrows = (iT)nrows;\n\
    int64_t ncols = (int64_t)adj0.ncols();\n\
    int64_t nvals0 = (int64_t)adj0.nvals();\n\
\n\
    // Init input with random numbers\n\
    DM input_emb;\n\
    readDM_npy<DM>(filename + \"Feat.npy\", &input_emb,\n\
                   DM::DENSE_MTX_TYPE::RM);\n\
    int64_t emb_size = (int64_t)input_emb.ncols();\n\
\n\
    DL labels;\n\
    readDM_npy<DL>(filename + \"Lab.npy\", &labels,\n\
                   DL::DENSE_MTX_TYPE::RM);\n\
\n\
    DBL train_mask_load;\n\
    readDM_npy<DBL>(filename + \"TnMsk.npy\", &train_mask_load,\n\
                    DBL::DENSE_MTX_TYPE::RM);\n\
    DBL valid_mask_load;\n\
    readDM_npy<DBL>(filename + \"VlMsk.npy\", &valid_mask_load,\n\
                    DBL::DENSE_MTX_TYPE::RM);\n\
    DBL test_mask_load;\n\
    readDM_npy<DBL>(filename + \"TsMsk.npy\", &test_mask_load,\n\
                    DBL::DENSE_MTX_TYPE::RM);\n\
\n\
    DB train_mask;\n\
    repopulate<DBL, DB>(&train_mask_load, &train_mask);\n\
    DB valid_mask;\n\
    repopulate<DBL, DB>(&valid_mask_load, &valid_mask);\n\
    DB test_mask;\n\
    repopulate<DBL, DB>(&test_mask_load, &test_mask);\n\
    int classes =\n\
    *std::max_element(labels.vals_ptr(), labels.vals_ptr() + labels.nvals()) + 1;\n\
    global_classes = classes;\n\
    global_emb_size = emb_size;";
            } else
            {
                fileLoadCode = "    SM adj0;\n\
    std::string filename = \"" + this->dataRoot + "/" + cNode->getParam(0) + "/\";\n\
    readSM_npy32<SM>(filename, &adj0);\n\
\n\
    // Adj info\n\
    iT nrows = adj0.nrows();\n\
    global_nrows = nrows;\n\
    iT ncols = adj0.ncols();\n\
    nT nvals0 = adj0.nvals();\n\
\n\
    // Init input with random numbers\n\
    DM input_emb;\n\
    readDM_npy<DM>(filename + \"Feat.npy\", &input_emb,\n\
                   DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);\n\
    iT emb_size = input_emb.ncols();\n\
\n\
    DL labels;\n\
    readDM_npy<DL>(filename + \"Lab.npy\", &labels,\n\
                   DenseMatrix<ind1_t, ind2_t, lab_t>::DENSE_MTX_TYPE::RM);\n\
\n\
    DBL train_mask_load;\n\
    readDM_npy<DBL>(filename + \"TnMsk.npy\", &train_mask_load,\n\
                    DBL::DENSE_MTX_TYPE::RM);\n\
    DBL valid_mask_load;\n\
    readDM_npy<DBL>(filename + \"VlMsk.npy\", &valid_mask_load,\n\
                    DBL::DENSE_MTX_TYPE::RM);\n\
    DBL test_mask_load;\n\
    readDM_npy<DBL>(filename + \"TsMsk.npy\", &test_mask_load,\n\
                    DBL::DENSE_MTX_TYPE::RM);\n\
\n\
    DB train_mask;\n\
    repopulate<DBL, DB>(&train_mask_load, &train_mask);\n\
    DB valid_mask;\n\
    repopulate<DBL, DB>(&valid_mask_load, &valid_mask);\n\
    DB test_mask;\n\
    repopulate<DBL, DB>(&test_mask_load, &test_mask);\n\
    int classes =\n\
    *std::max_element(labels.vals_ptr(), labels.vals_ptr() + labels.nvals()) + 1;\n\
    global_classes = classes;\n\
    global_emb_size = emb_size;";
            }
            preCode.addCode(fileLoadCode);

            // Graph output
            auto outputGraph = cNode->getOutput(1);
            std::string transformationCode = generateTransformation(outputGraph, transforms);
            preCode.addCode(transformationCode);
        } else if (cNode->getOp() == AGGREGATE_EDGE_SUM_OP)
        {
            hasFFNEdgeUpdate = true;
            bool isColTile = hasDOpt(cNode->getInput(2), COL_TILE_DOPT);

            if (encounteredAutograds.find(getKernelName(cNode)) == encounteredAutograds.end())
            {
                encounteredAutograds.insert(getKernelName(cNode));
                std::string autoGradFunction = "class " + getKernelName(cNode) + "_AutoGrad : public torch::autograd::Function<" + getKernelName(cNode) + "_AutoGrad> {\n\
public:\n\
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,\n\
                               torch::Tensor input_dense1,\n\
                               torch::Tensor input_dense2,\n\
                               int li) {\n";
                autoGradFunction += "        ctx->saved_data[\"li\"] = li;\n\
        torch::Tensor offset_graph = global_offset_graph[2 * li];\n\
        torch::Tensor columns_graph = global_columns_graph[2 * li];\n\
        torch::Tensor value_graph = global_value_graph[2 * li];\n";

                if (isColTile){
                    autoGradFunction += "        torch::Tensor bounds = global_bounds[2 * li];\n\
        int segments = global_segments[2 * li];\n\
        return edge_sddvv(input_dense1, input_dense2, offset_graph, columns_graph,\n\
                            value_graph, bounds, global_nrows, segments);";
                } else
                {
                    std::cout << "This unsup: AGGREGATE_EDGE_SUM_OP" << std::endl;
                    autoGradFunction += "unsupported\n";
                }
                autoGradFunction += "}\n";

                autoGradFunction += " static torch::autograd::tensor_list\n\
                    backward(torch::autograd::AutogradContext *ctx,\n\
                             torch::autograd::tensor_list grad_outputs) {\n\
                    torch::Tensor d_value_graph = grad_outputs[0];\n";
                autoGradFunction += " int li = ctx->saved_data[\"li\"].toInt();\n\
        torch::Tensor offset_graph_fwd = global_offset_graph[2 * li];\n\
        torch::Tensor columns_graph_fwd = global_columns_graph[2 * li];\n\
        torch::Tensor bounds_fwd = global_bounds[2 * li];\n\
        int segments_fwd = global_segments[2 * li];\n\
        torch::Tensor offset_graph_bwd = global_offset_graph[2 * li + 1];\n\
        torch::Tensor columns_graph_bwd = global_columns_graph[2 * li + 1];\n\
        torch::Tensor bounds_bwd = global_bounds[2 * li + 1];\n\
        int segments_bwd = global_segments[2 * li + 1];\n\
        torch::Tensor perm = global_transpose_perm[li];\n\
        torch::Tensor d_value_graph_b = d_value_graph.index_select(0, perm);\n\
        torch::Tensor back_res1 = node_spmv_backward_of_sddmm_eaggr(\n\
                    offset_graph_fwd, columns_graph_fwd,\n\
                    d_value_graph, bounds_fwd, global_nrows, segments_fwd);\n\
        torch::Tensor back_res2 = node_spmv_backward_of_sddmm_eaggr(\n\
                    offset_graph_bwd, columns_graph_bwd,\n\
                    d_value_graph_b, bounds_bwd, global_nrows, segments_bwd);\n\
        return {back_res1,\n\
                back_res2,\n\
                torch::Tensor()};\n\
    }\n\
};\n";
                kernelCallCode.addCode(autoGradFunction);
            }
            auto inGraphIndx = cNode->getInput(2)->getDataInfo()->getIndex();
            std::string tempForwardAggrCall = generateOutputString(cNode, outOfLoop) + " = " + getKernelName(cNode)
            + "_AutoGrad::apply(" + cNode->getInput(0)->getName() + ", " + cNode->getInput(1)->getName() + ", " + std::to_string(inGraphIndx) + ");";
            model.getForward()->addCode(tempForwardAggrCall);
        } else if (cNode->getOp() == AGGREGATE_EDGE_MUL_OP)
        {
            hasEdgeMulAggr = true;
            std::string tempForwardAggrCall = "";
            bool isColTile = hasDOpt(cNode->getInput(2), COL_TILE_DOPT);
            auto inGraphIndx = cNode->getInput(2)->getDataInfo()->getIndex();
            if (isColTile){
                std::string normCall = "auto options_ones_val = torch::TensorOptions()\n\
                           .dtype(torch::kFloat)\n\
                           .requires_grad(false)\n\
                           " + deviceOpt() + ";\n\
                torch::Tensor ones_val = torch::ones({global_nrows, 1}, options_ones_val);\n\
                torch::Tensor offset_graph_ones_val = global_offset_graph[2 * 0];\n\
                torch::Tensor columns_graph_ones_val = global_columns_graph[2 * 0];\n\
                torch::Tensor value_graph_ones_val = global_value_graph[2 * 0];\n\
                torch::Tensor bounds_ones_val = global_bounds[2 * 0];\n\
                int segments_ones_val = global_segments[2 * 0];\n\
                torch::Tensor degrees_val = aggregate_node_mul_sum_coarse2_call(ones_val, offset_graph_ones_val, columns_graph_ones_val,\n\
                                    value_graph_ones_val, bounds_ones_val, segments_ones_val);\n\
                torch::Tensor norm_val = torch::pow(degrees_val, -0.500000).detach();\n";
                model.getInv()->addCode(normCall);

                tempForwardAggrCall = "        torch::Tensor offset_graph_vals = global_offset_graph[2 * " + std::to_string(inGraphIndx) + "];\n\
            torch::Tensor columns_graph_vals = global_columns_graph[2 * " + std::to_string(inGraphIndx) + "];\n\
            torch::Tensor value_graph_vals = global_value_graph[2 * " + std::to_string(inGraphIndx) + "];\n";

                tempForwardAggrCall += "        torch::Tensor bounds_vals = global_bounds[2 * " + std::to_string(inGraphIndx) + "];\n\
        int segments_vals = global_segments[2 * " + std::to_string(inGraphIndx) + "];\n";
                tempForwardAggrCall += "torch::Tensor " + generateOutputString(cNode, outOfLoop) + " = " + getKernelName(cNode)
            + "(" + cNode->getInput(0)->getName() + "_val, " + cNode->getInput(1)->getName() + "_val, offset_graph_vals, columns_graph_vals, value_graph_vals, bounds_vals, segments_vals).detach();";
            } else
            {
                std::string normCall = "auto options_ones_val = torch::TensorOptions()\n\
                           .dtype(torch::kFloat)\n\
                           .requires_grad(false)\n\
                           " + deviceOpt() + ";\n\
                torch::Tensor ones_val = torch::ones({global_nrows, 1}, options_ones_val);\n\
                torch::Tensor offset_graph_ones_val = global_offset_graph[2 * 0];\n\
                torch::Tensor columns_graph_ones_val = global_columns_graph[2 * 0];\n\
                torch::Tensor value_graph_ones_val = global_value_graph[2 * 0];\n\
                torch::Tensor degrees_val = aggregate_node_mul_sum_call(ones_val, offset_graph_ones_val, columns_graph_ones_val,\n\
                                    value_graph_ones_val);\n\
                torch::Tensor norm_val = torch::pow(degrees_val, -0.500000).detach();\n";
                model.getInv()->addCode(normCall);

                tempForwardAggrCall = "        torch::Tensor offset_graph_vals = global_offset_graph[2 * " + std::to_string(inGraphIndx) + "];\n\
            torch::Tensor columns_graph_vals = global_columns_graph[2 * " + std::to_string(inGraphIndx) + "];\n\
            torch::Tensor value_graph_vals = global_value_graph[2 * " + std::to_string(inGraphIndx) + "];\n";

                tempForwardAggrCall += "torch::Tensor " + generateOutputString(cNode, outOfLoop) + " = " + getKernelName(cNode)
            + "_dir(" + cNode->getInput(0)->getName() + "_val, " + cNode->getInput(1)->getName() + "_val, offset_graph_vals, columns_graph_vals, value_graph_vals).detach();";
            }

            // tempForwardAggrCall += "torch::Tensor " + generateOutputString(cNode, outOfLoop) + " = " + getKernelName(cNode)
            // + "(" + cNode->getInput(0)->getName() + "_val, " + cNode->getInput(1)->getName() + "_val, offset_graph_vals, columns_graph_vals, value_graph_vals, bounds_vals, segments_vals).detach();";
            model.getInv()->addCode(tempForwardAggrCall);

            std::string resetVal = "global_value_graph[2 * " + std::to_string(inGraphIndx) + "] = " + generateOutputString(cNode, outOfLoop) + ";";
            model.getInv()->addCode(resetVal);
        } else if (cNode->getOp() == NON_LNR_OP_SOFTMAX)
        {
            bool isColTile = hasDOpt(cNode->getInput(0), COL_TILE_DOPT);

            if (encounteredAutograds.find(getKernelName(cNode)) == encounteredAutograds.end())
            {
                encounteredAutograds.insert(getKernelName(cNode));
                std::string autoGradFunction = "class " + getKernelName(cNode) + "_AutoGrad : public torch::autograd::Function<" + getKernelName(cNode) + "_AutoGrad> {\n\
public:\n\
  static torch::Tensor forward(torch::autograd::AutogradContext *ctx,\n\
                               torch::Tensor value_graph,\n\
                               int li) {\n";
                autoGradFunction += "        ctx->saved_data[\"li\"] = li;\n\
        torch::Tensor offset_graph = global_offset_graph[2 * li];\n\
        torch::Tensor columns_graph = global_columns_graph[2 * li];\n";

                if (isColTile){
                    autoGradFunction += "        torch::Tensor bounds = global_bounds[2 * li];\n\
        int segments = global_segments[2 * li];\n";
                } else
                {
                    std::cout << "This unsup: NON_LNR_OP_SOFTMAX" << std::endl;
                    autoGradFunction += "unsupported(NON_LNR_OP_SOFTMAX)\n";
                }

                autoGradFunction += "    value_graph = value_graph.clone();\n\
    sparse_softmax_stabilize(offset_graph, columns_graph, value_graph, bounds, global_nrows, segments);\n\
    torch::Tensor val_exp = torch::exp(value_graph);\n\
    torch::Tensor row_sum = node_spmv_backward_of_sddmm_nln(\n\
        offset_graph, columns_graph, val_exp, bounds, global_nrows,\n\
        segments);\n\
    auto options = torch::TensorOptions()\n\
                       .dtype(torch::kFloat)\n\
                       .requires_grad(true)\n\
                       " + deviceOpt() + ";\n\
    row_sum = torch::reciprocal(row_sum);\n\
    val_exp = inplace_softmax_sddvv(row_sum, offset_graph, columns_graph, \n\
                                    val_exp, bounds, global_nrows, segments);\n\
    ctx->save_for_backward({val_exp});\n\
    return val_exp;\n\
  }\n\
  static torch::autograd::tensor_list\n\
  backward(torch::autograd::AutogradContext *ctx,\n\
           torch::autograd::tensor_list grad_outputs) {\n\
    torch::Tensor d_value_graph = grad_outputs[0];\n\
    auto saved = ctx->get_saved_variables();\n";
                // Softmax is a per-row op: its backward must group over the SAME graph as
                // the forward (global_offset_graph[2*li]), not the reverse graph. For
                // undirected graphs offset[2*li]==offset[2*li+1] so this was masked; for
                // directed graphs the reverse graph is the transpose and breaks the gradient.
                autoGradFunction += " int li = ctx->saved_data[\"li\"].toInt();\n\
        torch::Tensor offset_graph = global_offset_graph[2 * li];\n\
        torch::Tensor columns_graph = global_columns_graph[2 * li];\n";

                if (isColTile){
                    autoGradFunction += "        torch::Tensor bounds = global_bounds[2 * li];\n\
        int segments = global_segments[2 * li];\n";
                } else
                {
                    autoGradFunction += "unsupported\n";
                }
                autoGradFunction += "    torch::Tensor value_graph = saved[0]; // n x 1\n\
    torch::Tensor sds = value_graph * d_value_graph; // e x 1\n\
    torch::Tensor accum = node_spmv_backward_of_sddmm_nln(\n\
        offset_graph, columns_graph, sds, bounds, global_nrows, segments); // n x 1\n\
    torch::Tensor res = inplace_softmax_sddvv_mult(\n\
        accum, offset_graph, columns_graph, value_graph, bounds, global_nrows,\n\
        segments);\n\
    res = sds - res;\n\
    return {res, torch::Tensor()};\n\
  }\n\
};\n";
                kernelCallCode.addCode(autoGradFunction);
            }
            // auto inGraphIndx = cNode->getInput(0)->getDataInfo()->getIndex();
            // TODO Temp fix
            cNode->getInput(0)->getDataInfo()->setIndex(0);
            auto inGraphIndx = 0;
            std::string tempForwardAggrCall = generateOutputString(cNode, outOfLoop) + " = " + getKernelName(cNode)
            + "_AutoGrad::apply(" + cNode->getInput(0)->getName() +", " + std::to_string(inGraphIndx) + ");";
            model.getForward()->addCode(tempForwardAggrCall);
        } else if (cNode->getOp() == AGGREGATE_MUL_SUM_OP)
        {
            if (hasCOpt(cNode, SAMPLE_COPT)){
                if (encounteredAutograds.find("random_r_b") == encounteredAutograds.end())
                {
                    encounteredAutograds.insert("random_r_b");
                    std::string forwardRandCall = "    global_ra = 5;\n\
    global_rb = 7;\n";
                    model.getForward()->addCode(forwardRandCall);
                }
            }
            if (hasCOpt(cNode, SAMPLE_DYNAMIC_COPT)){
                if (encounteredAutograds.find("random_r_b") == encounteredAutograds.end())
                {
                    encounteredAutograds.insert("random_r_b");
                    std::string forwardRandCall = "    std::random_device rd;\n\
    std::mt19937 gen(rd());\n\
    std::uniform_int_distribution<> distrib(0, 100);\n\
    global_ra = distrib(gen);\n\
    global_rb = distrib(gen);\n";
                    model.getForward()->addCode(forwardRandCall);
                }
            }
            
            if (hasFFNEdgeUpdate)
            {
                bool isColTile = hasDOpt(cNode->getInput(1), COL_TILE_DOPT);
                if (encounteredAutograds.find(getKernelName(cNode)) == encounteredAutograds.end())
                {
                    encounteredAutograds.insert(getKernelName(cNode));
                    std::string autoGradFunction = ""
    "class " + getKernelName(cNode) + "_AutoGrad : public torch::autograd::Function<" + getKernelName(cNode) + "_AutoGrad> {\n\
    public:\n\
        static torch::Tensor forward(torch::autograd::AutogradContext *ctx,\n\
                                     torch::Tensor input_dense, torch::Tensor value_graph, int li) {\n\
            ctx->saved_data[\"li\"] = li;\n\
            torch::Tensor offset_graph = global_offset_graph[2 * li];\n\
            torch::Tensor columns_graph = global_columns_graph[2 * li];\n\
            ctx->save_for_backward(\n\
                {value_graph, input_dense});\n";
                if (isColTile){
                    autoGradFunction += "        torch::Tensor bounds = global_bounds[2 * li];\n\
            int segments = global_segments[2 * li];\n\
             return " + getKernelName(cNode) + "_call(input_dense, offset_graph, columns_graph,\n\
                                value_graph, bounds, segments);\n";
                } else
                {
                    autoGradFunction += "        return " + getKernelName(cNode) + "_call(input_dense, offset_graph, columns_graph,\n\
                                      value_graph);\n";
                }
                autoGradFunction += "    }\n\
    \n\
        static torch::autograd::tensor_list\n\
        backward(torch::autograd::AutogradContext *ctx,\n\
                 torch::autograd::tensor_list grad_outputs) {\n\
            torch::Tensor dZ = grad_outputs[0];\n\
            auto saved = ctx->get_saved_variables();\n\
            torch::Tensor value_graph = saved[0];\n\
            torch::Tensor X = saved[1];\n\
            int li = ctx->saved_data[\"li\"].toInt();\n\
            torch::Tensor offset_graph = global_offset_graph[2 * li + 1];\n\
            torch::Tensor columns_graph = global_columns_graph[2 * li + 1];";
                if (isColTile){
                    autoGradFunction += "        torch::Tensor bounds_b = global_bounds[2 * li + 1];\n\
            int segments_b = global_segments[2 * li + 1];\n\
            torch::Tensor perm = global_transpose_perm[li];\n\
            torch::Tensor value_graph_b = value_graph.index_select(0, perm);\n\
            torch::Tensor offset_graph_f = global_offset_graph[2 * li];\n\
            torch::Tensor columns_graph_f = global_columns_graph[2 * li];\n\
            torch::Tensor bounds_f = global_bounds[2 * li];\n\
            int segments_f = global_segments[2 * li];\n\
            return {" + getKernelName(cNode) + "_call(dZ, offset_graph, columns_graph, value_graph_b,\n\
 bounds_b, segments_b),\n\
 edge_sddmm(dZ, X, offset_graph_f, columns_graph_f, value_graph,\n\
           bounds_f, global_nrows, segments_f),\n\
 torch::Tensor()};";
                } else
                {
                    // TODO add codegen for non-col tile
                    autoGradFunction += "\
            torch::Tensor value_graph_T = value_graph.index_select(0, global_transpose_perm.to(value_graph.device()));\n\
            return {" + getKernelName(cNode) + "_call(dZ, offset_graph, columns_graph,\n\
                                       value_graph_T),\n\
edge_sddmm(dZ, X, offset_graph, columns_graph, value_graph, bounds,\n\
           global_nrows, 1), torch::Tensor()};\n";
                }
        autoGradFunction += "\
        }\n\
    };";
            kernelCallCode.addCode(autoGradFunction);
                }
                if (outOfLoop)
                {
                    // TODO Check if the output name is res, if not then pass this along to the output
                    auto inGraphIndx = cNode->getInput(1)->getDataInfo()->getIndex();
                    std::string tempForwardAggrCall;
                    if (cNode->getOutput(0)->getName() == "res_n" || cNode->getOutput(0)->getName() == "t_iden_n")
                    {
                        tempForwardAggrCall =  "torch::Tensor t_iden_n = " + getKernelName(cNode)
                   + "_AutoGrad::apply(t_iden, 0);";
                        std::string aggrResStr = ", t_iden_n";
                        model.getCall()->addCode(aggrResStr);
                        std::string aggrResForward = ", torch::Tensor t_iden_n";
                        model.incForwardTensorArgs();
                        model.addForwardTensorArgName("t_iden_n");
                        model.getForwardCallInternal()->addCode(aggrResForward);
                    } else
                    {
                        tempForwardAggrCall =  "t_iden = " + getKernelName(cNode)
                   + "_AutoGrad::apply(t_iden, 0);"; // TODO: Always do 0 (for now, since it'll be used for all scenarios)
                    }

                    model.getInv()->addCode(tempForwardAggrCall);
                } else
                {
                    auto inGraphIndx = cNode->getInput(1)->getDataInfo()->getIndex();

                    std::string tempForwardAggrCall = "    if (ep % mod_v == 0) {\n\
      " + generateOutputString(cNode, outOfLoop) + " = " + getKernelName(cNode)
                    + "_AutoGrad::apply(" + cNode->getInput(0)->getName() +", attn, 0);\n\
    } else {\n\
      " + generateOutputString(cNode, outOfLoop) + " = " + getKernelName(cNode)
                    + "_AutoGrad::apply(" + cNode->getInput(0)->getName() +", attn, " + std::to_string(inGraphIndx) + ");\n\
    }";
                    model.getForward()->addCode(tempForwardAggrCall);
                }
            } else
            {
                bool isColTile = hasDOpt(cNode->getInput(1), COL_TILE_DOPT);
                if (encounteredAutograds.find(getKernelName(cNode)) == encounteredAutograds.end())
                {
                    encounteredAutograds.insert(getKernelName(cNode));
                    std::string autoGradFunction = ""
  
    "class " + getKernelName(cNode) + "_AutoGrad : public torch::autograd::Function<" + getKernelName(cNode) + "_AutoGrad> {\n\
    public:\n\
        static torch::Tensor forward(torch::autograd::AutogradContext *ctx,\n\
                                     torch::Tensor input_dense, int li) {\n\
            ctx->saved_data[\"li\"] = li;\n\
            torch::Tensor offset_graph = global_offset_graph[2 * li];\n\
            torch::Tensor columns_graph = global_columns_graph[2 * li];\n\
            torch::Tensor value_graph = global_value_graph[2 * li];\n";
                if (isColTile){
                    autoGradFunction += "        torch::Tensor bounds = global_bounds[2 * li];\n\
            int segments = global_segments[2 * li];\n\
             return " + getKernelName(cNode) + "_call(input_dense, offset_graph, columns_graph,\n\
                                value_graph, bounds, segments);\n";
                } else
                {
                    autoGradFunction += "        return " + getKernelName(cNode) + "_call(input_dense, offset_graph, columns_graph,\n\
                                      value_graph);\n";
                }
                autoGradFunction += "    }\n\
    \n\
        static torch::autograd::tensor_list\n\
        backward(torch::autograd::AutogradContext *ctx,\n\
                 torch::autograd::tensor_list grad_outputs) {\n\
            torch::Tensor input_dense = grad_outputs[0];\n\
            int li = ctx->saved_data[\"li\"].toInt();\n\
            torch::Tensor offset_graph = global_offset_graph[2 * li + 1];\n\
            torch::Tensor columns_graph = global_columns_graph[2 * li + 1];\n\
            torch::Tensor value_graph = global_value_graph[2 * li + 1];\n";
                if (isColTile){
                    autoGradFunction += "        torch::Tensor bounds = global_bounds[2 * li + 1];\n\
            int segments = global_segments[2 * li + 1];\n\
            return {" + getKernelName(cNode) + "_call(input_dense, offset_graph, columns_graph, value_graph, bounds, segments), torch::Tensor()};";
                } else
                {
                    autoGradFunction += "\
            return {" + getKernelName(cNode) + "_call(input_dense, offset_graph, columns_graph,\n\
                                       value_graph), torch::Tensor()};\n";
                }
        autoGradFunction += "\
        }\n\
    };";
            kernelCallCode.addCode(autoGradFunction);
                }
                if (outOfLoop)
                {
                    std::string tempForwardAggrCall;
                    // std::cout << "Name: " << cNode->getOutput(0)->getName() << std::endl;
                    if (cNode->getOutput(0)->getName() == "res_n" || cNode->getOutput(0)->getName() == "t_iden_n")
                    {
                        tempForwardAggrCall =  "torch::Tensor t_iden_n = " + getKernelName(cNode)
                   + "_AutoGrad::apply(t_iden, 0);";
                        std::string aggrResStr = ", t_iden_n";
                        model.getCall()->addCode(aggrResStr);
                        std::string aggrResForward = ", torch::Tensor t_iden_n";
                        model.incForwardTensorArgs();
                        model.addForwardTensorArgName("t_iden_n");
                        model.getForwardCallInternal()->addCode(aggrResForward);
                    } else
                    {
                        tempForwardAggrCall =  "t_iden = " + getKernelName(cNode)
                   + "_AutoGrad::apply(t_iden, 0);"; // TODO: Always do 0 (for now, since it'll be used for all scenarios)
                    }
                    model.getInv()->addCode(tempForwardAggrCall);

                    // std::string tempForwardAggrCall =  "t_iden = " + getKernelName(cNode)
                    // + "_AutoGrad::apply(t_iden, 0);";
                    // model.getInv()->addCode(tempForwardAggrCall);
                } else
                {
                    auto inGraphIndx = cNode->getInput(1)->getDataInfo()->getIndex();
                    std::string tempForwardAggrCall = "    if (ep % mod_v == 0) {\n\
      " + generateOutputString(cNode, outOfLoop) + " = " + getKernelName(cNode)
                    + "_AutoGrad::apply(" + cNode->getInput(0)->getName() +", 0);\n\
    } else {\n\
      " + generateOutputString(cNode, outOfLoop) + " = " + getKernelName(cNode)
                    + "_AutoGrad::apply(" + cNode->getInput(0)->getName() +", " + std::to_string(inGraphIndx) + ");\n\
    }";
                    model.getForward()->addCode(tempForwardAggrCall);

                    if (hasEdgeMulAggr &&  inGraphIndx != 0)
                    {
                        if (inGraphIndx == 1)
                        {
                            std::string tempForwardAggrCall_vals = "        torch::Tensor offset_graph_vals_b = global_offset_graph[1];\n\
        torch::Tensor columns_graph_vals_b = global_columns_graph[1];\n\
        torch::Tensor value_graph_vals_b = global_value_graph[1];\n";

                            if (isColTile){
                                tempForwardAggrCall_vals += "        torch::Tensor bounds_vals_b = global_bounds[1];\n\
        int segments_vals_b = global_segments[1];\n";
                                tempForwardAggrCall_vals += "torch::Tensor val_b = aggregate_edge_mul( norm_val, norm_val, offset_graph_vals_b, columns_graph_vals_b, value_graph_vals_b, bounds_vals_b, segments_vals_b).detach();";
                            } else
                            {
                               tempForwardAggrCall_vals += "torch::Tensor val_b = aggregate_edge_mul( norm_val, norm_val, offset_graph_vals_b, columns_graph_vals_b, value_graph_vals_b).detach();";
                            ;
                            }

                            model.getInv()->addCode(tempForwardAggrCall_vals);

                            std::string resetVal = "global_value_graph[1] = val_b;";
                            model.getInv()->addCode(resetVal);
                        }
                        std::string tempForwardAggrCall_vals = "        torch::Tensor offset_graph_vals" + std::to_string(inGraphIndx) + " = global_offset_graph[2 * " + std::to_string(inGraphIndx) + "];\n\
        torch::Tensor columns_graph_vals" + std::to_string(inGraphIndx) + " = global_columns_graph[2 * " + std::to_string(inGraphIndx) + "];\n\
        torch::Tensor value_graph_vals" + std::to_string(inGraphIndx) + " = global_value_graph[2 * " + std::to_string(inGraphIndx) + "];\n";

                        if (isColTile){
                            tempForwardAggrCall_vals += "        torch::Tensor bounds_vals" + std::to_string(inGraphIndx) + " = global_bounds[2 * " + std::to_string(inGraphIndx) + "];\n\
        int segments_vals" + std::to_string(inGraphIndx) + " = global_segments[2 * " + std::to_string(inGraphIndx) + "];\n";
                        } else
                        {
                            std::cout << "This unsup: AGGREGATE_EDGE_SUM_OP" << std::endl;
                            tempForwardAggrCall_vals += "unsupported\n";
                        }

                        tempForwardAggrCall_vals += "torch::Tensor val" + std::to_string(inGraphIndx)
                        + " = aggregate_edge_mul( norm_val, norm_val, offset_graph_vals" + std::to_string(inGraphIndx)
                        + ", columns_graph_vals" + std::to_string(inGraphIndx) + ", value_graph_vals"
                        + std::to_string(inGraphIndx) + ", bounds_vals" + std::to_string(inGraphIndx)
                        + ", segments_vals" + std::to_string(inGraphIndx) + ").detach();";
                        model.getInv()->addCode(tempForwardAggrCall_vals);

                        std::string resetVal = "global_value_graph[2 * " + std::to_string(inGraphIndx) + "] = val" + std::to_string(inGraphIndx) + ";";
                        model.getInv()->addCode(resetVal);

                        std::string tempForwardAggrCall_vals_b = "        torch::Tensor offset_graph_vals" + std::to_string(inGraphIndx) + "_b = global_offset_graph[2 * " + std::to_string(inGraphIndx) + "+1];\n\
        torch::Tensor columns_graph_vals" + std::to_string(inGraphIndx) + "_b = global_columns_graph[2 * " + std::to_string(inGraphIndx) + "+1];\n\
        torch::Tensor value_graph_vals" + std::to_string(inGraphIndx) + "_b = global_value_graph[2 * " + std::to_string(inGraphIndx) + "+1];\n";

                        if (isColTile){
                            tempForwardAggrCall_vals_b += "        torch::Tensor bounds_vals" + std::to_string(inGraphIndx) + "_b = global_bounds[2 * " + std::to_string(inGraphIndx) + "+1];\n\
        int segments_vals" + std::to_string(inGraphIndx) + "_b = global_segments[2 * " + std::to_string(inGraphIndx) + "+1];\n";
                        } else
                        {
                            std::cout << "This unsup: AGGREGATE_EDGE_SUM_OP" << std::endl;
                            tempForwardAggrCall_vals_b += "unsupported\n";
                        }

                        tempForwardAggrCall_vals_b += "torch::Tensor val" + std::to_string(inGraphIndx)
                        + "_b = aggregate_edge_mul( norm_val, norm_val, offset_graph_vals" + std::to_string(inGraphIndx)
                        + "_b, columns_graph_vals" + std::to_string(inGraphIndx) + "_b, value_graph_vals"
                        + std::to_string(inGraphIndx) + "_b, bounds_vals" + std::to_string(inGraphIndx)
                        + "_b, segments_vals" + std::to_string(inGraphIndx) + "_b).detach();";
                        model.getInv()->addCode(tempForwardAggrCall_vals_b);

                        std::string resetVal_b = "global_value_graph[2 * " + std::to_string(inGraphIndx) + "+1] = val" + std::to_string(inGraphIndx) + "_b;";
                        model.getInv()->addCode(resetVal_b);
                    }

                }
            }
        } else if (cNode->getOp() == AGGREGATE_MUL_SUM_DIRECT)
        {
            bool isColTile = hasDOpt(cNode->getInput(1), COL_TILE_DOPT);

            if (outOfLoop)
            {
                auto inGraphIndx = cNode->getInput(1)->getDataInfo()->getIndex();
                std::string directAggrCall = "  torch::Tensor offset_graph_ones = global_offset_graph[2 * " + std::to_string(inGraphIndx) + "];\n\
  torch::Tensor columns_graph_ones = global_columns_graph[2 * " + std::to_string(inGraphIndx) + "];\n\
  torch::Tensor value_graph_ones = global_value_graph[2 * " + std::to_string(inGraphIndx) + "];\n";

                if (isColTile){
  
                    directAggrCall += "  torch::Tensor bounds_ones = global_bounds[2 * " + std::to_string(inGraphIndx) + "];\n\
  int segments_ones = global_segments[2 * " + std::to_string(inGraphIndx) + "];\n"
                    + generateOutputString(cNode, outOfLoop) + " = " + getKernelName(cNode) + "_call(" + cNode->getInput(0)->getName() + ", offset_graph_ones, columns_graph_ones,\n\
    value_graph_ones, bounds_ones, segments_ones);\n";
                } else
                {
                    directAggrCall += "  " + generateOutputString(cNode, outOfLoop) +" = " + getKernelName(cNode) + "_call(" + cNode->getInput(0)->getName() + ", offset_graph_ones, columns_graph_ones,\n\
    value_graph_ones);\n";
                }
                model.getInv()->addCode(directAggrCall);

            } else
            {
                auto inGraphIndx = cNode->getInput(1)->getDataInfo()->getIndex();
                std::string directAggrCall = "            torch::Tensor offset_graph_ones = global_offset_graph[2 * " + std::to_string(inGraphIndx) + "];\n\
          torch::Tensor columns_graph_ones = global_columns_graph[2 * " + std::to_string(inGraphIndx) + "];\n\
          torch::Tensor value_graph_ones = global_value_graph[2 * " + std::to_string(inGraphIndx) + "];\n";

                if (isColTile){
                    directAggrCall += "        torch::Tensor bounds_ones = global_bounds[2 * " + std::to_string(inGraphIndx) + "];\n\
        int segments_ones = global_segments[2 * " + std::to_string(inGraphIndx) + "];\n        "
                    + generateOutputString(cNode, outOfLoop) + " = " + getKernelName(cNode) + "_call(" + cNode->getInput(0)->getName() + ", offset_graph_ones, columns_graph_ones,\n\
                            value_graph_ones, bounds_ones, segments_ones);\n";
                } else
                {
                    directAggrCall += "        " + generateOutputString(cNode, outOfLoop) +" = " + getKernelName(cNode) + "_call(" + cNode->getInput(0)->getName() + ", offset_graph_ones, columns_graph_ones,\n\
                                  value_graph_ones);\n";
                }
                model.getForward()->addCode(directAggrCall);
            }
        } else if (cNode->getOp() == POWER_OP)
        {
            if (cNode->getParam(0) == "0.000000")
            {
                cNode->setParam(0, "-1");
            }
            if (outOfLoop)
            {
                std::string powerCall = "   " + generateOutputString(cNode, outOfLoop) + " = torch::pow(" + cNode->getInput(0)->getName() + ", " + cNode->getParam(0) + ").detach();";
                model.getInv()->addCode(powerCall);
                // TODO: Temporary method to add kernel call
                std::string tempPassDegree = "," + cNode->getOutput(0)->getName();
                model.getCall()->addCode(tempPassDegree);

                std::string tempPassDegreeForward = ", torch::Tensor " + cNode->getOutput(0)->getName();
                model.incForwardTensorArgs();
                model.addForwardTensorArgName(cNode->getOutput(0)->getName());
                model.getForwardCallInternal()->addCode(tempPassDegreeForward);
            } else
            {
                std::string powerCall = "        " + generateOutputString(cNode, outOfLoop) + " = torch::pow(" + cNode->getInput(0)->getName() + ", " + cNode->getParam(0) + ");";
                model.getForward()->addCode(powerCall);
            }

        } else if (cNode->getOp() == ROW_BROADCAST_OP)
        {
            if (outOfLoop)
            {
                std::string rbCall;
                if (cNode->getInput(1)->getName() == "res_n")
                {
                    rbCall = "  t_iden_n = (" + cNode->getInput(0)->getName() + " * t_iden_n).detach();";
                } else
                {
                    rbCall = "  t_iden = (" + cNode->getInput(0)->getName() + " * t_iden).detach();";
                }

                model.getInv()->addCode(rbCall);
            } else
            {
                std::string rbCall = "        " + generateOutputString(cNode, outOfLoop) + " = " + cNode->getInput(0)->getName()
            + " * " + cNode->getInput(1)->getName() + ";";
                model.getForward()->addCode(rbCall);
            }

        } else if (cNode->getOp() == NON_LNR_OP_RELU || cNode->getOp() == NON_LNR_OP_ELU)
        {
            std::string fn = cNode->getOp() == NON_LNR_OP_ELU ? "torch::elu" : "torch::relu";
            std::string reluCall = "        " + generateOutputString(cNode, outOfLoop) + " = " + fn + "(" + cNode->getInput(0)->getName() + ");";
            model.getForward()->addCode(reluCall);
            std::string dropoutCall = "        " + generateOutputString(cNode, outOfLoop) + " = torch::dropout(" + generateOutputString(cNode, outOfLoop) + ", " + std::to_string(GALAFEContext::dropout) + ", this->is_training());";
            model.getForward()->addCode(dropoutCall);
        } else if (cNode->getOp() == NON_LNR_OP_LEAKY_RELU)
        {
            if (encounteredAutograds.find(getKernelName(cNode)) == encounteredAutograds.end())
            {
                encounteredAutograds.insert(getKernelName(cNode));
                std::string leakyReluInit = "     torch::nn::LeakyReLU leaky_relu(torch::nn::LeakyReLUOptions().negative_slope(0.2));";
                model.getForward()->addCode(leakyReluInit);
            }
            std::string leakyReluCall = "     " + generateOutputString(cNode, outOfLoop) + " = leaky_relu->forward(" + cNode->getInput(0)->getName() + ");";
            model.getForward()->addCode(leakyReluCall);
        } else if (cNode->getOp() == FFN_OP)
        {
            // TODO Check if input and output names are same. If they are use same, if not use something else
            if (fcCount == 0)
            {
                std::string inSize1 = "int size" + std::to_string(fcCount);
                model.getInitCall()->addCode(inSize1);

                std::string inSize2 = "int size" + std::to_string(fcCount + 1);
                model.getInitCall()->addCode(inSize2);

                std::string fcDef = "torch::nn::Linear fc" + std::to_string(fcCount) + "{nullptr};";
                model.getDef()->addCode(fcDef);

                std::string fcInit = "fc" + std::to_string(fcCount) + " = register_module(\"fc"
                + std::to_string(fcCount) + "\", torch::nn::Linear(size" + std::to_string(fcCount)
                + ", size" + std::to_string(fcCount + 1) + "));";
                model.getInit()->addCode(fcInit);

                // TODO need some way to add the inputs to the function call
                inputSizes.push_back(cNode->getInput(1)->getDataInfo()->getDimRow());
                inputSizes.push_back(cNode->getInput(1)->getDataInfo()->getDimCol());

                // TODO add the inputs to the forward call based on the actual inputs
                std::string forwardCall;
                if (generateOutputString(cNode, outOfLoop) == "res_n" && cNode->getInput(0)->getName() == "t_iden")
                {
                    forwardCall = generateOutputString(cNode, outOfLoop) + " = fc" + std::to_string(fcCount) + "->forward(" + cNode->getInput(0)->getName() + "_n);";
                } else if (cNode->getInput(0)->getName() == "")
                {
                    forwardCall = generateOutputString(cNode, outOfLoop) + " = fc" + std::to_string(fcCount) + "->forward(t_iden);";
                }
                else
                {
                    forwardCall = generateOutputString(cNode, outOfLoop) + " = fc" + std::to_string(fcCount) + "->forward(" + cNode->getInput(0)->getName() + ");";
                }
                model.getForward()->addCode(forwardCall);
            } else
            {
                std::string inSize2 = "int size" + std::to_string(fcCount + 1);
                model.getInitCall()->addCode(inSize2);

                std::string fcDef = "torch::nn::Linear fc" + std::to_string(fcCount) + "{nullptr};";
                model.getDef()->addCode(fcDef);

                std::string fcInit = "fc" + std::to_string(fcCount) + " = register_module(\"fc"
                + std::to_string(fcCount) + "\", torch::nn::Linear(size" + std::to_string(fcCount)
                + ", size" + std::to_string(fcCount + 1) + "));";
                model.getInit()->addCode(fcInit);

                inputSizes.push_back(cNode->getInput(1)->getDataInfo()->getDimCol());

                // std::cout << "cc1: " << generateOutputString(cNode, outOfLoop) << " -- " << std::to_string(fcCount) << std::endl;
                std::string forwardCall = generateOutputString(cNode, outOfLoop) + " = fc" + std::to_string(fcCount) + "->forward(" + cNode->getInput(0)->getName() + ");";
                model.getForward()->addCode(forwardCall);
  
            }
            fcCount++;
        }  else if (cNode->getOp() == FFN_OP_REPEAT)
        {
            // std::cout << "cc2: " << generateOutputString(cNode, outOfLoop) << " -- " << std::to_string(fcCount) << std::endl;
            std::string forwardCall = generateOutputString(cNode, outOfLoop) + " = fc" + std::to_string(fcCount - 1) + "->forward(" + cNode->getInput(0)->getName() + ");";
            model.getForward()->addCode(forwardCall);
        } else if (cNode->getOp() == FFN_OP_EDGE)
        {
            std::string fcDef = "torch::nn::Linear efc" + std::to_string(fcEdgeCount) + "{nullptr};";
            model.getDef()->addCode(fcDef);

            std::string fcInit = "efc" + std::to_string(fcEdgeCount) + " = register_module(\"efc"
            + std::to_string(fcEdgeCount) + "\", torch::nn::Linear(size" + std::to_string(fcCount)
            + ", 1));";
            model.getInit()->addCode(fcInit);

            std::string forwardCall = generateOutputString(cNode, outOfLoop) + " = efc" + std::to_string(fcEdgeCount) + "->forward(" + cNode->getInput(0)->getName() + ");";
            model.getForward()->addCode(forwardCall);
            fcEdgeCount++;
        }  else if (cNode->getOp() == FFN_OP_SELF)
        {
            std::string fcDef = "torch::nn::Linear sfc" + std::to_string(fcSelfCount) + "{nullptr};";
            model.getDef()->addCode(fcDef);

            if (fcSelfCount == 0)
            {
                std::string resInit = "res = t_iden;";
                model.getForward()->addCode(resInit);
            }

            std::string fcInit = "sfc" + std::to_string(fcSelfCount) + " = register_module(\"sfc"
            + std::to_string(fcSelfCount) + "\", torch::nn::Linear(size" + std::to_string(fcCount - 1)
             + ", size" + std::to_string(fcCount) + "));";
            model.getInit()->addCode(fcInit);

            // TODO Temp fix
            // std::string forwardCall = generateOutputString(cNode, outOfLoop) + " = sfc" + std::to_string(fcEdgeCount) + "->forward(" + cNode->getInput(0)->getName() + ");";
            std::string forwardCall = "res = sfc" + std::to_string(fcSelfCount) + "->forward(res);";
            model.getForward()->addCode(forwardCall);
            fcSelfCount++;
        } else if (cNode->getOp() == SCALAR_ADD_EPS_MULTIPLY_OP)
        {
            std::string epDef = "torch::Tensor eps" + std::to_string(epCount) + "{nullptr};";
            model.getDef()->addCode(epDef);

            std::string epInit = "eps" + std::to_string(epCount) + " = register_parameter(\"eps"
            + std::to_string(epCount) + "\", torch::tensor({(float)" + cNode->getParam(0) + "}));";
            model.getInit()->addCode(epInit);

            std::string forwardCall = generateOutputString(cNode, outOfLoop) + " = (1 + eps" + std::to_string(epCount) + ") * " + cNode->getInput(0)->getName() + ";";
            if (outOfLoop)
            {
                model.getInv()->addCode(forwardCall);
            } else
            {
                model.getForward()->addCode(forwardCall);
            }
            epCount++;
        } else if (cNode->getOp() == ADD_OP)
        {
            std::string forwardCall = generateOutputString(cNode, outOfLoop) + " = " + cNode->getInput(0)->getName() + " + " + cNode->getInput(1)->getName() + ";";
            if (outOfLoop)
            {
                model.getInv()->addCode(forwardCall);
            } else
            {
                model.getForward()->addCode(forwardCall);
            }
        } else if (cNode->getOp() == ONES_OP)
        {
            auto outputInfo = cNode->getOutput(0)->getDataInfo();

            std::string rowDims = processDims(outputInfo->getDimRow());
            std::string colDims = processDims(outputInfo->getDimCol());

            if (outOfLoop)
            {
                // TODO eventually use a device specific function for this.
                std::string tempOptionsOnes = "    auto options_" + cNode->getOutput(0)->getName() +" = torch::TensorOptions()\n\
                       .dtype(torch::kFloat)\n\
                       .requires_grad(false)\n\
                       " + deviceOpt() + ";";
                model.getInv()->addCode(tempOptionsOnes);

                std::string onesCall =  generateOutputString(cNode, outOfLoop) + " = torch::ones({" + rowDims
                + ", " + colDims + "}, options_" + cNode->getOutput(0)->getName() + ");";
                model.getInv()->addCode(onesCall);
            } else
            {
                // TODO eventually use a device specific function for this.
                std::string tempOptionsOnes = "    auto options_" + cNode->getOutput(0)->getName() +" = torch::TensorOptions()\n\
                       .dtype(torch::kFloat)\n\
                       .requires_grad(false)\n\
                       " + deviceOpt() + ";";
                model.getForward()->addCode(tempOptionsOnes);

                std::string onesCall =  generateOutputString(cNode, outOfLoop) + " = torch::ones({" + rowDims
                + ", " + colDims + "}, options_" + cNode->getOutput(0)->getName() + ");";
                model.getForward()->addCode(onesCall);
            }
        } else if (cNode->getOp() == FULL_OP)
        {
            auto outputInfo = cNode->getOutput(0)->getDataInfo();

            std::string rowDims = processDims(outputInfo->getDimRow());
            std::string colDims = processDims(outputInfo->getDimCol());

            if (outOfLoop)
            {
                // TODO eventually use a device specific function for this.
                std::string tempOptionsOnes = "    auto options_" + cNode->getOutput(0)->getName() +" = torch::TensorOptions()\n\
                       .dtype(torch::kFloat)\n\
                       .requires_grad(false)\n\
                       " + deviceOpt() + ";";
                model.getInv()->addCode(tempOptionsOnes);

                std::string onesCall =  generateOutputString(cNode, outOfLoop) + " = torch::full({" + rowDims
                + ", " + colDims + "}, " + cNode->getParam(0) + " * global_segments[0], options_" + cNode->getOutput(0)->getName() + ");";
                model.getInv()->addCode(onesCall);
            } else
            {
                // TODO eventually use a device specific function for this.
                std::string tempOptionsOnes = "    auto options_" + cNode->getOutput(0)->getName() +" = torch::TensorOptions()\n\
                       .dtype(torch::kFloat)\n\
                       .requires_grad(false)\n\
                       " + deviceOpt() + ";";
                model.getForward()->addCode(tempOptionsOnes);

                std::string onesCall =  generateOutputString(cNode, outOfLoop) + " = torch::full({" + rowDims
                + ", " + colDims + "}, " + cNode->getParam(0) + " * global_segments[0], options_" + cNode->getOutput(0)->getName() + ");";
                model.getForward()->addCode(onesCall);
            }
        }
    }

// TODO Put this in the common codegen? Doesn't seem to have any context specific content yet
    void generateCode(std::vector<CIRNode*>& program,
        std::vector<TransformEdge*>& transforms)
    {
        std::vector<int> inputSizes;
        int fcCount = 0;
        int fcEdgeCount = 0;
        int fcSelfCount = 0;
        int epCount = 0;
        bool hasFFNEdgeUpdate = false;
        bool hasEdgeMulAggr = false;

        // TODO add data transformations before data preparation.
        //  Should come from a middle end transformation.
        std::unordered_set<std::string> encounteredAutograds;
        for (int i = 0; i < program.size(); i++)
        {
            CIRNode* outNode = program[i];
            auto oNode = dynamic_cast<ComputeNode*>(outNode);
            if (oNode)
            {
                generateOpCode(oNode, fcCount, fcEdgeCount, fcSelfCount,
                    epCount, true, hasFFNEdgeUpdate, hasEdgeMulAggr, encounteredAutograds, inputSizes, transforms);

                // Generate the transfer code after the load operation
                if (oNode->getOp() == LOAD_OP)
                {
                    this->dataPrep(program);
                }

            } else {
                std::string modelDef = "struct GALAGNN : torch::nn::Module {";
                model.getDef()->addCode(modelDef);
                std::string modelCall =  "GALAGNN(";
                model.getInitCall()->addCode(modelCall);

                // TODO generate this based on the program
                std::string tempFowradCallPre = "std::vector<torch::Tensor>\n\
forward(torch::Tensor t_iden";
                model.getForwardCallPre()->addCode(tempFowradCallPre);
                model.incForwardTensorArgs();
                std::string tempFowradCallPost = ", int ep, int mod_v){\n";
                model.getForwardCallPost()->addCode(tempFowradCallPost);
                // std::string iden_n_init = "torch::Tensor t_iden_n = t_iden;\n";
                // model.getForward()->addCode(iden_n_init);

                std::unordered_set<std::string> encounteredTensors;
                // std::string resInit = "torch::Tensor res = input_dense;";
                // model.getForward()->addCode(resInit);

                auto loopNode = dynamic_cast<TrainingLoopNode*>(outNode);

                std::string numIterCode = "int num_iters = " + std::to_string(loopNode->getIter()) + ";";
                preCode.addCode(numIterCode);

                for (int ix = 0; ix < loopNode->getLoopNodeNum(); ix++)
                {
                    CIRNode* inNode = loopNode->getNode(ix);
                    auto cNode = dynamic_cast<ComputeNode*>(inNode);
                    if (encounteredTensors.find(generateOutputString(cNode, false)) == encounteredTensors.end())
                    {
                        encounteredTensors.insert(generateOutputString(cNode, false));
                        std::string initTensor = "    torch::Tensor " + generateOutputString(cNode, false) + ";";
                        model.getForwardCallPost()->addCode(initTensor);
                    }

                    generateOpCode(cNode, fcCount, fcEdgeCount, fcSelfCount,
                        epCount, false, hasFFNEdgeUpdate, hasEdgeMulAggr, encounteredAutograds, inputSizes, transforms);
                }
                CIRNode* inNode = loopNode->getNode(loopNode->getLoopNodeNum()-1);
                auto cNode = dynamic_cast<ComputeNode*>(inNode);
                // TODO Change this. (Remove and replace)
                std::string tempReturn = "return {" + cNode->getOutput(0)->getName() + "};";
                model.getForward()->addCode(tempReturn);

                std::string closeForward = "    }\n"
                                           "};";
                model.getForward()->addCode(closeForward);

                std::string closeInit = "   }";
                model.getInit()->addCode(closeInit);

                std::string closeInitCall = "){\n";
                model.getInitCall()->addCode(closeInitCall);

                std::string tempModelInit = "auto net = std::make_shared<GALAGNN>(";
                for (int ei = 0; ei < inputSizes.size(); ei++)
                {
                    tempModelInit += processDims(inputSizes[ei]);
                    if (ei < inputSizes.size() - 1)
                    {
                        tempModelInit += ", ";
                    }
                }
                tempModelInit += ");";
                model.getPreCall()->addCode(tempModelInit);

                std::string modelTransfer = "net->to(device);";
                model.getPreCall()->addCode(modelTransfer);

                // Initialize weights with Xavier/Glorot uniform (matching PyG's GATConv)
                std::string initCode = "for (auto& p : net->named_parameters()) {\n\
    if (p.value().dim() >= 2) {\n\
        torch::nn::init::xavier_uniform_(p.value());\n\
    } else {\n\
        torch::nn::init::zeros_(p.value());\n\
    }\n\
}\n";
                model.getPreCall()->addCode(initCode);

                // Differential-testing hook: GALA_SAVE_INIT=<file> saves the freshly
                // initialized parameters; GALA_LOAD_INIT=<file> replaces them, so two
                // backends can be run from identical weights and compared epoch by epoch.
                std::string initHook = "if (const char *p = std::getenv(\"GALA_SAVE_INIT\")) { torch::save(net, p); }\n\
if (const char *p = std::getenv(\"GALA_LOAD_INIT\")) { torch::load(net, p, device); }\n";
                model.getPreCall()->addCode(initHook);

                if (loopNode->getOptimizer() == ADAM)
                {
                    std::string optmCode = "torch::optim::Adam optimizer(\n\
    net->parameters(), torch::optim::AdamOptions(" + std::to_string(loopNode->getLearningRate()) +").weight_decay(" + std::to_string(loopNode->getWeightDecay()) + "));\n";
                    model.getPreCall()->addCode(optmCode);
                } else
                {
                    std::cout << "Optimizer not supported." << std::endl;
                }

                // Do validation step
                int testStep =  loopNode->getTestStep();
                if (testStep > 1){
                    std::string initTrinStepsStr = " int mod_v = " + std::to_string(testStep) + ";\n";
                    model.getPreCall()->addCode(initTrinStepsStr);
                } else {
                    std::string initTrinStepsStr = " int mod_v = 1;\n";
                    model.getPreCall()->addCode(initTrinStepsStr);
                }

                std::string skipEpochsStr = " int skip_cache_warmup = 5;\n";
                model.getPreCall()->addCode(skipEpochsStr);

                std::string accVarsStr = " float train_acc, test_acc, val_acc;\n";
                model.getPreCall()->addCode(accVarsStr);

                std::string eval = "Evaluator<GALAGNN> evaluator(skip_cache_warmup);\n";
                if (loopNode->getLossFunc() == MSE)
                {
                    eval += "evaluator.use_auc_metric();\n";
                }
                eval += "  evaluator.begin();";
                model.getPreCall()->addCode(eval);

                if (GALAFEContext::print_accuracy)
                {
                    std::string accInitStr = "  float max_acc = 0;\n";
                    model.getPreCall()->addCode(accInitStr);
                }

                std::string tempTrainLoopPreCall = " for (size_t epoch = 1; epoch <= num_iters; ++epoch) {\n\
    // Reset gradients.\n\
    optimizer.zero_grad();\n\
    // Execute the model on the input data.\n\
    " + syncCall() + "\n\
    evaluator.begin_forward();\n\
    torch::Tensor prediction =\n\
        net->forward(t_iden";

                // Build tempTrainLoopPostCall - shared prefix
                std::string tempTrainLoopPostCall = ", epoch, mod_v)[0];\n\
    " + syncCall() + "\n\
    evaluator.end_forward();\n\
    " + syncCall() + "\n\
    evaluator.begin_train();\n";

                if (loopNode->getLossFunc() == MSE)
                {
                    // The reconstruction target must be the RAW features:
                    // trainingInvariantCodeMotion may hoist the first layer's
                    // norm/aggregate transforms onto t_iden before the loop, so
                    // capture the untransformed tensor first (emitted at the end
                    // of preCode, which precedes the hoisted block).
                    std::string targetCapture = "torch::Tensor t_target = t_iden;";
                    preCode.addCode(targetCapture);
                    // Unsupervised autoencoder: full-graph feature reconstruction.
                    // No mask restriction and no gradient clipping (matches the PyGOD
                    // GCNAE reference). The per-node mean squared reconstruction error,
                    // taken from the same train-mode forward used for the loss and
                    // captured before the optimizer step, is the nomination score;
                    // ROC-AUC of that score against the binary interest labels is the
                    // reported metric.
                    tempTrainLoopPostCall += "\
    torch::Tensor d_loss = torch::mse_loss(prediction, t_target.detach());\n\
    torch::Tensor score_ae = (prediction.detach() - t_target.detach()).pow(2).mean(1);\n\
    d_loss.backward();\n\
    optimizer.step();\n\
    " + syncCall() + "\n\
    evaluator.end_train();\n\
    evaluator.test_auc(score_ae, t_labs, t_train_mask, t_valid_mask, t_test_mask, train_acc, val_acc, test_acc);\n\
        evaluator.train_step_report(epoch, " + std::to_string(GALAFEContext::log_interval) + ", d_loss, train_acc, test_acc, val_acc);\n";
                } else
                {
                    tempTrainLoopPostCall += "\
    torch::Tensor prediction_train = prediction.index({t_train_mask});\n\
    torch::Tensor labels_train = t_labs.index({t_train_mask});\n\
    auto criterion = torch::nn::CrossEntropyLoss();\n\
    torch::Tensor d_loss = criterion(prediction_train, labels_train);\n\
    d_loss.backward();\n" + (GALAFEContext::grad_clip > 0 ? "\
    torch::nn::utils::clip_grad_norm_(net->parameters(), " + std::to_string(GALAFEContext::grad_clip) + ");\n" : "") + "\
    optimizer.step();\n\
    " + syncCall() + "\n\
    evaluator.end_train();\n\
    net->eval();\n\
    " + generateEvaluatorTestCall() + "\n\
        evaluator.train_step_report(epoch, " + std::to_string(GALAFEContext::log_interval) + ", d_loss, train_acc, test_acc, val_acc);\n\
    net->train();\n";
                }

                if (GALAFEContext::print_accuracy && loopNode->getLossFunc() != MSE)
                {
                    tempTrainLoopPostCall += "    torch::Tensor prediction_test = prediction.index({t_test_mask});\n\
    torch::Tensor labels_test = t_labs.index({t_test_mask});\n\
    auto [pred_val, pred_idx] = torch::max({prediction_test}, 1);\n\
    auto correct = torch::sum(pred_idx == labels_test);\n\
    float acc = (correct.item<val_t>() * 100.0 / labels_test.sizes()[0]);\n\
    if (max_acc<acc){\n\
        max_acc = acc;\n\
    }\n";
                }
                if (GALAFEContext::print_accuracy && loopNode->getLossFunc() == MSE)
                {
                    // keep max_acc defined for the final print: eval AUC at best val
                    tempTrainLoopPostCall += "    max_acc = evaluator.best_test();\n";
                }

                tempTrainLoopPostCall += "  }";

                model.getPreCall()->addCode(tempTrainLoopPreCall);
                model.getPostCall()->addCode(tempTrainLoopPostCall);
            }
        }

        // Add evaluator end and report after the training loop
        std::string evalEnd = "  evaluator.end();\n\
  evaluator.report();";
        postCode.addCode(evalEnd);

        if (GALAFEContext::print_accuracy)
        {
            std::string printAcc = "  std::cout << max_acc << std::endl;\n";
            postCode.addCode(printAcc);
        }
        else if (GALAFEContext::print_memory)
        {
            std::string printMem = "  std::cout << printMemoryUsage() << std::endl;\n";
            postCode.addCode(printMem);
        }

        std::string closeMain = "}";
        postCode.addCode(closeMain);
    }

    GALAContext* getContext()
    {
        return this->context;
    }

    // Handle the stream to write to
    void openStream(const filesystem::path& outputPath)
    {
        std::string cmakePath = outputPath / "CMakeLists.txt";
        this->outStreamCMake = std::ofstream(cmakePath);

        std::string modelPath = outputPath / modelFileName();
        this->outStreamModel = std::ofstream(modelPath);
    }

    void closeStream()
    {
        this->outStreamModel.close();
        this->outStreamCMake.close();
    }

    void writeCode(Code &code, std::ofstream &outStream, const std::string &end = "\n", bool skipFirstEnd = false,
        bool skip2ndLastEnd = false){
        for (int ix = 0; ix < code.getNum(); ix++){
            auto codeLine = code.atLine(ix);
            outStream << *codeLine;
            // skip adding end to first
            if (skipFirstEnd and ix == 0)
            {
                continue;
            } else if (skip2ndLastEnd and ix >= code.getNum() - 2)
            {
                continue;
            } else
            {
                outStream << end;
            }
        }
    }

    // ---- Backend hooks. Defaults reproduce the CUDA backend; CPUGenerator overrides. ----
    // Device clause appended to torch::TensorOptions chains in generated code.
    virtual std::string deviceOpt() { return ".device(torch::kCUDA, 0)"; }
    // Host/device synchronization statement emitted around timed regions.
    virtual std::string syncCall() { return "cudaDeviceSynchronize();"; }
    // Name of the generated model source file.
    virtual std::string modelFileName() { return "gala.cu"; }
    // Materialize the host-side transpose permutation std::vector<int> perm_data_<name>
    // as torch::Tensor t_perm_<name> on the target device. nvalsExpr is a C++ expression.
    virtual std::string transposePermToDevice(const std::string& name, const std::string& nvalsExpr)
    {
        std::string r;
        r += "    int *dev_perm_" + name + ";\n";
        r += "    CUDA_CHECK(cudaMalloc((void**)&dev_perm_" + name + ", " + nvalsExpr + " * sizeof(int)));\n";
        r += "    CUDA_CHECK(cudaMemcpy(dev_perm_" + name + ", perm_data_" + name + ".data(),\n";
        r += "        " + nvalsExpr + " * sizeof(int), cudaMemcpyHostToDevice));\n";
        r += "    torch::Tensor t_perm_" + name + " = torch::from_blob(dev_perm_" + name + ",\n";
        r += "        {(int64_t)" + nvalsExpr + "},\n";
        r += "        torch::TensorOptions().dtype(torch::kInt).requires_grad(false).device(torch::kCUDA, 0));\n";
        return r;
    }
    // Code that publishes graph <index> (host CSR adj<index>, or the tiled tensors of
    // <name> when isColTile) as torch tensors t_offsets<index>/t_cols<index>/t_vals<index>
    // (plus _b twins when directed) and pushes them into the global_* arrays.

    // Default (CUDA) implementation of graphTransferCode: cudaMalloc/cudaMemcpy the CSR
    // arrays and wrap them as CUDA tensors.
    virtual std::string graphTransferCode(int index, const std::string& name, bool isColTile, bool directed)
    {
        std::string N = std::to_string(index);
        auto one = [&](const std::string& sfx) {
            std::string c;
            c += "  int *dA_csrOffsets" + N + sfx + ", *dA_columns" + N + sfx + "; \n\
  float *dA_values" + N + sfx + ";\n\
\n\
  CUDA_CHECK(cudaMalloc((void **)&dA_columns" + N + sfx + ", nvals" + N + " * sizeof(int)));\n\
  CUDA_CHECK(cudaMalloc((void **)&dA_values" + N + sfx + ", nvals" + N + " * sizeof(float)));\n";
            if (isColTile)
            {
                c += "\n\
  CUDA_CHECK(cudaMalloc((void **)&dA_csrOffsets" + N + sfx + ", (nrows + 1) * segments_" + name + sfx + " * sizeof(int)));\n\
\n\
  CUDA_CHECK(cudaMemcpy(dA_csrOffsets" + N + sfx + ", offset_ptr_" + name + sfx + ",\n\
                        (nrows + 1) * segments_" + name + sfx + " * sizeof(int), cudaMemcpyHostToDevice));\n\
  CUDA_CHECK(cudaMemcpy(dA_columns" + N + sfx + ", col_ptr_" + name + sfx + ", nvals" + N + " * sizeof(int),\n\
                        cudaMemcpyHostToDevice));\n\
  CUDA_CHECK(cudaMemcpy(dA_values" + N + sfx + ", val_ptr_" + name + sfx + ", nvals" + N + " * sizeof(float),\n\
                        cudaMemcpyHostToDevice));\n\
  torch::Tensor t_offsets" + N + sfx + " =\n\
      torch::from_blob(dA_csrOffsets" + N + sfx + ", {(nrows+ 1) * segments_" + name + sfx + "}, options_cu_int);\n";
            } else
            {
                c += "  CUDA_CHECK(cudaMalloc((void **)&dA_csrOffsets" + N + sfx + ", (nrows + 1) * sizeof(int)));\n\
\n\
  CUDA_CHECK(cudaMemcpy(dA_csrOffsets" + N + sfx + ", adj" + N + sfx + ".offset_ptr(),\n\
                        (nrows + 1) * sizeof(int), cudaMemcpyHostToDevice));\n\
  CUDA_CHECK(cudaMemcpy(dA_columns" + N + sfx + ", adj" + N + sfx + ".ids_ptr(), nvals" + N + " * sizeof(int),\n\
                        cudaMemcpyHostToDevice));\n\
  CUDA_CHECK(cudaMemcpy(dA_values" + N + sfx + ", adj" + N + sfx + ".vals_ptr(), nvals" + N + " * sizeof(float),\n\
                        cudaMemcpyHostToDevice));\n\
  torch::Tensor t_offsets" + N + sfx + " =\n\
      torch::from_blob(dA_csrOffsets" + N + sfx + ", {nrows+ 1}, options_cu_int);\n";
            }
            c += "  torch::Tensor t_cols" + N + sfx + " = torch::from_blob(dA_columns" + N + sfx + ", {nvals" + N + "}, options_cu_int);\n\
\n\
  torch::Tensor t_vals" + N + sfx + " =\n\
      torch::from_blob(dA_values" + N + sfx + ", {nvals" + N + "}, options_cu_float_ngrad);\n";
            c += "  global_offset_graph.push_back(t_offsets" + N + sfx + ");\n\
  global_columns_graph.push_back(t_cols" + N + sfx + ");\n\
  global_value_graph.push_back(t_vals" + N + sfx + ");\n";
            return c;
        };
        std::string code = one("");
        if (!directed)
        {
            // Undirected: the backward pass reuses the forward graph.
            code += "  global_offset_graph.push_back(t_offsets" + N + ");\n\
    global_columns_graph.push_back(t_cols" + N + ");\n\
    global_value_graph.push_back(t_vals" + N + ");\n";
        } else
        {
            code += one("_b");
        }
        return code;
    }

    // Walk the program and emit, once per distinct CSR input, the code that publishes
    // it to the global_* arrays (forward graph at 2*li, backward graph at 2*li+1).
    void generateTransferCodeForUniqueInput(ComputeNode* cNode,
        std::unordered_set<std::string> &encounteredStrings, bool &defaultLoaded)
    {
        std::string inputTransferCode = "";
        for (int inpI = 0; inpI < cNode->getNumInputs(); inpI++)
        {
            auto inputData = cNode->getInput(inpI);

            if (!defaultLoaded)
            {
                auto inputInfo =  inputData->getDataInfo();
                if (!(inputInfo->getIndex() <= 0)){
                    std::string dataName;
                    if (inputInfo->getDefaultName() == ""){
                        dataName = inputData->getName();
                    } else {
                        dataName = inputInfo->getDefaultName();
                    }
                    if (encounteredStrings.find(dataName) == encounteredStrings.end())
                    {
                        if (inputInfo->getFormat() == CSR_STYPE)
                        {
                            defaultLoaded = true;
                            int indexData = inputInfo->getDefaultIndex();
                            encounteredStrings.insert(dataName);
                            bool isColTile = hasDOpt(inputData, COL_TILE_DOPT);
                            inputTransferCode += graphTransferCode(indexData, dataName, isColTile, inputInfo->getDefaultDirected());
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
                    inputTransferCode += graphTransferCode(indexData, inputData->getName(), isColTile, inputInfo->getDirected());
                }
            }
        }
        preCode.addCode(inputTransferCode);
    }

    void transferGraphs(std::vector<CIRNode*>& program)
    {
        std::unordered_set<std::string> encounteredStrings;
        bool defaultLoaded = false;
        for (int i = 0; i < program.size(); i++)
        {
            CIRNode* outNode = program[i];
            auto oNode = dynamic_cast<ComputeNode*>(outNode);
            if (oNode)
            {
                generateTransferCodeForUniqueInput(oNode, encounteredStrings, defaultLoaded);
            } else
            {
                auto loopNode = dynamic_cast<TrainingLoopNode*>(outNode);
                for (int ix = 0; ix < loopNode->getLoopNodeNum(); ix++)
                {
                    CIRNode* inNode = loopNode->getNode(ix);
                    auto cNode = dynamic_cast<ComputeNode*>(inNode);
                    generateTransferCodeForUniqueInput(cNode, encounteredStrings, defaultLoaded);
                }
            }
        }
    }

    // Separate function so it can be extended in the architecture specific components
    virtual void initCMake()
    {
    };

    // Separate function so it can be extended in the architecture specific components
    virtual void initKernels(std::vector<CIRNode*>& program)
    {
    };

    virtual void dataPrep(std::vector<CIRNode*>& program)
    {
    };

    void commonPerCode()
    {
        // Will not change for now
        // TODO need to change the types based on the data
        std::string tempStdCommon;
        if (GALAFEContext::use_long)
        {
            tempStdCommon = "#include <algorithm>\n\
typedef int ind1_t;\n\
typedef int ind2_t;\n\
typedef long lab_t;\n\
typedef float val_t;\n\
typedef int mask_load_t;\n\
typedef bool mask_t;\n\
// Dense matrix with double values.\n\
typedef DenseMatrix<int64_t, int64_t, val_t> DM;\n\
typedef DenseMatrix<int64_t, int64_t, lab_t> DL;\n\
typedef DenseMatrix<int64_t, int64_t, mask_load_t> DBL;\n\
typedef DenseMatrix<int64_t, int64_t, mask_t> DB;\n\
typedef CSRCMatrix<ind1_t, ind2_t, val_t> SM;\n\
int global_nrows;\n\
int global_classes;\n\
int global_emb_size;\n\
int global_ra;\n\
int global_rb;\n\
std::vector<int> global_segments;\n\
bool global_is_directed;\n\
\n\
std::vector<torch::Tensor> global_offset_graph;\n\
std::vector<torch::Tensor> global_columns_graph;\n\
std::vector<torch::Tensor> global_value_graph;\n\
std::vector<torch::Tensor> global_bounds;\n\
std::vector<torch::Tensor> global_transpose_perm;\n";
        } else
        {
            tempStdCommon = "#include <algorithm>\n\
typedef int ind1_t;\n\
typedef int ind2_t;\n\
typedef long lab_t;\n\
typedef float val_t;\n\
typedef int mask_load_t;\n\
typedef bool mask_t;\n\
// Dense matrix with double values.\n\
typedef DenseMatrix<ind1_t, ind2_t, val_t> DM;\n\
typedef DenseMatrix<ind1_t, ind2_t, lab_t> DL;\n\
typedef DenseMatrix<ind1_t, ind2_t, mask_load_t> DBL;\n\
typedef DenseMatrix<ind1_t, ind2_t, mask_t> DB;\n\
typedef CSRCMatrix<ind1_t, ind2_t, val_t> SM;\n\
int global_nrows;\n\
int global_classes;\n\
int global_emb_size;\n\
int global_ra;\n\
int global_rb;\n\
std::vector<int> global_segments;\n\
bool global_is_directed;\n\
\n\
std::vector<torch::Tensor> global_offset_graph;\n\
std::vector<torch::Tensor> global_columns_graph;\n\
std::vector<torch::Tensor> global_value_graph;\n\
std::vector<torch::Tensor> global_bounds;\n\
std::vector<torch::Tensor> global_transpose_perm;\n";
        }

        importCode.addCode(tempStdCommon);

        std::string mainFuncCode  = "int main(int argc, char **argv) {\n\
  typedef typename SM::itype iT;\n\
  typedef typename SM::ntype nT;\n\
  typedef typename SM::vtype vT;\n\
\n\
  typedef typename DM::itype diT;\n\
  typedef typename DM::ntype dnT;\n\
  typedef typename DM::vtype dvT;\n\
  auto options_int_tile = \n\
    torch::TensorOptions().dtype(torch::kInt).requires_grad(false);\n\
  auto options_float_tile = \n\
    torch::TensorOptions().dtype(torch::kFloat).requires_grad(true);\n";
        preCode.addCode(mainFuncCode);
    }

    void writeCode(std::vector<CIRNode*> &program,
        std::vector<RelationEdge*>& dependencies,
        std::vector<RelationEdge*>& associations,
        std::vector<TransformEdge*>& transforms)
    {
        this->openStream(this->outputPath);
        // Kernel code - Architecture dependant
        // CMake (also has write for now?)
        initCMake();
        // Kernels
        // std::cout << "Works0" << std::endl;
        initKernels(program);
        // std::cout << "Works1" << std::endl;
        commonPerCode();

        generateCode(program, transforms);

        this->writeCode(cmakeCode, outStreamCMake);
        this->writeCode(importCode, outStreamModel);
        this->writeCode(kernelCode, outStreamModel);
        // std::cout << "Works2" << std::endl;
        this->writeCode(kernelCallCode, outStreamModel);
        this->writeCode(*model.getDef(), outStreamModel);
        this->writeCode(*model.getInitCall(), outStreamModel, ", ", true, true);
        this->writeCode(*model.getInit(), outStreamModel);
        // std::cout << "Works3" << std::endl;
        this->writeCode(*model.getForwardCallPre(), outStreamModel, "");
        this->writeCode(*model.getForwardCallInternal(), outStreamModel, "");
        this->writeCode(*model.getForwardCallPost(), outStreamModel);
        this->writeCode(*model.getForward(), outStreamModel);
        this->writeCode(preCode, outStreamModel);
        // std::cout << "Works4" << std::endl;
        this->writeCode(*model.getInv(), outStreamModel);
        this->writeCode(*model.getPreCall(), outStreamModel, "");
        this->writeCode(*model.getCall(), outStreamModel, "");
        this->writeCode(*model.getPostCall(), outStreamModel);
        // std::cout << "Works5" << std::endl;
        this->writeCode(postCode, outStreamModel);

        this->closeStream();
    }
};

#endif //GNN_ACCELERATION_LANGUAGE_COMMON_H
