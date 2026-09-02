//
//
#include <tests/common.h>
#include <iostream>

#ifdef TMKL
typedef long long int ind1_t;
#else
typedef uint32_t ind1_t;
#endif

#ifdef TMKL
typedef long long int ind2_t;
#else
typedef uint64_t ind2_t;
#endif
typedef float val_t;
typedef int val_int_t;

// IR classes
#include "../src/ir/data.h"
#include "../src/ir/compute.h"
#include "../src/ir/frontend_metadata.h"
#include "../src/codegen/cuda.h"
#include "../src/codegen/common.h"

// Matrix classes
#include <formats/dense_matrix.h>
#include <formats/csrc_matrix.h>

// Frontend
#include "../src/frontend/context.h"

#include "../src/middle-end/middle-end.h"

#include <filesystem>
#include <boost/program_options.hpp>

namespace po = boost::program_options;

extern void generate_ir();

extern FILE* yyin;
extern int yyparse();
ModelConfig m1;

po::options_description option_description("GALA options");

std::vector<CIRNode*> GALAFEContext::program;
std::vector<RelationEdge*> GALAFEContext::dependencies;
std::vector<RelationEdge*> GALAFEContext::associations;
std::vector<TransformEdge*> GALAFEContext::transforms;

bool GALAFEContext::operator_reordering;
bool GALAFEContext::sparse_rewrites;
bool GALAFEContext::train_code_motion;
bool GALAFEContext::training_subgraph;
bool GALAFEContext::print_accuracy;
bool GALAFEContext::print_memory;
bool GALAFEContext::use_long;
filesystem::path GALAFEContext::data_root;

std::string GALAFEContext::opt_input = "";
int GALAFEContext::log_interval;
float GALAFEContext::dropout = 0.5f;
// Gradient clipping max-norm for supervised training; 0 disables (the
// PyG baselines do not clip). Default preserves 1.1-validated behavior.
float GALAFEContext::grad_clip = 1.0f;

//Dense matrix with double values.
typedef DenseMatrix<ind1_t, ind2_t, val_t> DMd_t;
//Dense matrix with integer values.
typedef DenseMatrix<ind1_t, ind2_t, val_int_t> DMi_t;
// Sparse matrix (graph)
typedef CSRCMatrix<ind1_t, ind2_t, val_t> SM_t;

#define bool_opt(n, d) (n, po::value<bool>()->default_value(d), n)

/** RULES -- res input is always the 1st input for a computaiton op */
int main(int argc, char **argv) {
    option_description.add_options()
        bool_opt("operator_reordering", true)
        bool_opt("sparse_rewrites", true)
        bool_opt("train_code_motion", true)
        bool_opt("training_subgraph", true)
        bool_opt("print_accuracy", false)
        bool_opt("print_memory", false)
        bool_opt("use_long", false)
        ("log_interval", po::value<int>()->default_value(10), "log_interval")
        ("opt-input", po::value<std::string>()->default_value(""), "optional input")
        ("data-root", po::value<std::string>(), "data root")
        ("script", po::value<std::string>(), "script")
        ("output", po::value<std::string>(), "output")
        ;


    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, option_description), vm);
    po::notify(vm);

    GALAFEContext::opt_input = vm["opt-input"].as<std::string>();
    GALAFEContext::log_interval = vm["log_interval"].as<int>();
    GALAFEContext::operator_reordering = vm["operator_reordering"].as<bool>();
    GALAFEContext::sparse_rewrites = vm["sparse_rewrites"].as<bool>();
    GALAFEContext::train_code_motion = vm["train_code_motion"].as<bool>();
    GALAFEContext::training_subgraph = vm["training_subgraph"].as<bool>();
    GALAFEContext::print_accuracy = vm["print_accuracy"].as<bool>();
    GALAFEContext::print_memory = vm["print_memory"].as<bool>();
    GALAFEContext::use_long = vm["use_long"].as<bool>();
    GALAFEContext::data_root = filesystem::canonical(filesystem::absolute(vm["data-root"].as<std::string>()));

    filesystem::path inputFile = filesystem::canonical(vm["script"].as<std::string>());
	filesystem::path outputPath = filesystem::canonical(vm["output"].as<std::string>());

	m1 = ModelConfig();

	FILE *myfile = fopen(inputFile.c_str(), "r");
	if (!myfile) {
		std::cout << "Invalid File" << std::endl;
		return -1;
	}

	double start, end;
	start = get_time();

	yyin = myfile;
	yyparse();
	fclose(myfile);

	if (GALAFEContext::opt_input != ""){
		typedef int ind1_t;
		typedef int ind2_t;
		typedef long lab_t;
		typedef float val_t;
		typedef int mask_load_t;
		typedef bool mask_t;

		// Dense matrix with double values.
		typedef DenseMatrix<ind1_t, ind2_t, val_t> DM;
		typedef DenseMatrix<ind1_t, ind2_t, lab_t> DL;
		typedef CSRCMatrix<ind1_t, ind2_t, val_t> SM;

		typedef typename SM::itype iT;
		typedef typename SM::ntype nT;

		std::string filename;
		SM adj;
		filename = GALAFEContext::opt_input;
		readSM_npy32<SM>(filename, &adj);

		// Adj info
		iT nrows = adj.nrows();
		iT ncols = adj.ncols();
		nT nvals = adj.nvals();

		// Init input with random numbers
		DM input_emb;
		readDM_npy<DM>(filename + "Feat.npy", &input_emb,
					   DenseMatrix<ind1_t, ind2_t, val_t>::DENSE_MTX_TYPE::RM);
		iT emb_size = input_emb.ncols();

		DL labels;
		readDM_npy<DL>(filename + "Lab.npy", &labels,
					   DenseMatrix<ind1_t, ind2_t, lab_t>::DENSE_MTX_TYPE::RM);
		int classes = *std::max_element(labels.vals_ptr(), labels.vals_ptr() + labels.nvals()) + 1;

		// Set to true in the current version as the datasets do not store graph values
		m1.graph_transformations[UNDIRECTED] = true;
		m1.graph_transformations[UNWEIGHTED] = true;
		m1.compute_transformations[COARSE] = 2;
		m1.graph_transformations[FEAT_SIZE] = emb_size;
		m1.graph_transformations[LABEL_SIZE] = classes;
		if (((float)nvals / ((long)nrows * nrows)) > 0.001) {

			m1.addDataTransformation(COL_TILE, nrows / 5);
		}
	}

	cout << " ---------------- printing model config ----------------------\n";
	cout << m1.to_string() << '\n';
	cout << "---------------------------------------------------------------\n";

	generate_ir();
	cout << " --------     checking generated ir output ------------ \n";
	cout << "PROGRAM (CIR Nodes): " << GALAFEContext::program.size() << '\n';

	for (int i = 0; i < GALAFEContext::program.size(); i++){
		cout << "        program node " << i << "\n";
	}
	auto p1 = dynamic_cast<ComputeNode*>(GALAFEContext::program[0]);
	std::cout << p1->getOutput(1)->getName() << " " << p1->getOutput(1)->getDataInfo()->getDirected() << std::endl;
	std::cout << p1->getOutput(1)->getName() << " " << p1->getOutput(1)->getDataInfo()->getWeighted() << std::endl;

	auto l1 = dynamic_cast<TrainingLoopNode*>(GALAFEContext::program[1]);
	auto o1 = l1->getNode(4);
	std::cout << "bb: " << o1->getOp() << " " << o1->getNumOpts() << std::endl;

	cout << "DEPENDENCIES " << GALAFEContext::dependencies.size() << '\n';
	for (int i = 0; i < GALAFEContext::dependencies.size(); i++){
		cout << "     dependency edge " << i << " with nodes " <<
			GALAFEContext::dependencies[i]->getNode1()->getName() <<
				", " << GALAFEContext::dependencies[i]->getNode2()->getName() << '\n';
	}
	std::cout << GALAFEContext::dependencies[1]->getNode1()->getName() << " " << GALAFEContext::dependencies[1]->getNode1()->getDataInfo()->getDirected() << std::endl;
	cout << "ASSOCIATIONS " << GALAFEContext::associations.size() << '\n';
	for (int i = 0; i < GALAFEContext::associations.size(); i++){
		cout << "     associations edge " << i << " with nodes " <<
			GALAFEContext::associations[i]->getNode1()->getName() <<
				", " << GALAFEContext::associations[i]->getNode2()->getName() << '\n';
	}
	std::cout << GALAFEContext::associations[0]->getNode1()->getName() << " " << GALAFEContext::associations[0]->getNode1()->getDataInfo()->getDirected() << std::endl;
	std::cout << GALAFEContext::associations[1]->getNode1()->getName() << " " << GALAFEContext::associations[1]->getNode1()->getDataInfo()->getDirected() << std::endl;
	cout << "TRANSFORMS " << GALAFEContext::transforms.size() << '\n';
	for (int i = 0; i < GALAFEContext::transforms.size(); i++){
		cout << "     transform edge " << i << " with nodes " <<
			GALAFEContext::transforms[i]->getNode1()->getName() <<
				", " << GALAFEContext::transforms[i]->getNode2()->getName() << '\n';
	}

	auto ctx = new GALAContext(GPU_DEVICE, SINGLE_NODE_SINGLE);
	std::string outputPathStr = outputPath.string();
	auto genCode = CUDAGenerator(ctx, outputPathStr, GALAFEContext::data_root.string());
	if (GALAFEContext::operator_reordering)
	{
		GALATransformations::complexityOperatorReordering(GALAFEContext::program, GALAFEContext::dependencies,
		GALAFEContext::associations, GALAFEContext::transforms);
	}
	if (GALAFEContext::sparse_rewrites)
	{
		GALATransformations::sparsityAwareRewrites(GALAFEContext::program, GALAFEContext::dependencies,
		GALAFEContext::associations, GALAFEContext::transforms);
	}
	if (GALAFEContext::train_code_motion)
	{
		GALATransformations::trainingInvariantCodeMotion(GALAFEContext::program, GALAFEContext::dependencies,
			GALAFEContext::associations, GALAFEContext::transforms);
	}
	// Only sound for train-mask-restricted objectives (see gala_driver.h).
	if (GALAFEContext::training_subgraph && m1.loss_fn == CROSS_ENTROPY)
	{
		GALATransformations::trainingSubGraph(GALAFEContext::program, GALAFEContext::dependencies,
			GALAFEContext::associations, GALAFEContext::transforms);
	}
	genCode.writeCode(GALAFEContext::program, GALAFEContext::dependencies,
		GALAFEContext::associations, GALAFEContext::transforms);

	end = get_time();
	std::cout << "Time taken for GALA compilation: " << (end - start)*1000  << std::endl;
}
