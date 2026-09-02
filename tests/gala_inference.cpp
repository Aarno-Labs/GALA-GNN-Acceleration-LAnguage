//
#include "gala_driver.h"

std::vector<CIRNode*> GALAFEContext::program;
std::vector<RelationEdge*> GALAFEContext::dependencies;
std::vector<RelationEdge*> GALAFEContext::associations;
std::vector<TransformEdge*> GALAFEContext::transforms;

bool GALAFEContext::operator_reordering = true;
bool GALAFEContext::sparse_rewrites = true;
bool GALAFEContext::train_code_motion = true;
bool GALAFEContext::training_subgraph = true;
bool GALAFEContext::print_accuracy = false;
bool GALAFEContext::print_memory = false;
bool GALAFEContext::use_long = false;

std::string GALAFEContext::opt_input = "";
filesystem::path GALAFEContext::data_root;
int GALAFEContext::log_interval = 10;
float GALAFEContext::dropout = 0.5f;
// Gradient clipping max-norm for supervised training; 0 disables (the
// PyG baselines do not clip). Default preserves 1.1-validated behavior.
float GALAFEContext::grad_clip = 1.0f;

ModelConfig m1;

int main(int argc, char **argv) {
	return gala_run(argc, argv, false, 5);
}
