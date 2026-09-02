//

//

#ifndef CONTEXT_H
#define CONTEXT_H

#include "../ir/data.h"
#include "../ir/compute.h"
#include "../ir/frontend_metadata.h"

#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <filesystem>


class GALAFEContext {
    public:
        static std::vector<CIRNode*> program;
        static std::vector<RelationEdge*> dependencies;
        static std::vector<RelationEdge*> associations;
        static std::vector<TransformEdge*> transforms;

        static filesystem::path data_root;

		static bool operator_reordering;
		static bool sparse_rewrites;
		static bool training_subgraph;
		static bool train_code_motion;
		static bool print_memory;
		static bool print_accuracy;
		static bool use_long;
        static bool instrument_evluator;

		static std::string opt_input;
		static int log_interval;
		static float dropout;
		static float grad_clip;
		static std::string target; // code-generation target: "cuda" (default) or "cpu"
};

#endif //CONTEXT_H
