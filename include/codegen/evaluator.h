#ifndef TESTER_H_
#define TESTER_H_

#include "../../src/utils/threading_utils.h"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <tests/common.h>
#include <torch/torch.h>
#include <vector>

template <typename M> class Evaluator {
private:
  uint skip;
  float best_val_acc;
  float best_test_acc;
  int begin_mem, end_mem;
  double begin_time, end_time;
  std::vector<double> forward_timing_starts;
  std::vector<double> forward_timing_ends;
  std::vector<double> train_timing_starts;
  std::vector<double> train_timing_ends;

int gpuMemoryUsage() {
  size_t freeMem, totalMem;
  cudaMemGetInfo(&freeMem, &totalMem);
  return (int)((totalMem - freeMem) / (1024 * 1024));
}

public:
        Evaluator(int skip_cache_warmup) : skip(skip_cache_warmup) {}

  // Called after model is initialized
  void begin() { 
    begin_time = get_time(); 
    begin_mem = gpuMemoryUsage();
  }

  void end() { 
    end_time = get_time(); 
    end_mem = gpuMemoryUsage();
  }
  void begin_forward() { forward_timing_starts.push_back(get_time()); }
  void end_forward() { forward_timing_ends.push_back(get_time()); }
  void begin_train() { train_timing_starts.push_back(get_time()); }
  void end_train() { train_timing_ends.push_back(get_time()); }

  template <typename... Args>
  void test(M *m, std::vector<torch::Tensor> (M::*forward)(Args..., int, int),
            Args... args, int ep, int mod, torch::Tensor &label,
            torch::Tensor &train, torch::Tensor &test, torch::Tensor &validate,
            float &train_acc, float &test_acc, float &val_acc) {
    auto out = (*m.*forward)(args..., ep, mod)[0];
    auto pred = out.argmax(1);

    auto train_correct = pred.index({train}).eq(label.index({train}));
    auto test_correct = pred.index({test}).eq(label.index({test}));
    auto validate_correct = pred.index({validate}).eq(label.index({validate}));

    train_acc = (train_correct.sum() / train.sum()).item().toFloat();
    test_acc = (test_correct.sum() / test.sum()).item().toFloat();
    val_acc = (validate_correct.sum() / validate.sum()).item().toFloat();
  }
  void train_step_report(int epoch, int mod, torch::Tensor d_loss,
                         float &train_acc, float &test_acc, float &val_acc) {

    if (best_val_acc < val_acc) {
      best_val_acc = val_acc;
      best_test_acc = test_acc;
    }

    if (epoch % mod == 0) {
      //Epoch 0180 | loss 0.1072 | train 1.0000 | val 0.5460 | test 0.5950
      std::cout << "Epoch " << epoch << " | ";
      std::cout << "loss " << d_loss.item().toFloat() << " | ";
      std::cout << "train " << train_acc << " | ";
      std::cout << "val " << val_acc << " | ";
      std::cout << "test " << test_acc << std::endl;
    }
  }
  void report() {
//    auto fs = forward_timing_starts.begin();
//    auto fe = forward_timing_ends.begin();
//    std::advance(fs, 5);
//    std::advance(fe, 5);
//    std::vector<double> out;
//    std::transform(fs, forward_timing_starts.end(), fe, std::back_inserter(out),
//                   [](const auto &s, const auto &e) { return e - s; });
 //   std::cout << "Forward mean timing: " << calc_mean(out) << std::endl;

// >>> Training finished in 0.75s
// Best validation accuracy : 0.7000
// Test accuracy at best val: 0.7020

// === Per‑stage resource usage (GPU memory shows PEAK) =============
// Stage             Time (s)  CPU %  Cores Used  GPU 0 %  GPU 0 Peak (MiB)
// ------------------------------------------------------------------------
// load_dataset      0.08      0.0    0.0         0.0      47              
// dataset_stats     0.02      100.0  32.0        0.0      47              
// undirected_edges  0.03      50.0   16.0        0.0      48              
// model_init        0.03      0.0    0.0         0.0      47              
// training          0.76      2.9    0.9         0.0      72              
// TOTAL             0.92      -      -           -        -               
    std::cout << std::endl 
              << ">>> Training finished in: " << (end_time - begin_time) << "s" << std::endl;
    std::cout << "Best validation accuracy: " << best_val_acc << std::endl;
    std::cout << "Test accuracy at best val: " << best_test_acc << std::endl;

    std::cout << std::endl;
    std::cout << "model_init " << begin_mem << std::endl;
    std::cout << "training " << end_mem << std::endl;
  }
};

#endif // TESTER_H_
