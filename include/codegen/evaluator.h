#ifndef TESTER_H_
#define TESTER_H_

#include "../../src/utils/threading_utils.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <tests/common.h>
#include <torch/torch.h>
#include <vector>

template <typename M> class Evaluator {
private:
  uint skip;
  float best_val_acc = -1.0f;
  float best_test_acc = 0.0f;
  const char *val_label = "Best validation accuracy: ";
  const char *test_label = "Test accuracy at best val: ";
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
  // Switch reporting labels to vertex-nomination (AUC) terminology.
  void use_auc_metric() {
    val_label = "Best validation AUC: ";
    test_label = "Eval AUC at best val: ";
  }
  float best_val() const { return best_val_acc; }
  float best_test() const { return best_test_acc; }

  // Rank-based ROC-AUC (Mann-Whitney U with average ranks for ties) of score
  // against binary labels, restricted to mask. All tensors must be on CPU.
  static float mask_auc(const torch::Tensor &score, const torch::Tensor &label,
                        const torch::Tensor &mask) {
    auto s = score.index({mask}).contiguous();
    auto l = label.index({mask}).contiguous();
    int64_t n = s.numel();
    int64_t npos = l.sum().item().toLong();
    int64_t nneg = n - npos;
    if (npos == 0 || nneg == 0)
      return std::nanf("");
    auto order = s.argsort();
    auto s_sorted = s.index({order}).contiguous();
    auto l_sorted = l.index({order}).contiguous();
    float *sv = s_sorted.data_ptr<float>();
    long *lv = l_sorted.data_ptr<long>();
    double pos_rank_sum = 0;
    int64_t i = 0;
    while (i < n) {
      int64_t j = i;
      while (j + 1 < n && sv[j + 1] == sv[i])
        j++;
      double avg_rank = 0.5 * ((double)(i + 1) + (double)(j + 1)); // 1-based
      for (int64_t k = i; k <= j; k++)
        if (lv[k])
          pos_rank_sum += avg_rank;
      i = j + 1;
    }
    return (float)((pos_rank_sum - 0.5 * (double)npos * (double)(npos + 1)) /
                   ((double)npos * (double)nneg));
  }

  // Per-epoch AUC over the train/val/eval pools from a per-node score tensor.
  void test_auc(const torch::Tensor &score_dev, const torch::Tensor &label_dev,
                const torch::Tensor &train_dev, const torch::Tensor &val_dev,
                const torch::Tensor &test_dev, float &train_auc, float &val_auc,
                float &test_auc) {
    auto score = score_dev.detach().to(torch::kCPU);
    auto label = label_dev.to(torch::kCPU);
    train_auc = mask_auc(score, label, train_dev.to(torch::kCPU));
    val_auc = mask_auc(score, label, val_dev.to(torch::kCPU));
    test_auc = mask_auc(score, label, test_dev.to(torch::kCPU));
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
    std::cout << val_label << best_val_acc << std::endl;
    std::cout << test_label << best_test_acc << std::endl;

    std::cout << std::endl;
    std::cout << "model_init " << begin_mem << std::endl;
    std::cout << "training " << end_mem << std::endl;
  }
};

#endif // TESTER_H_
