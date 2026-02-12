#ifndef TESTER_H_
#define TESTER_H_

#include "../../src/utils/threading_utils.h"
#include <iostream>
#include <torch/torch.h>
#include <vector>
#include <iterator>
#include <algorithm>
#include <tests/common.h>

template <typename M> class Evaluator {
private:
  uint skip;
  float best_val_acc;
  float best_test_acc;
  std::vector<double> forward_timing_starts;
  std::vector<double> forward_timing_ends;
  std::vector<double> train_timing_starts;
  std::vector<double> train_timing_ends;

public:
  Evaluator(int skip_cache_warmup) : skip(skip_cache_warmup) {}

  void begin() {}
  void end() {}
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
  void train_step_report(int epoch, int mod,
                         torch::Tensor d_loss, float &train_acc, float &test_acc, float &val_acc) {

    if (best_val_acc < val_acc) {
      best_val_acc = val_acc;
      best_test_acc = test_acc;
    }

    if (epoch % mod == 0) {
       std::cout << "Epoch: " << epoch << ", " << "Loss: " << d_loss.item().toFloat() << std::endl;
       std::cout << "Train: " << train_acc << ", " << "Val:" << val_acc << ", " << "Test: " << test_acc << std::endl;
    }

  }

  void report() {
    auto fs = forward_timing_starts.begin();
    auto fe = forward_timing_ends.begin();
    std::advance(fs, 5);
    std::advance(fe, 5);
    std::vector<double> out;
    std::transform(fs, forward_timing_starts.end(), fe, std::back_inserter(out),
                   [](const auto &s, const auto &e) { return e - s; });
    std::cout << "Best validation accuracy: " << best_val_acc << std::endl;
    std::cout << "Test accuracy at best validation: " << best_test_acc
              << std::endl;
    std::cout << "Forward mean timing: " << calc_mean(out) << std::endl;
  }
};

#endif // TESTER_H_
