#ifndef TESTER_H_
#define TESTER_H_

#include <vector>
#include <iostream>
#include <torch/torch.h>

template<typename M>
class Evaluator {
private:
    int skip;
    float best_val_acc;
    float best_test_acc;
    std::vector<double> forward_timing;
    std::vector<double> train_timing;
public:
    Evaluator(int skip_cache_warmup) : skip(skip_cache_warmup) {}

    void begin() {}
    void end() {}
    void begin_forward() {}
    void end_forward() {}
    void begin_train() {}
    void end_train() {}

    template<typename... Args>
    void test(M *m,
              std::vector<torch::Tensor> (M::*forward)(Args..., int, int),
              Args... args,
              int ep, int mod,
              torch::Tensor &label,
              torch::Tensor &train, torch::Tensor &test, torch::Tensor &validate)
    {
        auto out = (*m.*forward)(args..., ep, mod)[0];
        auto pred = out.argmax(1);

        auto train_correct = pred.index({train}).eq(label.index({train}));
        auto test_correct = pred.index({test}).eq(label.index({test}));
        auto validate_correct = pred.index({validate}).eq(label.index({validate}));

        auto train_acc = train_correct.sum() / train.sum();
        auto test_acc = (test_correct.sum() / test.sum()).item().toFloat();
        auto val_acc = (validate_correct.sum() / validate.sum()).item().toFloat();

        if (best_val_acc < val_acc) {
            best_val_acc = val_acc;
            best_test_acc = test_acc;
        }
    }

        void report()
        {
            std::cout << "Best validation accuracy: " << best_val_acc << std::endl;
            std::cout << "Test accuracy at best validation: " << best_test_acc << std::endl;
        }
};



#endif // TESTER_H_
