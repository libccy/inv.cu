#pragma once

class LBFGSOptimizer : public Optimizer {
protected:
    size_t lbfgs_mem;
    size_t lbfgs_used;
    float lbfgs_thresh;

    float calcStep(size_t step_count, float step_max, int &status) {
        return backtrack(step_count, step_max, status);
    };
    void restartSearch() {
        Optimizer::restartSearch();
        lbfgs_used = 0;
    };
    int computeDirection() {
        return 0;
    };

public:
    void init(Config *config, Solver *solver, Misfit *misfit) {
        Optimizer::init(config, solver, misfit);
        lbfgs_mem = config->i["lbfgs_mem"];
        lbfgs_thresh = config->f["lbfgs_thresh"];
        lbfgs_used = 0;
    };
};
