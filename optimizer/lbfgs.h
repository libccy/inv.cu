#pragma once

class LBFGSOptimizer : public Optimizer {
protected:
    size_t lbfgs_mem;
    float lbfgs_thresh;

public:
    void init(Config *config, Solver *solver, Misfit *misfit) {
        Optimizer::init(config, solver, misfit);
        lbfgs_mem = config->i["lbfgs_mem"];
        lbfgs_thresh = config->f["lbfgs_thresh"];
    };
    int computeDirection(){
        return 0;
    };
};
