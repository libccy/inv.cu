#pragma once

class LBFGSOptimizer : public Optimizer {
protected:
    size_t lbfgs_mem;

public:
    void init(Config *config, Solver *solver, Misfit *misfit) {
        Optimizer::init(config, solver, misfit);
        lbfgs_mem = config->i["lbfgs_mem"];
    };
    void computeDirection(){};
};
