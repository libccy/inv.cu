#pragma once

class LBFGSOptimizer : public Optimizer {
protected:
    size_t lbfgs_mem;

public:
    void init(Config *config, Solver *solver, Misfit *misfit) {
        Optimizer::init(config, solver, misfit);
        lbfgs_mem = config->i["lbfgs_mem"];
    };
    void restartSearch() {
        // device::calc(p_new, -1, g_new, Dim(solver->nx, solver->nz));
    };
    int computeDirection(){
        return 0;
    };
    int lineSearch(float f) {
        return 0;
    };
};
