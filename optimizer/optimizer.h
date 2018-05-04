#pragma once

class Optimizer {
    Misfit *misfit;
    Solver *solver;

public:
    void run() {
        misfit->calc(true);
        solver->exportKernels();
    };
    // virtual void computeDirection() = 0;
    virtual void init(Config *config, Solver *solver, Filter *filter, Misfit *misfit) {
        misfit->init(config, solver, filter);
    };
};
