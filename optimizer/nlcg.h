#pragma once

class NLCGOptimizer : public Optimizer {
protected:

public:
    void restartSearch() {
        // device::calc(p_new, -1, g_new, Dim(solver->nx, solver->nz));
    };
    int lineSearch(float f) {
        return 0;
    };
    int computeDirection() {
        // from here
        return 0;
    };
};
