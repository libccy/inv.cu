#pragma once

class Optimizer {
protected:
    Misfit *misfit;
    Solver *solver;

    bool inv_lambda;
    bool inv_mu;
    bool inv_rho;

    size_t inv_maxiter;
    size_t inv_iteration;

    float ls_steplenmax;
    float ls_stepleninit;
    float ls_stepcountma;
    float ls_thresh;
    float unsharp_mask;

public:
    size_t neval = 0;
    void run() {
        Dim dim(solver->nx, solver->nz);
        solver->exportAxis();
        solver->exportModels();
        for(int iter = 0; iter < inv_iteration; iter++){
            std:: cout << "Starting iteration " << iter + 1 << " / " << inv_iteration << std::endl;
            // float f = misfit->calc(true);
            // if(iter == 0){
            //     misfit->ref = f;
            // }
            // neval += 2;
            //
            // int dir = computeDirection();
            // if(dir < 0){
            //     restartSearch();
            // }
            // lineSearch(f);
            //
            // device::copy(p_old, p_new, dim);
            // device::copy(g_old, g_new, dim);
            solver->exportKernels(iter + 1);
            solver->exportModels(iter + 1);
        }
    };
    virtual void computeDirection() = 0;
    virtual void init(Config *config, Solver *solver, Misfit *misfit) {
        inv_lambda = solver->inv_lambda;
        inv_mu = solver->inv_mu;
        inv_rho = solver->inv_rho;

        inv_maxiter = config->i["inv_maxiter"];
        inv_iteration = config->i["inv_iteration"];

        ls_steplenmax = config->f["ls_steplenmax"];
        ls_stepleninit = config->f["ls_stepleninit"];
        ls_stepcountma = config->f["ls_stepcountma"];
        ls_thresh = config->f["ls_thresh"];
        unsharp_mask = config->f["unsharp_mask"];

        this->misfit = misfit;
        this->solver = solver;
    };
};
