#pragma once

class Optimizer {
protected:
    enum Parameter { lambda = 0, mu = 1, rho = 2 };
    Misfit *misfit;
    Solver *solver;

    bool inv_parameter[3];

    size_t inv_iteration;
    size_t inv_cycle;

    float ls_steplenmax;
    float ls_stepleninit;
    float ls_stepcountma;
    float ls_thresh;
    float unsharp_mask;

    float **m_new;
    float **m_old;
    float **g_new;
    float **g_old;
    float **p_new;
    float **p_old;

    float *ls_gtg;
    float *ls_gtp;

    size_t eval_count;
    size_t inv_count;
    size_t ls_count;

public:
    void run() {
        Dim dim(solver->nx, solver->nz);

        solver->exportAxis();
        solver->exportModels();

        eval_count = 0;
        inv_count = 0;
        ls_count = 0;

        for(int iter = 0; iter < inv_iteration; iter++){
            std:: cout << "Starting iteration " << iter + 1 << " / " << inv_iteration << std::endl;
            float f = misfit->calc(true);
            if (iter == 0) misfit->ref = f;
            eval_count += 2;

            std::cout << "  misfit = " << f / misfit->ref << std::endl;
            if (computeDirection() < 0) {
                restartSearch();
            }
            lineSearch(f);

            for (size_t i = 0; i < 3; i++) {
                if (inv_parameter[i]) {
                    device::copy(m_old[i], m_new[i], dim);
                    device::copy(p_old[i], p_new[i], dim);
                    device::copy(g_old[i], g_new[i], dim);
                }
            }

            solver->exportKernels(iter + 1);
            solver->exportModels(iter + 1);
        }
    };
    virtual void restartSearch() = 0;
    virtual int lineSearch(float) = 0;
    virtual int computeDirection() = 0;
    virtual void init(Config *config, Solver *solver, Misfit *misfit) {
        this->misfit = misfit;
        this->solver = solver;

        inv_parameter[lambda] = (bool) solver->inv_lambda;
        inv_parameter[mu] = (bool) solver->inv_mu;
        inv_parameter[rho] = (bool) solver->inv_rho;

        inv_cycle = config->i["inv_cycle"];
        inv_iteration = config->i["inv_iteration"];

        ls_steplenmax = config->f["ls_steplenmax"];
        ls_stepleninit = config->f["ls_stepleninit"];
        ls_stepcountma = config->f["ls_stepcountma"];
        ls_thresh = config->f["ls_thresh"];
        unsharp_mask = config->f["unsharp_mask"];

        ls_gtg = host::create(inv_iteration);
        ls_gtp = host::create(inv_iteration);

        size_t len = solver->nx * solver->nz;

        m_new = host::create<float *>(3);
        g_new = host::create<float *>(3);
        m_old = device::create2D(3, len);
        g_old = device::create2D(3, len);
        p_new = device::create2D(3, len);
        p_old = device::create2D(3, len);

        m_new[lambda] = solver->lambda;
        m_new[mu] = solver->mu;
        m_new[rho] = solver->rho;
        g_new[lambda] = solver->k_lambda;
        g_new[mu] = solver->k_mu;
        g_new[rho] = solver->k_rho;
    };
};
