#pragma once

class Optimizer {
protected:
    enum Parameter { lambda = 0, mu = 1, rho = 2 };
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

    float **m_new;
    float **m_old;
    float **g_new;
    float **g_old;
    float **p_new;
    float **p_old;

public:
    size_t neval = 0;
    void run() {
        Dim dim(solver->nx, solver->nz);

        solver->exportAxis();
        solver->exportModels();

        for(int iter = 0; iter < inv_iteration; iter++){
            std:: cout << "Starting iteration " << iter + 1 << " / " << inv_iteration << std::endl;
            float f = misfit->calc(true);
            if (iter == 0) misfit->ref = f;
            neval += 2;

            std::cout << "misfit: " << f << std::endl;
            // if (computeDirection() < 0) {
            //     restartSearch();
            // }
            // lineSearch(f);
            if (inv_lambda) {
                device::copy(p_old[lambda], p_new[lambda], dim);
                device::copy(g_old[lambda], g_new[lambda], dim);
            }
            if (inv_mu) {
                device::copy(p_old[mu], p_new[mu], dim);
                device::copy(g_old[mu], g_new[mu], dim);
            }
            if (inv_mu) {
                device::copy(p_old[rho], p_new[rho], dim);
                device::copy(g_old[rho], g_new[rho], dim);
            }

            solver->exportKernels(iter + 1);
            solver->exportModels(iter + 1);
        }
    };
    virtual int computeDirection() = 0;
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
