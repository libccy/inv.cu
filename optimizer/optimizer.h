#pragma once

class Optimizer {
protected:
    Misfit *misfit;
    Solver *solver;

    bool param[3];
    size_t inv_cycle;
    size_t inv_iteration;
    float unsharp_mask;

    float **m_new;
    float **m_old;
    float **g_new;
    float **g_old;
    float **p_new;
    float **p_old;

    float *ls_gtg;
    float *ls_gtp;

    size_t ls_count_max;
    float ls_len_init;
    float ls_len_max;
    float ls_thresh;

    size_t eval_count;
    size_t inv_count;
    size_t ls_count;

    float pdot(float **a, float **b, Dim &dim) {
        float dot_ab = 0;
        for (size_t ip = 0; ip < 3; ip++) {
            if (param[ip]) {
                dot_ab += device::dot(a[ip], b[ip], dim);
            }
        }
        return dot_ab;
    };
    void pcalc(float **c, float ka, float **a, Dim &dim) {
        for (size_t ip = 0; ip < 3; ip++) {
            if (param[ip]) {
                device::calc(c[ip], ka, a[ip], dim);
            }
        }
    };
    void pcalc(float **c, float ka, float **a, float kb, float **b, Dim &dim) {
        for (size_t ip = 0; ip < 3; ip++) {
            if (param[ip]) {
                device::calc(c[ip], ka, a[ip], kb, b[ip], dim);
            }
        }
    };
    void pcopy(float **data, float **source, Dim &dim) {
        for (size_t ip = 0; ip < 3; ip++) {
            if (param[ip]) {
                device::copy(data[ip], source[ip], dim);
            }
        }
    };

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

            pcopy(m_old, m_new, dim);
            pcopy(p_old, p_new, dim);
            pcopy(g_old, g_new, dim);

            solver->exportKernels(iter + 1);
            solver->exportModels(iter + 1);
        }
    };
    void restartSearch() {

    };
    void lineSearch(float f) {
        // from here: class lineSearch
    };
    virtual int computeDirection() = 0;
    virtual void init(Config *config, Solver *solver, Misfit *misfit) {
        enum Parameter { lambda = 0, mu = 1, rho = 2 };
        this->misfit = misfit;
        this->solver = solver;

        param[lambda] = solver->inv_lambda;
        param[mu] = solver->inv_mu;
        param[rho] = solver->inv_rho;
        inv_iteration = config->i["inv_iteration"];
        inv_cycle = config->i["inv_cycle"];
        unsharp_mask = config->f["unsharp_mask"];

        ls_count_max = config->i["ls_count_max"];
        ls_len_max = config->f["ls_len_max"];
        ls_len_init = config->f["ls_len_init"];
        ls_thresh = config->f["ls_thresh"];

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
