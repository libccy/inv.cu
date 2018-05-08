#pragma once

class Optimizer {
protected:
    Misfit *misfit;
    Solver *solver;

    float **m_new;
    float **m_old;
    float **g_new;
    float **g_old;
    float **p_new;
    float **p_old;

    float *ls_lens;
    float *ls_vals;
    float *ls_gtg;
    float *ls_gtp;

    bool inv_parameter[3];
    size_t inv_iteration_cycle;
    size_t inv_iteration;
    float inv_sharpen;

    size_t ls_step;
    float ls_step_max;
    float ls_step_init;
    float ls_thresh;

    size_t eval_count;
    size_t inv_count;
    size_t ls_count;

    float p_amax(float **a) {
        float amax_a = 0;
        for (size_t ip = 0; ip < 3; ip++) {
            if (inv_parameter[ip]) {
                amax_a = std::max(amax_a, device::amax(a[ip], solver->dim));
            }
        }
        return amax_a;
    };
    float p_dot(float **a, float **b) {
        float dot_ab = 0;
        for (size_t ip = 0; ip < 3; ip++) {
            if (inv_parameter[ip]) {
                dot_ab += device::dot(a[ip], b[ip], solver->dim);
            }
        }
        return dot_ab;
    };
    void p_calc(float **c, float ka, float **a) {
        for (size_t ip = 0; ip < 3; ip++) {
            if (inv_parameter[ip]) {
                device::calc(c[ip], ka, a[ip], solver->dim);
            }
        }
    };
    void p_calc(float **c, float ka, float **a, float kb, float **b) {
        for (size_t ip = 0; ip < 3; ip++) {
            if (inv_parameter[ip]) {
                device::calc(c[ip], ka, a[ip], kb, b[ip], solver->dim);
            }
        }
    };
    void p_copy(float **data, float **source) {
        for (size_t ip = 0; ip < 3; ip++) {
            if (inv_parameter[ip]) {
                device::copy(data[ip], source[ip], solver->dim);
            }
        }
    };
    float p_angle(float **p, float **g, float k){
        float xx = p_dot(p, p);
        float yy = p_dot(g, g);
        float xy = k * p_dot(p, g);
        return acos(xy / sqrt(xx * yy));
    };

    float bracket(size_t step_count, float step_max, int &status) {
        status = 1;
        return step_max;
    };
    float backtrack(size_t step_count, float step_max, int &status) {
        status = 1;
        return step_max;
    };

    virtual float calcStep(size_t, float, int &) = 0;
    virtual int lineSearch(float f) {
        int status = 0;
        float alpha = 0;

        float norm_m = p_amax(m_new);
        float norm_p = p_amax(p_new);
        float gtg = p_dot(g_new, g_new);
        float gtp = p_dot(g_new, p_new);

        float step_max = ls_step_max * norm_m / norm_p;
        size_t step_count = 0;
        ls_lens[ls_count] = 0;
        ls_vals[ls_count] = f;
        ls_gtg[inv_count - 1] = gtg;
        ls_gtp[inv_count - 1] = gtp;
        ls_count++;

        float alpha_old = 0;

        if(ls_step_init && ls_count <= 1){
            alpha = ls_step_init * norm_m / norm_p;
        }
        else{
            alpha = calcStep(step_count, step_max, status);
        }

        while(true){
            p_calc(m_new, 1, m_new, alpha - alpha_old, p_new);
            alpha_old = alpha;
            ls_lens[ls_count] = alpha;
            ls_vals[ls_count] = misfit->run(false);
            ls_count++;
            eval_count++;
            step_count++;

            alpha = calcStep(step_count, step_max, status);
            std::cout << "  step " << (step_count < 10 ? "0" : "")
                << step_count << "  misfit = "
                << ls_vals[ls_count - 1] / misfit->ref
                << std::endl;

            if(status > 0){
                std::cout << "  alpha = " << alpha << std::endl;
                p_calc(m_new, 1, m_new, alpha - alpha_old, p_new);
                return status;
            }
            else if(status < 0){
                p_calc(m_new, 1, m_new, -alpha_old, p_new);
                if(p_angle(p_new, g_new, -1) < 1e-3){
                    std::cout << "  line search failed" << std::endl;
                    return status;
                }
                else{
                    printf("  restarting line search...\n");
                    restartSearch();
                    return lineSearch(f);
                }
            }
        }
    };
    virtual int computeDirection() = 0;
    virtual void restartSearch() {
        p_calc(p_new, -1, g_new);
        ls_count = 0;
        inv_count = 1;
    };

public:
    void run() {
        solver->exportAxis();
        solver->exportModels();

        for(int iter = 0; iter < inv_iteration; iter++){
            std:: cout << "Starting iteration " << iter + 1 << " / " << inv_iteration << std::endl;
            float f = misfit->run(true);
            if (iter == 0) misfit->ref = f;
            eval_count += 2;

            std::cout << "  step 00  misfit = " << f / misfit->ref << std::endl;
            if (computeDirection() < 0) {
                restartSearch();
            }
            if (lineSearch(f) < 0) {
                break;
            }

            p_copy(m_old, m_new);
            p_copy(p_old, p_new);
            p_copy(g_old, g_new);

            solver->exportKernels(iter + 1);
            solver->exportModels(iter + 1);
        }
    };
    virtual void init(Config *config, Solver *solver, Misfit *misfit) {
        enum Parameter { lambda = 0, mu = 1, rho = 2 };
        this->misfit = misfit;
        this->solver = solver;

        inv_parameter[lambda] = solver->inv_lambda;
        inv_parameter[mu] = solver->inv_mu;
        inv_parameter[rho] = solver->inv_rho;
        inv_iteration = config->i["inv_iteration"];
        inv_iteration_cycle = config->i["inv_iteration_cycle"];
        inv_sharpen = config->f["inv_sharpen"];

        ls_step = config->i["ls_step"];
        ls_step_max = config->f["ls_step_max"];
        ls_step_init = config->f["ls_step_init"];
        ls_thresh = config->f["ls_thresh"];

        int nstep = inv_iteration * ls_step;
        ls_vals = host::create(nstep);
        ls_lens = host::create(nstep);
        ls_gtg = host::create(inv_iteration);
        ls_gtp = host::create(inv_iteration);

        size_t len = solver->dim;
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

        eval_count = 0;
        inv_count = 0;
        ls_count = 0;
    };
};
