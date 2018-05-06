#pragma once

class NLCGOptimizer : public Optimizer {
protected:
    size_t nlcg_type;
    float nlcg_thresh;

    float fletcher(Dim &dim) {
        pcalc(p_new, 1, g_new, -1, g_old, dim);
        float num = pdot(g_new, p_new, dim);
        float den = pdot(g_old, g_old, dim);
        return num / den;
    };
    float pollak(Dim &dim) {
        float num = pdot(g_new, g_new, dim);
        float den = pdot(g_old, g_old, dim);
        return num / den;
    };
    float descent(Dim &dim) {
        return pdot(p_new, g_new, dim);
    };
    float conjugacy(Dim &dim) {
        return fabs(pdot(g_new, g_old, dim) / pdot(g_new, g_new, dim));
    };

public:
    void init(Config *config, Solver *solver, Misfit *misfit) {
        Optimizer::init(config, solver, misfit);
        nlcg_type = config->i["nlcg_type"];
        nlcg_thresh = config->f["nlcg_thresh"];
    };
    int computeDirection() {
        Dim dim(solver->nx, solver->nz);
        inv_count++;
        if (inv_count == 1) {
            pcalc(p_new, -1, g_new, dim);
            return 0;
        }
        else if(inv_cycle && inv_cycle < inv_count) {
            std::cout << "  restarting NLCG... [periodic restart]" << std::endl;
            return -1;
        }
        else {
            float beta = nlcg_type ? pollak(dim) : fletcher(dim);
            pcalc(p_new, -1, g_new, beta, p_old, dim);
            if(conjugacy(dim) > nlcg_thresh){
                std::cout << "  restarting NLCG... [loss of conjugacy]" << std::endl;
                return -1;
            }
            if(descent(dim) > 0){
                std::cout << "  restarting NLCG... [not a descent direction]" << std::endl;
                return -1;
            }
            return 1;
        }
    };
};
