#pragma once

class Misfit {
protected:
	Solver *solver;
	Filter *filter;
	float **obs_x;
	float **obs_y;
	float **obs_z;

public:
	float calc(bool kernel = false) {
		float misfit = 0;
		size_t &nsrc = solver->nsrc, &nrec = solver->nrec, &nt = solver->nt;
		Dim dim(nrec, nt);
		if (kernel) {
			solver->initKernels();
		}
		if(!solver->sh){
	        device::init(solver->adstf_y, 0, dim);
	    }
	    if(!solver->psv){
	        device::init(solver->adstf_x, 0, dim);
	        device::init(solver->adstf_z, 0, dim);
	    }
		for (size_t isrc = 0; isrc < nsrc; isrc++) {
			solver->runForward(isrc, true);
			for (size_t irec = 0; irec < nrec; irec++) {
				size_t irt = irec * nt;
				if (solver->sh) {
					misfit += calc(solver->out_y + irt, obs_y[isrc] + irt, solver->adstf_y + irt);
				}
				if (solver->psv) {
					misfit += calc(solver->out_x + irt, obs_x[isrc] + irt, solver->adstf_x + irt);
					misfit += calc(solver->out_z + irt, obs_z[isrc] + irt, solver->adstf_z + irt);
				}
			}
			if (kernel) {
				solver->runAdjoint(isrc);
			}
		}
		if (kernel && filter) {
			filter->apply(solver->k_lambda);
			filter->apply(solver->k_mu);
			filter->apply(solver->k_rho);
		}
		return misfit;
	};
	virtual void init(Config *config, Solver *solver, Filter *filter = nullptr) {
		this->solver = solver;
		this->filter = filter;

		solver->init(config);
		solver->importModel(true);
		filter->init(solver->nx, solver->nz, config->i["filter_param"]);

		size_t &nsrc = solver->nsrc, &nrec = solver->nrec, &nt = solver->nt;
		Dim dim(nt, nrec);
		obs_x = device::create2D(nsrc, dim);
		obs_y = device::create2D(nsrc, dim);
		obs_z = device::create2D(nsrc, dim);

		for (size_t isrc = 0; isrc < nsrc; isrc++) {
			solver->runForward(isrc, true);
			if (solver->sh) {
				device::copy(obs_y[isrc], solver->out_y, dim);
			}
			if (solver->psv) {
				device::copy(obs_x[isrc], solver->out_x, dim);
				device::copy(obs_z[isrc], solver->out_z, dim);
			}
		}
        solver->importModel(false);
	};
	virtual float calc(float *, float *, float *) = 0;
	virtual ~Misfit() {};
};
