#pragma once

class Misfit {
protected:
	Solver *solver;
	Filter *filter;
	float **obs_x;
	float **obs_y;
	float **obs_z;

public:
	float ref;
	void importTraces(size_t isrc, string &path) {
		size_t &nrec = solver->nrec, &nt = solver->nt;

	    int header1[28];
	    short int header2[2];
	    short int header3[2];
	    float header4[30];

		float *buffer = host::create(nt * nrec);
		auto filename = [&](string comp) {
			string istr = std::to_string(isrc);
			for (size_t i = istr.size(); i < 6; i++) {
				istr = "0" + istr;
			}
			return path + "/u" + comp + "_" + istr + ".su";
		};
		auto read = [&](string comp, float *data) {
			std::ifstream infile(filename(comp), std::ifstream::binary);
			for (size_t ir = 0; ir < nrec; ir++) {
				infile.read(reinterpret_cast<char*>(header1), 28 * sizeof(int));
	            infile.read(reinterpret_cast<char*>(header2), 2 * sizeof(short int));
	            infile.read(reinterpret_cast<char*>(header3), 2 * sizeof(short int));
	            infile.read(reinterpret_cast<char*>(header4), 30 * sizeof(float));
	            infile.read(reinterpret_cast<char*>(buffer + ir * nt), nt * sizeof(float));
			}
			infile.close();
			host::toDevice(data, buffer, nt * nrec);
		};

		if (solver->sh) {
			read("y", obs_y[isrc]);
		}
		if (solver->psv) {
			read("x", obs_x[isrc]);
			read("z", obs_z[isrc]);
		}

		free(buffer);
	};
	float run(bool kernel = false) {
		float misfit = 0;
		size_t &nsrc = solver->nsrc, &nrec = solver->nrec, &nt = solver->nt;
		Dim dim(nt, nrec);
		if (kernel) {
			solver->initKernels();
		}
		if(!solver->sh){
	        device::init(solver->adstf_y, 0.0f, dim);
	    }
	    if(!solver->psv){
	        device::init(solver->adstf_x, 0.0f, dim);
	        device::init(solver->adstf_z, 0.0f, dim);
	    }
		for (size_t isrc = 0; isrc < nsrc; isrc++) {
			solver->runForward(isrc, true);
			for (size_t irec = 0; irec < nrec; irec++) {
				size_t irt = irec * nt;
				if (solver->sh) {
					misfit += run(solver->out_y + irt, obs_y[isrc] + irt, solver->adstf_y + irt);
				}
				if (solver->psv) {
					misfit += run(solver->out_x + irt, obs_x[isrc] + irt, solver->adstf_x + irt);
					misfit += run(solver->out_z + irt, obs_z[isrc] + irt, solver->adstf_z + irt);
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
	virtual float run(float *, float *, float *) = 0;
	virtual void init(Config *config, Solver *solver, Filter *filter = nullptr) {
		this->solver = solver;
		this->filter = filter;

		solver->init(config);
		filter->init(solver->dim.nx, solver->dim.nz, config->i["filter_sigma"]);

		ref = 1;

		size_t &nsrc = solver->nsrc, &nrec = solver->nrec, &nt = solver->nt;
		Dim dim(nt, nrec);
		obs_x = device::create2D(nsrc, dim);
		obs_y = device::create2D(nsrc, dim);
		obs_z = device::create2D(nsrc, dim);

		if (config->i["trace_file"]) {
			std::cout << "Using " << config->s["trace"] << std::endl;
			for (size_t isrc = 0; isrc < nsrc; isrc++) {
				this->importTraces(isrc, config->s["trace"]);
			}
		}
		else {
			std::cout << "Generating traces" << std::endl;
			solver->importModel(true);
			for (size_t isrc = 0; isrc < nsrc; isrc++) {
				solver->runForward(isrc, true, true);
				if (solver->sh) {
					device::copy(obs_y[isrc], solver->out_y, dim);
				}
				if (solver->psv) {
					device::copy(obs_x[isrc], solver->out_x, dim);
					device::copy(obs_z[isrc], solver->out_z, dim);
				}
			}
		}

        solver->importModel(false);
	};
	virtual ~Misfit() {};
};
