#include "lib/index.h"

using std::string;
using std::map;

int main(int argc, const char *argv[]){
	cublasCreate(&device::cublas_handle);
	cusolverDnCreate(&device::solver_handle);

	map<string, string> cfg;

	for (size_t i = 0; i < argc; i++) {
		string arg = argv[i];
		size_t pos = arg.find("=");
		if (pos != string::npos && arg[0] == '-') {
			string key = arg.substr(1, pos - 1);
			string value = arg.substr(pos + 1);
			cfg[key] = value;
		}
		else {
			pos = arg.find(".cfg");
			if (pos != string::npos) {
				cfg["config"] = arg;
			}
		}
	}

	if (!cfg["config"].size()) {
		std::cout << "Using example/checker.cfg" << std::endl;
		cfg["config"] = "example/checker.cfg";
	}

	Config config(cfg);
	switch ((size_t)std::round(config["mode"])) {
		case 0: {
			// from here
			// Optimizer *optimizer = module::optimizer(config["optimizer"]);
		}
		case 1: {
			Solver *solver = module::solver(config["solver"]);
			solver->init(config);
			solver->importModel(true);
			solver->runForward(-1, false, true, true);
			break;
		}
		case 2: {
			Solver *solver = module::solver(config["solver"]);
			solver->init(config);
			solver->importModel(true);

			Filter *filter = module::filter(config["filter"]);
			filter->init(solver->nx, solver->nz, config["filter_param"]);

			Misfit *misfit = module::misfit(config["misfit"]);
			misfit->init(solver, filter);
			misfit->generateTraces();
			solver->importModel(false);
			misfit->calc(true);
			solver->exportKernels();
			break;
		}
	}


	/* clock_t start = clock();
	solver->run(true);
	double duration = (clock() - start) / (double) CLOCKS_PER_SEC;
	cout << "Elapsed time: " << duration << endl; */

	cublasDestroy(device::cublas_handle);
	cusolverDnDestroy(device::solver_handle);

	return 0;
}
