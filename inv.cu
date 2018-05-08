#include "lib/index.h"

using std::string;
using std::map;

int main(int argc, const char *argv[]){
	cublasCreate(&device::cublas_handle);

	map<string, string> cfg;

	for (size_t i = 1; i < argc; i++) {
		string arg = argv[i];
		size_t pos = arg.find("=");
		if (pos != string::npos && arg[0] == '-') {
			string key = arg.substr(1, pos - 1);
			string value = arg.substr(pos + 1);
			cfg[key] = value;
		}
		else {
			cfg["config"] = arg;
		}
	}

	if (!cfg["config"].size()) {
		std::cout << "Using example/checker" << std::endl;
		cfg["config"] = "example/checker";
	}

	Config *config = new Config(cfg);
	switch (config->i["mode"]) {
		case 0: {
			Solver *solver = module::solver(config->i["solver"]);
	        Filter *filter = module::filter(config->i["filter"]);
			Misfit *misfit = module::misfit(config->i["misfit"]);
			Optimizer *optimizer = module::optimizer(config->i["optimizer"]);

			misfit->init(config, solver, filter);
			optimizer->init(config, solver, misfit);
			optimizer->run();
			break;
		}
		case 1: {
			Solver *solver = module::solver(config->i["solver"]);
			solver->init(config);
			solver->importModel(true);
			solver->exportAxis();
			solver->runForward(-1, false, true, true);
			break;
		}
		case 2: {
			Solver *solver = module::solver(config->i["solver"]);
			Filter *filter = module::filter(config->i["filter"]);
			Misfit *misfit = module::misfit(config->i["misfit"]);

			misfit->init(config, solver, filter);
			misfit->run(true);
			solver->exportAxis();
			solver->exportKernels();
			break;
		}
	}


	/* clock_t start = clock();
	solver->run(true);
	double duration = (clock() - start) / (double) CLOCKS_PER_SEC;
	cout << "Elapsed time: " << duration << endl; */

	cublasDestroy(device::cublas_handle);

	return 0;
}
