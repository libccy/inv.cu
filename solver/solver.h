#pragma once

namespace module {
	Source *source(size_t);
}

class Solver {
protected:
	string path_output;
	string path_model_init;
	string path_model_true;

	size_t obs;
	size_t sfe;
	size_t wfe;

	size_t abs_width;
	float abs_param;
	bool abs_left;
	bool abs_top;
	bool abs_right;
	bool abs_bottom;

	size_t model_npt;
	float *model_x;
	float *model_z;
	float *model_vp;
	float *model_vs;
	float *model_rho;

	float *src_x;
	float *src_z;
	float *rec_x;
	float *rec_z;

public:
	bool sh;
	bool psv;
	bool inv_lambda;
	bool inv_mu;
	bool inv_rho;

	size_t nx;
	size_t nz;
	size_t nt;
	size_t nsrc;
	size_t nrec;

	float dx;
	float dz;
	float dt;

	float *stf_x;
	float *stf_y;
	float *stf_z;

	float *adstf_x;
	float *adstf_y;
	float *adstf_z;

	float *out_x;
	float *out_y;
	float *out_z;

	float *lambda;
	float *mu;
	float *rho;

	float *k_lambda;
	float *k_mu;
	float *k_rho;

	void exportTraces(size_t isrc, bool adjoint) {
		createDirectory(path_output);

		int header1[28];
		short int header2[2];
		short int header3[2];
		float header4[30];

		for (size_t i = 0; i < 28; i++) header1[i] = 0;
		for (size_t i = 0; i < 2; i++) header2[i] = 0;
		for (size_t i = 0; i < 2; i++) header3[i] = 0;
		for (size_t i = 0; i < 30; i++) header4[i] = 0;

		float xs = device::get(src_x, isrc);
		float zs = device::get(src_z, isrc);

		float *host_rec_x = host::create(nrec, rec_x);
		float *host_rec_z = host::create(nrec, rec_z);

		short int dt_int2;
	    if(dt * 1e6 > pow(2, 15)){
	        dt_int2 = 0;
	    }
	    else{
	        dt_int2 = dt * 1e6;
	    }

		header1[18] = std::round(xs);
	    header1[19] = std::round(zs);

	    header2[0] = 0;
	    header2[1] = nt;

	    header3[0] = dt_int2;
	    header3[1] = 0;

		auto filename = [&](string comp) {
			string istr = std::to_string(isrc);
			for (size_t i = istr.size(); i < 6; i++) {
				istr = "0" + istr;
			}
			return path_output + ((!adjoint&&obs==0)?"/v":"/u") + comp + "_" + istr + ".su";
		};
		auto write = [&](string comp, float *data) {
			std::ofstream outfile(filename(comp), std::ofstream::binary);
			for (size_t ir = 0; ir < nrec; ir++) {
				header1[0] = ir + 1;
				header1[9] = std::round(host_rec_x[ir] - xs);
				header1[20] = std::round(host_rec_x[ir]);
				header1[21] = std::round(host_rec_z[ir]);
				if(nrec > 1) header4[1] = host_rec_x[1] - host_rec_x[0];

				outfile.write(reinterpret_cast<char*>(header1), 28 * sizeof(int));
		        outfile.write(reinterpret_cast<char*>(header2), 2 * sizeof(short int));
		        outfile.write(reinterpret_cast<char*>(header3), 2 * sizeof(short int));
		        outfile.write(reinterpret_cast<char*>(header4), 30 * sizeof(float));
		        outfile.write(reinterpret_cast<char*>(data + ir * nt), nt * sizeof(float));
			}
			outfile.close();
			free(data);
		};

		if (sh) {
			write("y", host::create(nt * nrec, out_y));
		}
		if (psv) {
			write("x", host::create(nt * nrec, out_x));
			write("z", host::create(nt * nrec, out_z));
		}

		free(host_rec_x);
		free(host_rec_z);
	};
	void exportData(string comp, float *data, size_t n = 0) {
		string istr = std::to_string(n);
		for (size_t i = istr.size(); i < 6; i++) {
			istr = "0" + istr;
		}
		int npt = nx * nz;
		std::ofstream outfile(path_output + "/proc" + istr + "_" + comp + ".bin", std::ofstream::binary);
		outfile.write(reinterpret_cast<char*>(&npt), sizeof(int));
		outfile.write(reinterpret_cast<char*>(data), npt * sizeof(float));
		outfile.close();
		free(data);
	};
	void exportKernels(size_t n = 0) {
		size_t len = nx * nz;
		if(inv_lambda) exportData("k_lambda", host::create(len, k_lambda), n);
		if(inv_mu) exportData("k_mu", host::create(len, k_mu), n);
		if(inv_rho) exportData("k_rho", host::create(len, k_rho), n);
	};
	void exportModels(size_t n = 0) {
		size_t len = nx * nz;
		if(inv_lambda) exportData("lambda", host::create(len, lambda), n);
		if(inv_mu) exportData("mu", host::create(len, mu), n);
		if(inv_rho) exportData("rho", host::create(len, rho), n);
	};
	virtual void init(Config *config) {
		dt = config->f["dt"];
		obs = config->i["obs"];
		nt = config->i["nt"];
		sfe = config->i["sfe"];
		wfe = config->i["wfe"];
		sh = (bool) config->i["sh"];
		psv = (bool) config->i["psv"];

		abs_param = config->f["abs_param"];
		abs_width = config->i["abs_width"];
		abs_left = (bool) config->i["abs_left"];
		abs_top = (bool) config->i["abs_top"];
		abs_right = (bool) config->i["abs_right"];
		abs_bottom = (bool) config->i["abs_bottom"];

		path_output = config->s["output"];
		path_model_init = config->s["model_init"];
		path_model_true = config->s["model_true"];

		nsrc = config->src.size();
		nrec = config->rec.size();

		float *host_src_x = host::create(nsrc);
		float *host_src_z = host::create(nsrc);
		float *host_rec_x = host::create(nrec);
		float *host_rec_z = host::create(nrec);

		float *host_stf_x = host::create(nsrc * nt);
		float *host_stf_y = host::create(nsrc * nt);
		float *host_stf_z = host::create(nsrc * nt);

		for (size_t is = 0; is < nsrc; is++) {
			host_src_x[is] = config->src[is][0];
			host_src_z[is] = config->src[is][1];
			Source *src = module::source(config->src[is][2]);
			src->init(config->src[is] + 3, nt, dt);
			memcpy(host_stf_x + is * nt, src->stf_x, nt * sizeof(float));
			memcpy(host_stf_y + is * nt, src->stf_y, nt * sizeof(float));
			memcpy(host_stf_z + is * nt, src->stf_z, nt * sizeof(float));
			delete src;
		}

		for (size_t ir = 0; ir < nrec; ir++) {
			host_rec_x[ir] = config->rec[ir][0];
			host_rec_z[ir] = config->rec[ir][1];
		}

		src_x = device::create(nsrc, host_src_x);
		src_z = device::create(nsrc, host_src_z);
		rec_x = device::create(nrec, host_rec_x);
		rec_z = device::create(nrec, host_rec_z);

		stf_x = device::create(nsrc * nt, host_stf_x);
		stf_y = device::create(nsrc * nt, host_stf_y);
		stf_z = device::create(nsrc * nt, host_stf_z);

		if (config->i["mode"] != 1) {
	        inv_lambda = (bool) config->i["inv_lambda"];
	        inv_mu = (bool) config->i["inv_mu"];
	        inv_rho = (bool) config->i["inv_rho"];

	        adstf_x = device::create(nrec * nt);
	        adstf_y = device::create(nrec * nt);
	        adstf_z = device::create(nrec * nt);
		}

		if (sh) {
			out_y = device::create(nrec * nt);
		}
		if (psv) {
			out_x = device::create(nrec * nt);
			out_z = device::create(nrec * nt);
		}

		free(host_src_x);
		free(host_src_z);
		free(host_rec_x);
		free(host_rec_z);

		free(host_stf_x);
		free(host_stf_y);
		free(host_stf_z);
	};
	virtual void importModel(bool model) {
		string &path = model ? path_model_true : path_model_init;
		int npt = 0;
		auto read = [&](string comp, float* &data) {
			std::ifstream infile(path + "/proc000000_" + comp + ".bin", std::ifstream::binary);
			infile.read(reinterpret_cast<char*>(&npt), sizeof(int));
			if (model_npt != npt) {
				std::cout << "error: invalid model" << std::endl;
			}

			float *buffer = host::create(model_npt);
			infile.read(reinterpret_cast<char*>(buffer), model_npt * sizeof(float));
			data = device::create(model_npt, buffer);
			free(buffer);
			infile.close();
		};
		read("x", model_x);
		read("z", model_z);
		read("vp", model_vp);
		read("vs", model_vs);
		read("rho", model_rho);
	};
	virtual void initKernels() = 0;
	virtual void exportAxis() = 0;
	virtual void runForward(int, bool = false, bool = false, bool = false) = 0;
	virtual void runAdjoint(int, bool = false) = 0;
	virtual ~Solver() {
		std::cout << "deleted" << std::endl;
	};
};
