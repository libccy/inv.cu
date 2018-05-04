#pragma once

namespace _FdmSolver {
	__global__ void mesh2grid(
			float *lambda, float *mu, float *rho,
			float *device_x, float *device_z,
			float *device_vp, float *device_vs, float *device_rho, size_t npt,
			float dmax, float dx, float dz, Dim dim) {
		size_t k = dim();
		float x = blockIdx.x * dx;
		float z = threadIdx.x * dz;
		float dmin = dmax;
		for (size_t k = 0; k < npt; k++) {
			float dx = x - device_x[k];
			float dz = z - device_z[k];
			float d = dx * dx + dz * dz;
			if (d < dmin) {
				dmin = d;
				lambda[k] = device_vp[k];
				mu[k] = device_vs[k];
				rho[k] = device_rho[k];
			}
		}
	}
	__global__ void vps2lm(float *lambda, float *mu, float *rho, Dim dim) {
		size_t k = dim();
		float &vp = lambda[k];
		float &vs = mu[k];
		if (vp > vs) {
			lambda[k] = rho[k] * (vp * vp - 2 * vs * vs);
		}
		else {
			lambda[k] = 0;
		}
		mu[k] = rho[k] * vs * vs;
	}
	__global__ void lm2vps(float *vp, float *vs, float *rho, Dim dim) {
		size_t k = dim();
		float &lambda = vp[k];
		float &mu = vs[k];
		vp[k] = sqrt((lambda + 2 * mu) / rho[k]);
		vs[k] = sqrt(mu / rho[k]);
	}
	__global__ void calcIdx(size_t *coord_n_id, float *coord_n, float Ln, float n){
	    size_t i = blockIdx.x;
	    coord_n_id[i] = (int)(coord_n[i] / Ln * (n - 1) + 0.5);
	}
	__global__ void initAbsbound(
		float *absbound, size_t abs_width, float abs_param,
		bool abs_left, bool abs_right, bool abs_bottom, bool abs_top, Dim dim) {
	    size_t i, j, k;
		dim(i, j, k);
	    absbound[k] = 1;

	    if (abs_left) {
	        if (i + 1 < abs_width) {
	            absbound[k] *= exp(-pow(abs_param * (abs_width - i - 1), 2));
	        }
	    }
	    if (abs_right) {
	        if (i > dim.nx - abs_width) {
	            absbound[k] *= exp(-pow(abs_param * (abs_width + i - dim.nx), 2));
	        }
	    }
	    if (abs_bottom) {
			if (j > dim.nz - abs_width) {
	            absbound[k] *= exp(-pow(abs_param * (abs_width + j - dim.nz), 2));
	        }
	    }
	    if (abs_top) {
			if (j + 1 < abs_width) {
			   absbound[k] *= exp(-pow(abs_param * (abs_width - j - 1), 2));
		   }
	    }
	}

	__global__ void divSY(float *dsy, float *sxy, float *szy, float dx, float dz, Dim dim){
	    size_t i, j, k;
		dim(i, j, k);
	    if (i >= 2 && i < dim.nx - 2) {
			dsy[k] = 9 * (sxy[k] - sxy[dim(-1,0)]) / (8 * dx) - (sxy[dim(1,0)] - sxy[dim(-2,0)]) / (24 * dx);
	    }
	    else{
	        dsy[k] = 0;
	    }
	    if (j >= 2 && j < dim.nz - 2) {
	        dsy[k] += 9 * (szy[k] - szy[dim(0,-1)]) / (8 * dz) - (szy[dim(0,1)] - szy[dim(0,-2)]) / (24 * dz);
	    }
	}
	__global__ void divSXZ(float *dsx, float *dsz, float *sxx, float *szz, float *sxz, float dx, float dz, Dim dim){
		size_t i, j, k;
		dim(i, j, k);
	    if (i >= 2 && i < dim.nx - 2) {
	        dsx[k] = 9 * (sxx[k] - sxx[dim(-1,0)])/(8 * dx) - (sxx[dim(1,0)] - sxx[dim(-2,0)]) / (24 * dx);
	        dsz[k] = 9 * (sxz[k] - sxz[dim(-1,0)])/(8 * dx) - (sxz[dim(1,0)] - sxz[dim(-2,0)]) / (24 * dx);
	    }
	    else{
	        dsx[k] = 0;
	        dsz[k] = 0;
	    }
	    if (j >= 2 && j < dim.nz - 2) {
	        dsx[k] += 9 * (sxz[k] - sxz[dim(0,-1)])/(8 * dz) - (sxz[dim(0,1)] - sxz[dim(0,-2)]) / (24 * dz);
	        dsz[k] += 9 * (szz[k] - szz[dim(0,-1)])/(8 * dz) - (szz[dim(0,1)] - szz[dim(0,-2)]) / (24 * dz);
	    }
	}
	__global__ void divVY(float *dvydx, float *dvydz, float *vy, float dx, float dz, Dim dim){
		size_t i, j, k;
		dim(i, j, k);
	    if (i >= 1 && i < dim.nx - 2) {
	        dvydx[k] = 9 * (vy[dim(1,0)] - vy[k]) / (8 * dx) - (vy[dim(2,0)] - vy[dim(-1,0)]) / (24 * dx);
	    }
	    else{
	        dvydx[k] = 0;
	    }
	    if (j >= 1 && j < dim.nz - 2) {
	        dvydz[k] = 9 * (vy[dim(0,1)] - vy[k]) / (8 * dz) - (vy[dim(0,2)] - vy[dim(0,-1)]) / (24 * dz);
	    }
	    else{
	        dvydz[k] = 0;
	    }
	}
	__global__ void divVXZ(float *dvxdx, float *dvxdz, float *dvzdx, float *dvzdz, float *vx, float *vz, float dx, float dz, Dim dim){
		size_t i, j, k;
		dim(i, j, k);
	    if (i >= 1 && i < dim.nx - 2) {
	        dvxdx[k] = 9 * (vx[dim(1,0)] - vx[k]) / (8 * dx) - (vx[dim(2,0)] - vx[dim(-1,0)]) / (24 * dx);
	        dvzdx[k] = 9 * (vz[dim(1,0)] - vz[k]) / (8 * dx) - (vz[dim(2,0)] - vz[dim(-1,0)]) / (24 * dx);
	    }
	    else{
	        dvxdx[k] = 0;
	        dvzdx[k] = 0;
	    }
	    if (j >= 1 && j < dim.nz - 2) {
	        dvxdz[k] = 9 * (vx[dim(0,1)] - vx[k]) / (8 * dz) - (vx[dim(0,2)] - vx[dim(0,-1)]) / (24 * dz);
	        dvzdz[k] = 9 * (vz[dim(0,1)] - vz[k]) / (8 * dz) - (vz[dim(0,2)] - vz[dim(0,-1)]) / (24 * dz);
	    }
	    else{
	        dvxdz[k] = 0;
	        dvzdz[k] = 0;
	    }
	}

	__global__ void updateSY(float *sxy, float *szy, float *dvydx, float *dvydz, float *mu, float dt, Dim dim){
		size_t k = dim();
		sxy[k] += dt * mu[k] * dvydx[k];
		szy[k] += dt * mu[k] * dvydz[k];
	}
	__global__ void updateSXZ(float *sxx, float *szz, float *sxz, float *dvxdx, float *dvxdz, float *dvzdx, float *dvzdz,
		float *lambda, float *mu, float dt, Dim dim){
		size_t k = dim();
		sxx[k] += dt * ((lambda[k] + 2 * mu[k]) * dvxdx[k] + lambda[k] * dvzdz[k]);
		szz[k] += dt * ((lambda[k] + 2 * mu[k]) * dvzdz[k] + lambda[k] * dvxdx[k]);
		sxz[k] += dt * (mu[k] * (dvxdz[k] + dvzdx[k]));
	}
	__global__ void updateVY(float *vy, float *uy, float *dsy, float *rho, float *absbound, float dt, Dim dim){
		size_t k = dim();
		vy[k] = absbound[k] * (vy[k] + dt * dsy[k] / rho[k]);
		uy[k] += vy[k] * dt;
	}
	__global__ void updateVXZ(float *vx, float *vz, float *ux, float *uz, float *dsx, float *dsz, float *rho, float *absbound, float dt, Dim dim){
		size_t k = dim();
		vx[k] = absbound[k] * (vx[k] + dt * dsx[k] / rho[k]);
		vz[k] = absbound[k] * (vz[k] + dt * dsz[k] / rho[k]);
		ux[k] += vx[k] * dt;
		uz[k] += vz[k] * dt;
	}

	__global__ void addSTF(float *dsx, float *dsy, float *dsz, float *stf_x, float *stf_y, float *stf_z,
	    size_t *src_x_id, size_t *src_z_id, int isrc, bool sh, bool psv, size_t it, size_t nt, Dim dim){
	    size_t is = blockIdx.x;
	    size_t xs = src_x_id[is];
	    size_t zs = src_z_id[is];
		size_t ks = is * nt + it;
		size_t km = dim.k(xs, zs);

	    if (isrc < 0 || isrc == is) {
	        if (sh) {
	            dsy[km] += stf_y[ks];
	        }
	        if (psv) {
	            dsx[km] += stf_x[ks];
	            dsz[km] += stf_z[ks];
	        }
	    }
	}
	__global__ void saveRec(float *out_x, float *out_y, float *out_z, float *wx, float *wy, float *wz,
	    size_t *rec_x_id, size_t *rec_z_id, bool sh, bool psv, size_t it, size_t nt, Dim dim){
	    size_t ir = blockIdx.x;
	    size_t xr = rec_x_id[ir];
	    size_t zr = rec_z_id[ir];
		size_t kr = ir * nt + it;
		size_t km = dim.k(xr, zr);

	    if(sh){
	        out_y[kr] = wy[km];
	    }
	    if(psv){
	        out_x[kr] = wx[km];
	        out_z[kr] = wz[km];
	    }
	}

	__global__ void interactionRhoY(float *k_rho, float *vy, float *vy_fw, float ndt, Dim dim){
	    size_t k = dim();
	    k_rho[k] -= vy_fw[k] * vy[k] * ndt;
	}
	__global__ void interactionRhoXZ(float *k_rho, float *vx, float *vx_fw, float *vz, float *vz_fw, float ndt, Dim dim){
	    size_t k = dim();
	    k_rho[k] -= (vx_fw[k] * vx[k] + vz_fw[k] * vz[k]) * ndt;
	}
	__global__ void interactionMuY(float *k_mu, float *dvydx, float *dvydx_fw, float *dvydz, float *dvydz_fw, float ndt, Dim dim){
	    size_t k = dim();
	    k_mu[k] -= (dvydx[k] * dvydx_fw[k] + dvydz[k] * dvydz_fw[k]) * ndt;
	}
	__global__ void interactionMuXZ(float *k_mu, float *dvxdx, float *dvxdx_fw, float *dvxdz, float *dvxdz_fw,
	    float *dvzdx, float *dvzdx_fw, float *dvzdz, float *dvzdz_fw, float ndt, Dim dim){
	    size_t k = dim();
	    k_mu[k] -= (2 * dvxdx[k] * dvxdx_fw[k] + 2 * dvzdz[k] * dvzdz_fw[k] +
	        (dvxdz[k] + dvzdx[k]) * (dvzdx_fw[k] + dvxdz_fw[k])) * ndt;
	}
	__global__ void interactionLambdaXZ(float *k_lambda, float *dvxdx, float *dvxdx_fw, float *dvzdz, float *dvzdz_fw, float ndt, Dim dim){
	    size_t k = dim();
	    k_lambda[k] -= ((dvxdx[k] + dvzdz[k]) * (dvxdx_fw[k] + dvzdz_fw[k])) * ndt;
	}
}

class FdmSolver : public Solver  {
private:
	size_t nsfe;
	float xmax;
	float zmax;

	size_t *src_x_id;
	size_t *src_z_id;
	size_t *rec_x_id;
	size_t *rec_z_id;

	float *lambda;
	float *mu;
	float *rho;
	float *absbound;

	float *vx;
	float *vy;
	float *vz;
	float *ux;
	float *uy;
	float *uz;
	float *sxx;
	float *syy;
	float *szz;
	float *sxy;
	float *sxz;
	float *szy;

	float *dsx;
    float *dsy;
    float *dsz;
    float *dvxdx;
    float *dvxdz;
    float *dvydx;
    float *dvydz;
    float *dvzdx;
    float *dvzdz;

	float *dvydx_fw;
	float *dvydz_fw;

	float *dvxdx_fw;
	float *dvxdz_fw;
	float *dvzdx_fw;
	float *dvzdz_fw;

	float **vx_forward;
	float **vy_forward;
	float **vz_forward;

	float **ux_forward;
	float **uy_forward;
	float **uz_forward;

	void exportModel(string comp, float *data, size_t n = 0) {
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
	void exportAxis() {
		createDirectory(path_output);
		Dim dim(nx, nz);
		float *x = host::create(dim);
		float *z = host::create(dim);

		for (size_t i = 0; i < nx; i++) {
			for (size_t j = 0; j < nz; j++) {
				size_t k = dim.hk(i, j);
				x[k] = i * dx;
				z[k] = j * dz;
			}
		}

		exportModel("x", x);
		exportModel("z", z);
	};
	void initWavefields() {
		Dim dim(nx, nz);
		if (sh) {
			device::init(vy, 0, dim);
			device::init(uy, 0, dim);
			device::init(sxy, 0, dim);
			device::init(szy, 0, dim);
		}
		if (psv) {
			device::init(vx, 0, dim);
			device::init(vz, 0, dim);
			device::init(ux, 0, dim);
			device::init(uz, 0, dim);
			device::init(sxx, 0, dim);
			device::init(szz, 0, dim);
			device::init(sxz, 0, dim);
		}
	};
	void initKernels() {
		Dim dim(nx, nz);
		device::init(k_lambda, 0, dim);
		device::init(k_mu, 0, dim);
		device::init(k_rho, 0, dim);
	};
	void divS(Dim dim) {
		using namespace _FdmSolver;
		if (sh) {
			divSY<<<dim.dg, dim.db>>>(dsy, sxy, szy, dx, dz, dim);
		}
		if (psv) {
			divSXZ<<<dim.dg, dim.db>>>(dsx, dsz, sxx, szz, sxz, dx, dz, dim);
		}
	};
	void divV(Dim dim) {
		using namespace _FdmSolver;
		if(sh){
			updateVY<<<dim.dg, dim.db>>>(vy, uy, dsy, rho, absbound, dt, dim);
			divVY<<<dim.dg, dim.db>>>(dvydx, dvydz, vy, dx, dz, dim);
			updateSY<<<dim.dg, dim.db>>>(sxy, szy, dvydx, dvydz, mu, dt, dim);
		}
		if(psv){
			updateVXZ<<<dim.dg, dim.db>>>(vx, vz, ux, uz, dsx, dsz, rho, absbound, dt, dim);
			divVXZ<<<dim.dg, dim.db>>>(dvxdx, dvxdz, dvzdx, dvzdz, vx, vz, dx, dz, dim);
			updateSXZ<<<dim.dg, dim.db>>>(sxx, szz, sxz, dvxdx, dvxdz, dvzdx, dvzdz, lambda, mu, dt, dim);
		}
	};
	void exportSnapshot(size_t it) {
		Dim dim(nx, nz);
		if (wfe && (it + 1) % wfe == 0) {
			switch (obs) {
				case 0: {
					if (sh) {
						exportModel("vy", host::create(dim, vy), it + 1);
					}
					if (psv) {
						exportModel("vx", host::create(dim, vx), it + 1);
						exportModel("vz", host::create(dim, vz), it + 1);
					}
					break;
				}
				case 1: {
					if (sh) {
						exportModel("uy", host::create(dim, uy), it + 1);
					}
					if (psv) {
						exportModel("ux", host::create(dim, ux), it + 1);
						exportModel("uz", host::create(dim, uz), it + 1);
					}
					break;
				}
			}
		}
	};

public:
	void init(Config *config) {
		Solver::init(config);
		using namespace _FdmSolver;

		model_npt = 0;
		int npt = 0;
		auto read = [&](string comp, float &max) {
			std::ifstream infile(path_model_true + "/proc000000_" + comp + ".bin", std::ifstream::binary);
			infile.read(reinterpret_cast<char*>(&npt), sizeof(int));
			if (model_npt) {
				if (model_npt != npt) {
					std::cout << "error: invalid model" << std::endl;
				}
			}
			else {
				model_npt = npt;
			}

			float *buffer = host::create(model_npt);
			infile.read(reinterpret_cast<char*>(buffer), model_npt * sizeof(float));
			max = 0;
			for (size_t i = 0; i < model_npt; i++) {
				if (buffer[i] > max) {
					max = buffer[i];
				}
			}
			free(buffer);
			infile.close();
		};
		read("x", xmax);
		read("z", zmax);

		nx = std::round(sqrt(model_npt * xmax / zmax));
		nz = std::round(model_npt / nx);
		dx = xmax / (nx - 1);
		dz = zmax / (nz - 1);

		Dim dim(nx, nz);
		nsfe = nt / sfe;

		lambda = device::create(dim);
		mu = device::create(dim);
		rho = device::create(dim);
		absbound = device::create(dim);

		if (sh) {
			vy = device::create(dim);
			uy = device::create(dim);
			sxy = device::create(dim);
			szy = device::create(dim);
			dsy = device::create(dim);
			dvydx = device::create(dim);
			dvydz = device::create(dim);
		}

		if (psv) {
			vx = device::create(dim);
			vz = device::create(dim);
			ux = device::create(dim);
			uz = device::create(dim);
			sxx = device::create(dim);
			szz = device::create(dim);
			sxz = device::create(dim);
			dsx = device::create(dim);
			dsz = device::create(dim);
			dvxdx = device::create(dim);
			dvxdz = device::create(dim);
			dvzdx = device::create(dim);
			dvzdz = device::create(dim);
		}

		if(config->i["mode"] != 1){
            if(sh){
                dvydx_fw = device::create(dim);
                dvydz_fw = device::create(dim);

				vy_forward = host::create2D(nsfe, dim);
				uy_forward = host::create2D(nsfe, dim);
            }
            if(psv){
                dvxdx_fw = device::create(dim);
                dvxdz_fw = device::create(dim);
                dvzdx_fw = device::create(dim);
                dvzdz_fw = device::create(dim);

				vx_forward = host::create2D(nsfe, dim);
				ux_forward = host::create2D(nsfe, dim);
				vz_forward = host::create2D(nsfe, dim);
				uz_forward = host::create2D(nsfe, dim);
            }

            k_lambda = device::create(dim);
            k_mu = device::create(dim);
            k_rho = device::create(dim);
        }

		src_x_id = device::create<size_t>(nsrc);
		src_z_id = device::create<size_t>(nsrc);
		rec_x_id = device::create<size_t>(nrec);
		rec_z_id = device::create<size_t>(nrec);

		calcIdx<<<nsrc, 1>>>(src_x_id, src_x, xmax, nx);
        calcIdx<<<nsrc, 1>>>(src_z_id, src_z, zmax, nz);
        calcIdx<<<nrec, 1>>>(rec_x_id, rec_x, xmax, nx);
        calcIdx<<<nrec, 1>>>(rec_z_id, rec_z, zmax, nz);

		initAbsbound<<<dim.dg, dim.db>>>(
			absbound, abs_width, abs_param, abs_left, abs_right, abs_bottom, abs_top, dim
        );
	};
	void importModel(bool model) {
		Solver::importModel(model);
		using namespace _FdmSolver;

		Dim dim(nx, nz);
		mesh2grid<<<dim.dg, dim.db>>>(
			lambda, mu, rho,
			model_x, model_z, model_vp, model_vs, model_rho, model_npt,
			xmax * xmax + zmax * zmax, dx, dz, dim
		);
		vps2lm<<<dim.dg, dim.db>>>(lambda, mu, rho, dim);
	};
	void exportKernels() {
		createDirectory(path_output);
		exportAxis();
		Dim dim(nx, nz);
		exportModel("k_lambda", host::create(dim, k_lambda));
		exportModel("k_mu", host::create(dim, k_mu));
		exportModel("k_rho", host::create(dim, k_rho));
	};
	void runForward(int isrc, bool adjoint = false, bool trace = false, bool snapshot = false) {
		using namespace _FdmSolver;
		Dim dim(nx, nz);

		if (wfe && snapshot) {
			exportAxis();
		}

		initWavefields();

		for (size_t it = 0; it < nt; it++) {
			int isfe = -1;
			if (sfe && adjoint && (it + 1) % sfe == 0) {
				isfe = nsfe - (it + 1) / sfe;
			}
			if (isfe >= 0) {
				if(sh){
					device::toHost(uy_forward[isfe], uy, dim);
				}
				if(psv){
					device::toHost(ux_forward[isfe], ux, dim);
					device::toHost(uz_forward[isfe], uz, dim);
				}
			}
			divS(dim);
			addSTF<<<nsrc, 1>>>(
				dsx, dsy, dsz, stf_x, stf_y, stf_z,
				src_x_id, src_z_id, isrc, sh, psv, it, nt, dim
			);
			divV(dim);
			if (adjoint || obs == 1) {
				saveRec<<<nrec, 1>>>(
					out_x, out_y, out_z, ux, uy, uz,
					rec_x_id, rec_z_id, sh, psv, it, nt, dim
				);
			}
			else if(obs == 0) {
				saveRec<<<nrec, 1>>>(
					out_x, out_y, out_z, vx, vy, vz,
					rec_x_id, rec_z_id, sh, psv, it, nt, dim
				);
			}
			if (isfe >= 0) {
				if(sh){
					device::toHost(vy_forward[isfe], vy, dim);
				}
				if(psv){
					device::toHost(vx_forward[isfe], vx, dim);
					device::toHost(vz_forward[isfe], vz, dim);
				}
			}

			if (snapshot) {
				exportSnapshot(it);
			}
		}

		if (trace) {
			exportTraces(isrc);
		}
	};
	void runAdjoint(int isrc, bool snapshot = false) {
		using namespace _FdmSolver;
		Dim dim(nx, nz);

		initWavefields();

		for (size_t it = 0; it < nt; it++) {
			divS(dim);
			addSTF<<<nrec, 1>>>(
				dsx, dsy, dsz, adstf_x, adstf_y, adstf_z,
				rec_x_id, rec_z_id, -1, sh, psv, it, nt, dim
			);
			divV(dim);
			if ((it + sfe) % sfe == 0) {
				size_t isfe = (it + sfe) / sfe - 1;
				float ndt = sfe * dt;
				if (sh) {
					host::toDevice(dsy, uy_forward[isfe], dim);
                    divVY<<<dim.dg, dim.db>>>(dvydx, dvydz, uy, dx, dz, dim);
                    divVY<<<dim.dg, dim.db>>>(dvydx_fw, dvydz_fw, dsy, dx, dz, dim);
                    host::toDevice(dsy, vy_forward[isfe], dim);
                    interactionRhoY<<<dim.dg, dim.db>>>(k_rho, vy, dsy, ndt, dim);
                    interactionMuY<<<dim.dg, dim.db>>>(k_mu, dvydx, dvydx_fw, dvydz, dvydz_fw, ndt, dim);
				}
				if (psv) {
					host::toDevice(dsx, ux_forward[isfe], dim);
                    host::toDevice(dsz, uz_forward[isfe], dim);
                    divVXZ<<<dim.dg, dim.db>>>(dvxdx, dvxdz, dvzdx, dvzdz, ux, uz, dx, dz, dim);
                    divVXZ<<<dim.dg, dim.db>>>(dvxdx_fw, dvxdz_fw, dvzdx_fw, dvzdz_fw, dsx, dsz, dx, dz, dim);

                    host::toDevice(dsx, vx_forward[isfe], dim);
                    host::toDevice(dsz, vz_forward[isfe], dim);
                    interactionRhoXZ<<<dim.dg, dim.db>>>(k_rho, vx, dsx, vz, dsz, ndt, dim);
                    interactionMuXZ<<<dim.dg, dim.db>>>(k_mu, dvxdx, dvxdx_fw, dvxdz, dvxdz_fw, dvzdx, dvzdx_fw, dvzdz, dvzdz_fw, ndt, dim);
                    interactionLambdaXZ<<<dim.dg, dim.db>>>(k_lambda, dvxdx, dvxdx_fw, dvzdz, dvzdz_fw, ndt, dim);
				}
			}

			if (snapshot) {
				exportSnapshot(it);
			}
		}
	};
};
