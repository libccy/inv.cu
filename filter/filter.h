#pragma once

class Filter {
protected:
	size_t nx;
	size_t nz;
	size_t sigma;

public:
	virtual void init(size_t nx, size_t nz, size_t sigma) {
		this->nx = nx;
		this->nz = nz;
		this->sigma = sigma;
	};
	virtual void apply(float *) = 0;
	virtual ~Filter() {};
};