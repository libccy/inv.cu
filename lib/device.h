#pragma once

class Dim {
public:
	size_t nx;
	size_t nz;
	size_t dg;
	size_t db;
	bool row_major;
	Dim(size_t nx, size_t nz, bool row_major=true):
		nx(nx), nz(nz), row_major(row_major) {
		dg = nx;
		db = nz;
	};
	__device__ size_t k(size_t i, size_t j) {
		if (row_major) {
			return i * nz + j;
		}
		else {
			return j * nx + i;
		}
	};
	__device__ void operator()(size_t &i, size_t &j, size_t &k, int di = 0, int dj = 0) {
		i = blockIdx.x + di;
		j = threadIdx.x + dj;
		k = this->k(i, j);
	};
	__device__ size_t operator()(int di = 0, int dj = 0) {
		size_t i, j, k;
		this->operator()(i, j, k, di, dj);
		return k;
	};

	size_t hk(size_t i, size_t j) {
		if (row_major) {
			return i * nz + j;
		}
		else {
			return j * nx + i;
		}
	};
	operator size_t() const {
		return nx * nz;
	}
};

namespace host {
	template<typename T = float>
	T *create(size_t len) {
		T *data = (T *)malloc(len * sizeof(T));
		return data;
	}
	template<typename T = float>
	T *create(size_t len, T *data) {
		T *h_data = (T *)malloc(len * sizeof(T));
		cudaMemcpy(h_data, data, len * sizeof(T), cudaMemcpyDeviceToHost);
		return h_data;
	}
	template<typename T = float>
	T **create2D(size_t n, size_t len) {
		T **data = (T **)malloc(n * sizeof(T *));
		for (size_t i = 0; i < n; i++) {
			data[i] =  (T *)malloc(len * sizeof(T));
		}
		return data;
	}
	template<typename T = float>
	T *toDevice(T *data, T *h_data, size_t len) {
		cudaMemcpy(data, h_data, len * sizeof(T), cudaMemcpyHostToDevice);
		return h_data;
	}
}

namespace device {
	__constant__ float pi = 3.1415927;

	cublasHandle_t cublas_handle = NULL;
	cusolverDnHandle_t solver_handle = NULL;
	// size_t max_thread = 1024;

	template<typename T = float>
	T *create(size_t len) {
		T *data;
		cudaMalloc((void **)&data, len * sizeof(T));
		return data;
	}
	template<typename T = float>
	T *create(size_t len, T *h_data) {
		T *data;
		cudaMalloc((void **)&data, len * sizeof(T));
		cudaMemcpy(data, h_data, len * sizeof(T), cudaMemcpyHostToDevice);
		return data;
	}
	template<typename T = float>
	T **create2D(size_t n, size_t len) {
		T **data = (T **)malloc(n * sizeof(T *));
		for (size_t i = 0; i < n; i++) {
			cudaMalloc((void **)&data[i], len * sizeof(T));
		}
		return data;
	}
	template<typename T = float>
	T *toHost(T *h_data, T *data, size_t len) {
		cudaMemcpy(h_data, data, len * sizeof(T), cudaMemcpyDeviceToHost);
		return h_data;
	}

	template<typename T = float>
	__global__ void _copy(T *data, T *source, Dim dim) {
		size_t k = dim();
		data[k] = source[k];
	}
	template<typename T = float>
	__global__ void _init(T *data, T value, Dim dim){
		size_t k = dim();
		data[k] = value;
	}
	template<typename T = float>
	void copy(T*data, T *source, Dim dim) {
		_copy<<<dim.dg, dim.db>>>(data, source, dim);
	}
	template<typename T = float>
	T *copy(T *source, Dim dim) {
		T *data = create<T>(dim);
		_copy<<<dim.dg, dim.db>>>(data, source, dim);
		return data;
	}
	template<typename T = float>
	void init(T *data, T value, Dim dim){
		_init<<<dim.dg, dim.db>>>(data, value, dim);
	}
	template<typename T = float>
	T get(T *a, size_t ia) {
		T b[1];
		toHost(b, a + ia, 1);
		return b[0];
	}

	__global__ void _calc(float *c, float ka, float *a, Dim dim) {
		size_t k = dim();
		c[k] = ka * a[k];
	}
	__global__ void _calc(float *c, float ka, float *a, float kb, float *b, Dim dim) {
		size_t k = dim();
		c[k] = ka * a[k] + kb * b[k];
	}
	__global__ void _calc(float *c, float *a, float *b, Dim dim) {
		size_t k = dim();
		c[k] = a[k] * b[k];
	}
	void calc(float *c, float ka, float *a, Dim dim) {
		_calc<<<dim.dg, dim.db>>>(c, ka, a, dim);
	}
	void calc(float *c, float ka, float *a, float kb, float *b, Dim dim) {
		_calc<<<dim.dg, dim.db>>>(c, ka, a, kb, b, dim);
	}
	void calc(float *c, float *a, float *b, Dim dim) {
		_calc<<<dim.dg, dim.db>>>(c, a, b, dim);
	}
	float amax(float *a, size_t len){
		int index = 0;
		cublasIsamax_v2(cublas_handle, len, a, 1, &index);
		return fabs(get(a, index - 1));
	}
	float norm(float *a, size_t len){
        float norm_a = 0;
        cublasSnrm2_v2(cublas_handle, len, a, 1, &norm_a);
        return norm_a;
    }
}
