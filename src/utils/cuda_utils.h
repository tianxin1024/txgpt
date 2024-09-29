#pragma once

#include "utils/string_utils.h"
#include "utils/logger.h"

#include <fstream>
#include <string>
#include <iostream>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace fastertransformer {

#define MAX_CONFIG_NUM 20
#define COL32_ 32
// workspace for cublas gemm : 32MB
#define CUBLAS_WORKSPACE_SIZE 33554432

typedef struct __align__(4) {
    half x, y, z, w;
}
half4;

/* **************************** type definition ***************************** */

enum CublasDataType {
    FLOAT_DATATYPE = 0,
    HALF_DATATYPE = 1,
    BFLOAT16_DATATYPE = 2,
    INT8_DATATYPE = 3,
    FP8_DATATYPE = 4
};

enum FtCudaDataType {
    FP32 = 0,
    FP16 = 1,
    BF16 = 2,
    INT8 = 3,
    FP8 = 4
};

enum class OperationType {
    FP32,
    FP16,
    BF16,
    INT8,
    FP8
};

/* **************************** debug tools ********************************* */
static const char *_cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorString(error);
}

static const char *_cudaGetErrorEnum(cublasStatus_t error) {
    switch (error) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";

    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";

    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    }
    return "<unknown>";
}

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " "
                                 + file + ":" + std::to_string(line) + " \n");
    }
}

#define check_cuda_error(val) check((val), #val, __FILE__, __LINE__)
#define check_cuda_error_2(val, file, line) check((val), #val, file, line)

inline void syncAndCheck(const char *const file, int const line) {
    // When FT_DEBUG_LEVEL=DEBUG, must check error
    static char *level_name = std::getenv("FT_DEBUG_LEVEL");
    if (level_name != nullptr) {
        static std::string level = std::string(level_name);
        if (level == "DEBUG") {
            cudaDeviceSynchronize();
            cudaError_t result = cudaGetLastError();
            if (result) {
                throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result))
                                         + " " + file + ":" + std::to_string(line) + " \n");
            }
            FT_LOG_DEBUG(fmtstr("run syncAndCheck at %s:%d", file, line));
        }
    }

#ifndef NDEBUG
    cudaDeviceSynchronize();
    cudaError_t result = cudaGetLastError();
    if (result) {
        throw std::runtime_error(std::string("[FT][ERROR] CUDA runtime error: ") + (_cudaGetErrorEnum(result)) + " "
                                 + file + ":" + std::to_string(line) + " \n");
    }
#endif
}

#define sync_check_cuda_error() syncAndCheck(__FILE__, __LINE__)

#define checkCUDNN(expression)                                                                                       \
    {                                                                                                                \
        cudnnStatus_t status = (expression);                                                                         \
        if (status != CUDNN_STATUS_SUCCESS) {                                                                        \
            std::cerr << "Error on file " << __FILE__ << " line " << __LINE__ << ": " << cudnnGetErrorString(status) \
                      << std::endl;                                                                                  \
            std::exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                            \
    }

template <typename T>
void print_to_file(const T *result,
                   const int size,
                   const char *file,
                   cudaStream_t stream = 0,
                   std::ios::openmode open_mode = std::ios::out);

template <typename T>
void print_abs_mean(const T *buf, uint size, cudaStream_t stream, std::string name = "");

template <typename T>
void print_to_screen(const T *result, const int size);

template <typename T>
void printMatrix(T *ptr, int m, int k, int stride, bool is_device_ptr);

void printMatrix(unsigned long long *ptr, int m, int k, int stride, bool is_device_ptr);
void printMatrix(int *ptr, int m, int k, int stride, bool is_device_ptr);
void printMatrix(size_t *ptr, int m, int k, int stride, bool is_device_ptr);

template <typename T>
void check_max_val(const T *result, const int size);

template <typename T>
void check_abs_mean_val(const T *result, const int size);

#define PRINT_FUNC_NAME_()                                              \
    do {                                                                \
        std::cout << "[FT][CALL] " << __FUNCTION__ << " " << std::endl; \
    } while (0)

[[noreturn]] inline void throwRuntimeError(const char *const file, int const line, std::string const &info = "") {
    throw std::runtime_error(std::string("[FT][ERROR] ") + info + " Assertion fail: " + file + ":"
                             + std::to_string(line) + " \n");
}

inline void myAssert(bool result, const char *const file, int const line, std::string const &info = "") {
    if (!result) {
        throwRuntimeError(file, line, info);
    }
}

#define FT_CHECK(val) myAssert(val, __FILE__, __LINE__)
#define FT_CHECK_WITH_INFO(val, info)                                              \
    do {                                                                           \
        bool is_valid_val = (val);                                                 \
        if (!is_valid_val) {                                                       \
            fastertransformer::myAssert(is_valid_val, __FILE__, __LINE__, (info)); \
        }                                                                          \
    } while (0)

#define FT_THROW(info) throwRuntimeError(__FILE__, __LINE__, info)

#ifdef SPARSITY_ENABLED
#define CHECK_CUSPARSE(func)                                                                           \
    {                                                                                                  \
        cusparseStatus_t status = (func);                                                              \
        if (status != CUSPARSE_STATUS_SUCCESS) {                                                       \
            throw std::runtime_error(std::string("[FT][ERROR] CUSPARSE API failed at line ")           \
                                     + std::to_string(__LINE__) + " in file " + __FILE__ + ": "        \
                                     + cusparseGetErrorString(status) + " " + std::to_string(status)); \
        }                                                                                              \
    }
#endif

/*************Time Handling**************/
class CudaTimer {
private:
    cudaEvent_t event_start_;
    cudaEvent_t event_stop_;
    cudaStream_t stream_;

public:
    explicit CudaTimer(cudaStream_t stream = 0) {
        stream_ = stream;
    }
    void start() {
        check_cuda_error(cudaEventCreate(&event_start_));
        check_cuda_error(cudaEventCreate(&event_stop_));
        check_cuda_error(cudaEventRecord(event_start_, stream_));
    }
    float stop() {
        float time;
        check_cuda_error(cudaEventRecord(event_stop_, stream_));
        check_cuda_error(cudaEventSynchronize(event_stop_));
        check_cuda_error(cudaEventElapsedTime(&time, event_start_, event_stop_));
        check_cuda_error(cudaEventDestroy(event_start_));
        check_cuda_error(cudaEventDestroy(event_stop_));
        return time;
    }
    ~CudaTimer() {
    }
};

static double diffTime(timeval start, timeval end) {
    return (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
}

} // namespace fastertransformer
