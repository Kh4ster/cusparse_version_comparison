#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cusparse.h>
#include <cuda_runtime_api.h>
#include <chrono>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        exit(-1);                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        exit(-1);                                                   \
    }                                                                          \
}

// Function to load vector from file
template <typename T>
void load_vector(std::vector<T>& vec, const std::string& filename)
{
    std::ifstream in_file(filename, std::ios::binary);
    if(!in_file)
    {
        std::cerr << "Cannot open the file: " << filename << std::endl;
        return;
    }

    size_t size;
    in_file.read(reinterpret_cast<char*>(&size), sizeof(size));
    vec.resize(size);
    in_file.read(reinterpret_cast<char*>(vec.data()), vec.size() * sizeof(T));
    in_file.close();
}

// Compute SpMV on CPU
std::vector<double> spmv_cpu(const std::vector<double>& values, 
                             const std::vector<int>& indices, 
                             const std::vector<int>& offsets, 
                             const std::vector<double>& x,
                             int n_constraints)
{
    // Initialize output vector with zeros
    std::vector<double> y(n_constraints, 0.0);
    
    // Ensure offsets size matches n_constraints + 1 for valid CSR representation
    if (offsets.size() != n_constraints + 1) {
        std::cerr << "Invalid CSR representation: offsets size does not match number of constraints.\n";
        return y;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    // Loop over each row
    for (int i = 0; i < n_constraints; ++i) {
        // Loop over non-zeros in the current row
        for (int j = offsets[i]; j < offsets[i + 1]; ++j) {
            // Perform multiplication and accumulate results
            y[i] += values[j] * x[indices[j]];
        }
    }

    return y;
}

std::vector<double> spmv_gpu(const double* d_values,
                             const int* d_indices,
                             const int* d_offsets,
                             const double* d_x,
                             cusparseHandle_t& handle,
                             int n_variables,
                             int n_constraints)
{
    constexpr double alpha = 1.0;
    constexpr double beta = 0.0;

    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, n_constraints, n_variables, n_constraints,
                      (void*)d_offsets, (void*)d_indices, (void*)d_values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    cusparseDnVecDescr_t vecX;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, n_variables, (void*)d_x, CUDA_R_64F));

    // Vector used only for output as beta == 0
    double* d_y;
    CHECK_CUDA(cudaMalloc(&d_y, n_constraints * sizeof(double)));
    cusparseDnVecDescr_t vecY;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, n_constraints, (void*)d_y, CUDA_R_64F));
    
    size_t bufferSize = 0;
    void* dBuffer;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                 CUSPARSE_SPMV_CSR_ALG2, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    
    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, matA, vecX, &beta, vecY, CUDA_R_64F, 
                 CUSPARSE_SPMV_CSR_ALG2, dBuffer));

    std::vector<double> y(n_constraints);
    CHECK_CUDA(cudaMemcpy(y.data(), d_y, n_constraints * sizeof(double), cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    
    return y;
}

int main() {
    int major_version, minor_version;
    cusparseGetProperty(libraryPropertyType_t::MAJOR_VERSION, &major_version);
    cusparseGetProperty(libraryPropertyType_t::MINOR_VERSION, &minor_version);
    std::cout << "cuSparse version = " << major_version << "." << minor_version << std::endl;

    constexpr int n_variables = 17680;
    constexpr int n_constraints = 69608;

    // Load data
    std::vector<double> values, x;
    std::vector<int> indices, offsets;

    load_vector(values, "A_values");
    load_vector(x, "X");
    load_vector(offsets, "A_offsets");
    load_vector(indices, "A_indices");

    // Transfer data to device
    double* d_values;
    double* d_x;
    int* d_indices;
    int* d_offsets;
    CHECK_CUDA(cudaMalloc(&d_values, values.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_x, x.size() * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_indices, indices.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_offsets, offsets.size() * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_values, values.data(), values.size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_indices, indices.data(), indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Create cuSPARSE handle
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    
    // Compute SpMV on GPU
    std::vector<double> y_gpu = spmv_gpu(d_values, d_indices, d_offsets, d_x, handle, n_variables, n_constraints);
    std::vector<double> y_cpu = spmv_cpu(values, indices, offsets, x, n_constraints);
    
    // Compare the results
    double l2_diff = 0.0;
    int count_diff = 0;
    for (size_t i = 0; i < y_cpu.size(); ++i) {
        double diff = y_cpu[i] - y_gpu[i];
        l2_diff += diff * diff;  // add the square of the difference
        if (std::abs(diff) > 0.1) {
            ++count_diff;  // increment the counter
        }
    }
    l2_diff = std::sqrt(l2_diff);  // take the square root to complete the L2 norm computation
    std::cout << "L2 difference between CPU and custom GPU results: " << l2_diff << std::endl;
    std::cout << "Number of values with absolute difference > 0.1: " << count_diff << std::endl;

    // Clean up
    CHECK_CUDA(cudaFree(d_values));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_indices));
    CHECK_CUDA(cudaFree(d_offsets));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return 0;
}