#include <fstream>
#include <filesystem>
#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <algorithm>
#include <omp.h>
#include "mpi.h"
#include "param.h"
#include "util.h"
#include "psi.h"

// Macro for checking CUDA errors.
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Performance monitoring variables
static double total_file_read_time = 0;
static double total_computation_time = 0;
static double total_memory_transfer_time = 0;
static double total_communication_time = 0;
static double total_verification_time = 0;
static double cmp_time1=0, cmp_time2=0, cmp_time3=0, cmp_time4=0, cmp_time5=0;

// GPU memory management
static char *buf_cuda = nullptr;
static int *bias_cuda = nullptr;
static unsigned int *hash_result_cuda = nullptr;
static unsigned int *cuda_buf1 = nullptr;
static unsigned char* cuda_cmp_result = nullptr;
static int* cuda_reduce_result = nullptr;
static int* cuda_reduce_cnt = nullptr;
static int* cuda_reduce_buf = nullptr;
static unsigned char* cmp_result = nullptr;
static int* reduce_result = nullptr;
static int *par = nullptr, *par_new = nullptr;

// Bucket and key-related variables
static int max_bucket, num_key, buf_c;
static int file_offload;

// Constants for optimization
#define REDUCE_BLOCK 240
#define REDUCE_EDGE_TH 50
#define BLOCK_SIZE 32
#define TILE_SIZE 32
#define THREAD_NUM BLOCK_SIZE*BLOCK_SIZE
#define REDUCE_THREAD 32

namespace fs = std::filesystem;

// Parameters
static int num_hash = 128;          // Number of hash functions
static int len_shingle = 5;         // Length of each shingle
static int b = 16;                  // Number of bands
static double threshold = 0.8;       // Similarity threshold
static int num_file = 2;             // Number of files (Party A and Party B)
static int th;                      // Similarity threshold (integer form)

// Hashing variables
static unsigned int *p, *q, *r;
static unsigned int *_p, *_q, *_r;
static unsigned int *hash_result, *total_hash_result;
static unsigned int *buf[C];
static int *file_idx[C], *line_idx[C];

// Function declarations
void process_signature_batch(const std::vector<std::string>& lines, const std::vector<int>& line_lengths, std::ofstream& outfile);
void process_buckets_batch(const std::vector<std::vector<unsigned int>>& signatures, std::ofstream& outfile);
void process_candidates_batch(const std::vector<std::vector<unsigned int>>& a_buckets_batch, 
                             const std::vector<int>& a_indices, 
                             const std::vector<std::vector<unsigned int>>& b_buckets_all, 
                             std::ofstream& outfile);

// Kernel declarations
__global__ void hash_string_kernel_psi(char *buf, int *bias, unsigned int *_p, unsigned int *_q, unsigned int *_r, int num_line, int len_shingle, int num_hash, int b, unsigned int *hash_result);
__global__ void generate_key_kernel_psi(unsigned int *hash_result, int num_line, int num_hash, int b, int num_key);
__global__ void reduce_kernel(int* input, int* output, int size);
__global__ void compute_matches_kernel(unsigned int* a_signatures, unsigned int* b_signatures, int* matches, int batch_size, int num_hash);

// Optimized kernels from lsh.cu
__global__ void compare_lsh_kernel(unsigned int *buf, unsigned char* result, int line_num, int num_hash, int th);
__global__ void reduce_compare_result1(unsigned char* result, int *cuda_reduce_buf, int line_num);
__global__ void reduce_compare_result2(int* cuda_reduce_buf, int *total);
__global__ void reduce_compare_result3(unsigned char* result, int *cuda_reduce_buf, int *output, int line_num);



// Print comparison time (from lsh.cu)
void print_cmp_time_psi() {
    std::cout << "  - file read + buffering: " <<  cmp_time1 << " seconds" << std::endl;
    std::cout << "  - Comm time1: " <<  cmp_time2 << " seconds" << std::endl;
    std::cout << "  - GPU kernel : " <<  cmp_time3 << " seconds" << std::endl;
    std::cout << "  - Comm time2: " <<  cmp_time4 << " seconds" << std::endl;
    std::cout << "  - Union time: " <<  cmp_time5 << " seconds" << std::endl;
}

// CUDA kernel for reduction operation (from lsh.cu)
__global__ void reduce_kernel(int* input, int* output, int size) {
    __shared__ int sdata[REDUCE_BLOCK];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < size) {
        sdata[tid] = input[i];
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();
    
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Perform reduction on GPU (from lsh.cu)
int perform_reduction(int* d_input, int size) {
    int num_blocks = (size + REDUCE_BLOCK - 1) / REDUCE_BLOCK;
    int* d_output;
    cudaMalloc(&d_output, sizeof(int) * num_blocks);
    
    reduce_kernel<<<num_blocks, REDUCE_BLOCK>>>(d_input, d_output, size);
    cudaDeviceSynchronize();
    
    int* h_output = (int*)malloc(sizeof(int) * num_blocks);
    cudaMemcpy(h_output, d_output, sizeof(int) * num_blocks, cudaMemcpyDeviceToHost);
    
    int result = 0;
    for (int i = 0; i < num_blocks; i++) {
        result += h_output[i];
    }
    
    free(h_output);
    cudaFree(d_output);
    return result;
}

// Kernel to compute hash values for strings (from lsh.cu)
__global__ void hash_string_kernel_psi(char *buf, int *bias, unsigned int *_p, unsigned int *_q, unsigned int *_r, int num_line, int len_shingle, int num_hash, int b, unsigned int *hash_result) {
    int line_id = blockIdx.x;       // Line index
    int hash_id = threadIdx.x;      // Hash index

    if (hash_id >= num_hash) return;

    unsigned int sum = 0;           // Intermediate hash sum
    unsigned int res = 0;           // Minimum hash value

    int len = bias[line_id + 1] - bias[line_id]; // Length of the string segment
    if (len < len_shingle) {
        // For short texts, use the entire text
        unsigned int hash = 0;
        unsigned int p_val = _p[hash_id];
        unsigned int q_val = _q[hash_id];
        char *text = buf + bias[line_id];
        for (int i = 0; i < len; i++) {
            hash = (hash * p_val + text[i]) % q_val;
        }
        res = hash;
    } else {
        char *text = buf + bias[line_id];          // Start of the text for this line

        unsigned int p = _p[hash_id];              // Prime coefficient for hash function
        unsigned int q = _q[hash_id];              // Modulus for hash function
        unsigned int r = _r[hash_id];              // Precomputed power for hash rolling

        // Compute hash for the initial window
        for (int i = 0; i < len_shingle; i++) {
            sum = (sum * p + text[i]) % q;
        }
        res = sum;
        // Compute hash for the rolling window
        for (int i = len_shingle; i < len; i++) {
            sum = (sum * p + ((unsigned int)text[i - len_shingle]) * r + text[i]) % q;
            res = min(res, sum);
        }
    }
    // Store the resulting hash value
    hash_result[line_id * (num_hash + b) + hash_id] = res;
}

// Kernel to generate keys from hashed values (from lsh.cu)
__global__ void generate_key_kernel_psi(unsigned int *hash_result, int num_line, int num_hash, int b, int num_key) {
    int line_id = blockIdx.x;       // Line index
    int b_id = threadIdx.x;         // Band index

    if (b_id >= b) return;

    unsigned int sum = 0;           // Sum of hash values for the band
    int h = num_hash / b;           // Number of hash functions per band

    // Sum hash values for the current band
    for(int i=b_id*h; i<(b_id+1)*h; i++) sum+=hash_result[line_id*(num_hash+b) + i];
    
    // Generate the band key
    hash_result[line_id*(num_hash+b) + num_hash+b_id] = sum%num_key;
}
void process_candidates_batch(const std::vector<std::vector<unsigned int>>& a_buckets_batch, 
                             const std::vector<int>& a_indices, 
                             const std::vector<std::vector<unsigned int>>& b_buckets_all, 
                             std::ofstream& outfile);

// GPU kernel declarations
__global__ void generate_minhash_batch(const char* text, int* text_offsets, int shingle_len, 
                                    unsigned int* p, unsigned int* q, unsigned int* r, 
                                    int num_hash, unsigned int* signatures);

// Find root of a node for union-find
int root(int x) {
    if(par[x]==x) return x;
    int tmp=root(par[x]);
    par[x]=tmp;
    return tmp;
}

// Merge two sets in union-find
void merge(int x, int y) {
    x=root(x);
    y=root(y);
    if(x==y) return;
    par[y]=x;
}

// Initialize PSI with LSH parameters
void init_psi(int _num_hash, int _shingle_len, int _bucket, double _threshold) {
    num_hash = _num_hash;
    len_shingle = _shingle_len;
    b = _bucket;
    threshold = _threshold;
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Initialize hash functions (similar to LSH implementation)
    p = (unsigned int*)malloc(sizeof(unsigned int) * num_hash);
    q = (unsigned int*)malloc(sizeof(unsigned int) * num_hash);
    r = (unsigned int*)malloc(sizeof(unsigned int) * num_hash);
    
    for(int i=0; i<num_hash; i++) {
        q[i] = 4294967; // Prime number
        p[i] = 257 + i;
        
        r[i] = q[i] - 1;
        for(int j=0; j<len_shingle; j++) {
            r[i] = (r[i] * p[i]) % q[i];
        }
    }
    
    // Allocate CUDA memory
    gpuErrchk(cudaMalloc((void**)&_p, sizeof(unsigned int) * num_hash));
    gpuErrchk(cudaMalloc((void**)&_q, sizeof(unsigned int) * num_hash));
    gpuErrchk(cudaMalloc((void**)&_r, sizeof(unsigned int) * num_hash));
    
    // Allocate GPU memory for batch processing
    gpuErrchk(cudaMalloc((void**)&hash_result_cuda, sizeof(unsigned int) * MAX_LINE * (num_hash + b)));
    gpuErrchk(cudaMalloc((void**)&bias_cuda, sizeof(int) * (MAX_LINE + 1)));
    gpuErrchk(cudaMalloc((void**)&buf_cuda, 3e9)); // 3GB buffer for text data
    
    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Set parameters and get bucket-related values
    set_param(2, num_hash + b); // Assume 2 files for PSI
    get_param(num_key, max_bucket, buf_c, file_offload);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Allocate host memory
    hash_result = (unsigned int*)malloc(sizeof(unsigned int) * MAX_LINE * (num_hash + b));
    
    // Allocate buffers for bucket processing
    for(int i=0; i<buf_c; i++) {
        buf[i] = (unsigned int*)malloc(sizeof(unsigned int) * num_hash * max_bucket);
        file_idx[i] = (int*)malloc(sizeof(int) * max_bucket);
        line_idx[i] = (int*)malloc(sizeof(int) * max_bucket);
    }
    
    // Allocate GPU memory for comparison and reduction
    gpuErrchk(cudaMalloc((void**)&cuda_cmp_result, sizeof(unsigned char) * max_bucket * max_bucket));
    MPI_Barrier(MPI_COMM_WORLD);
    gpuErrchk(cudaMalloc((void**)&cuda_buf1, sizeof(unsigned int) * num_hash * max_bucket));
    
    gpuErrchk(cudaMalloc((void**)&cuda_reduce_result, sizeof(int) * max_bucket * REDUCE_EDGE_TH * 2)); 
    gpuErrchk(cudaMalloc((void**)&cuda_reduce_buf, sizeof(int) * REDUCE_BLOCK));
    gpuErrchk(cudaMalloc((void**)&cuda_reduce_cnt, sizeof(int)));
    
    // Allocate host memory for comparison and reduction
    cmp_result = (unsigned char*)malloc(sizeof(unsigned char) * max_bucket * max_bucket);
    reduce_result = (int*)malloc(sizeof(int) * max_bucket * max_bucket);
    
    par = (int*)malloc(sizeof(int) * MAX_LINE * 2); // Assume 2 files
    par_new = (int*)malloc(sizeof(int) * MAX_LINE * 2);
    
    // Copy hash parameters to device
    gpuErrchk(cudaMemcpy(_p, p, sizeof(unsigned int) * num_hash, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(_q, q, sizeof(unsigned int) * num_hash, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(_r, r, sizeof(unsigned int) * num_hash, cudaMemcpyHostToDevice));
    
    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
        std::cout << "PSI initialized with parameters: " << std::endl;
        std::cout << "  - Number of hash functions: " << num_hash << std::endl;
        std::cout << "  - Shingle length: " << len_shingle << std::endl;
        std::cout << "  - Number of bands: " << b << std::endl;
        std::cout << "  - Similarity threshold: " << threshold << std::endl;
        std::cout << "  - Max bucket: " << max_bucket << std::endl;
        std::cout << "  - Number of keys: " << num_key << std::endl;
    }
}

// Generate shingles from text
std::vector<std::string> generate_shingles(const std::string& text) {
    std::vector<std::string> shingles;
    if (text.length() < len_shingle) {
        shingles.push_back(text);
        return shingles;
    }
    
    for (size_t i = 0; i <= text.length() - len_shingle; i++) {
        shingles.push_back(text.substr(i, len_shingle));
    }
    return shingles;
}

// Generate MinHash signature for a document using rolling hash
__global__ void generate_minhash(const char* text, int text_len, int shingle_len, 
                               unsigned int* p, unsigned int* q, unsigned int* r, 
                               int num_hash, unsigned int* signature) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_hash) {
        unsigned int min_hash = UINT_MAX;
        
        if (text_len < shingle_len) {
            // For short texts, use the entire text as a single shingle
            unsigned int hash = 0;
            unsigned int p_val = p[tid];
            unsigned int q_val = q[tid];
            for (int i = 0; i < text_len; i++) {
                hash = (hash * p_val + text[i]) % q_val;
            }
            min_hash = hash;
        } else {
            // Use rolling hash technique for efficiency
            unsigned int sum = 0;
            unsigned int p_val = p[tid];
            unsigned int q_val = q[tid];
            unsigned int r_val = r[tid];
            
            // Compute hash for the initial window
            for (int i = 0; i < shingle_len; i++) {
                sum = (sum * p_val + text[i]) % q_val;
            }
            min_hash = sum;
            
            // Compute hash for rolling window
            for (int i = shingle_len; i < text_len; i++) {
                // Rolling hash formula: remove the leftmost character and add the new character
                sum = (sum * p_val + ((unsigned int)text[i - shingle_len]) * r_val + text[i]) % q_val;
                if (sum < min_hash) {
                    min_hash = sum;
                }
            }
        }
        signature[tid] = min_hash;
    }
}

// Generate LSH signatures for a dataset with batch processing
void generate_signatures(const std::string& input_file, const std::string& output_signatures) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::ifstream infile(input_file);
    std::ofstream outfile(output_signatures, std::ios::binary);
    std::string line;
    
    // Batch processing parameters
    const int BATCH_SIZE = 1024;
    std::vector<std::string> batch_lines;
    std::vector<int> batch_line_lengths;
    
    auto file_read_start = std::chrono::high_resolution_clock::now();
    
    // Read and process in batches
    while (std::getline(infile, line)) {
        batch_lines.push_back(line);
        batch_line_lengths.push_back(line.length());
        
        // Process batch when it's full
        if (batch_lines.size() >= BATCH_SIZE) {
            process_signature_batch(batch_lines, batch_line_lengths, outfile);
            batch_lines.clear();
            batch_line_lengths.clear();
        }
    }
    
    auto file_read_end = std::chrono::high_resolution_clock::now();
    total_file_read_time += std::chrono::duration<double>(file_read_end - file_read_start).count();
    
    // Process remaining lines
    if (!batch_lines.empty()) {
        process_signature_batch(batch_lines, batch_line_lengths, outfile);
    }
    
    infile.close();
    outfile.close();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::cout << "Signature generation completed in " << total_time << " seconds" << std::endl;
    }
}

// Process a batch of lines for signature generation (from lsh.cu)
void process_signature_batch(const std::vector<std::string>& lines, const std::vector<int>& line_lengths, std::ofstream& outfile) {
    auto batch_start = std::chrono::high_resolution_clock::now();
    
    int batch_size = lines.size();
    if (batch_size == 0) return;
    
    // Calculate total text size for batch
    int total_text_size = 0;
    for (int len : line_lengths) {
        total_text_size += len + 1; // +1 for null terminator
    }
    
    // Allocate host memory for text and signatures
    char* h_text = (char*)malloc(total_text_size);
    unsigned int* h_signatures = (unsigned int*)malloc(sizeof(unsigned int) * (num_hash + b) * batch_size);
    
    // Copy text to host buffer
    int text_offset = 0;
    std::vector<int> text_offsets(batch_size + 1);
    text_offsets[0] = 0;
    for (int i = 0; i < batch_size; i++) {
        int len = line_lengths[i];
        memcpy(h_text + text_offset, lines[i].c_str(), len);
        h_text[text_offset + len] = '\0';
        text_offset += len + 1;
        text_offsets[i + 1] = text_offset;
    }
    
    auto mem_transfer_start = std::chrono::high_resolution_clock::now();
    
    // Use pre-allocated GPU memory (from lsh.cu)
    int required_size = total_text_size;
    if (required_size > 3e9) {
        // Handle large batches by splitting
        std::cerr << "Batch size too large, splitting..." << std::endl;
        free(h_text);
        free(h_signatures);
        return;
    }
    
    // Copy data to device
    cudaMemcpy(buf_cuda, h_text, total_text_size, cudaMemcpyHostToDevice);
    cudaMemcpy(bias_cuda, text_offsets.data(), sizeof(int) * (batch_size + 1), cudaMemcpyHostToDevice);
    
    auto computation_start = std::chrono::high_resolution_clock::now();
    
    // Launch hash string kernel (from lsh.cu)
    dim3 hash_grid(batch_size);
    dim3 hash_block(num_hash);
    hash_string_kernel_psi<<<hash_grid, hash_block>>>(buf_cuda, bias_cuda, _p, _q, _r, batch_size, len_shingle, num_hash, b, hash_result_cuda);
    
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    
    // Launch generate key kernel (from lsh.cu)
    dim3 key_grid(batch_size);
    dim3 key_block(b);
    generate_key_kernel_psi<<<key_grid, key_block>>>(hash_result_cuda, batch_size, num_hash, b, num_key);
    
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    
    auto computation_end = std::chrono::high_resolution_clock::now();
    total_computation_time += std::chrono::duration<double>(computation_end - computation_start).count();
    cmp_time3 += std::chrono::duration<double>(computation_end - computation_start).count();
    
    // Copy results back to host
    cudaMemcpy(h_signatures, hash_result_cuda, sizeof(unsigned int) * (num_hash + b) * batch_size, cudaMemcpyDeviceToHost);
    
    auto mem_transfer_end = std::chrono::high_resolution_clock::now();
    total_memory_transfer_time += std::chrono::duration<double>(mem_transfer_end - mem_transfer_start).count();
    
    // Write signatures to file
    for (int i = 0; i < batch_size; i++) {
        int line_len = line_lengths[i];
        outfile.write(reinterpret_cast<const char*>(&line_len), sizeof(int));
        outfile.write(lines[i].c_str(), line_len);
        outfile.write(reinterpret_cast<const char*>(&h_signatures[i * (num_hash + b)]), sizeof(unsigned int) * num_hash);
    }
    
    // Clean up
    free(h_text);
    free(h_signatures);
    
    auto batch_end = std::chrono::high_resolution_clock::now();
    double batch_time = std::chrono::duration<double>(batch_end - batch_start).count();
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0 && batch_size >= 1024) {
        std::cout << "Processed batch of " << batch_size << " lines in " << batch_time << " seconds" << std::endl;
    }
}

// Generate MinHash signatures for a batch of documents
__global__ void generate_minhash_batch(const char* text, int* text_offsets, int shingle_len, 
                                    unsigned int* p, unsigned int* q, unsigned int* r, 
                                    int num_hash, unsigned int* signatures) {
    int doc_id = blockIdx.x;
    int hash_id = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (hash_id < num_hash) {
        int text_start = text_offsets[doc_id];
        int text_end = text_offsets[doc_id + 1];
        int text_len = text_end - text_start - 1; // subtract null terminator
        
        const char* doc_text = text + text_start;
        unsigned int min_hash = UINT_MAX;
        
        if (text_len < shingle_len) {
            // For short texts, use the entire text as a single shingle
            unsigned int hash = 0;
            unsigned int p_val = p[hash_id];
            unsigned int q_val = q[hash_id];
            for (int i = 0; i < text_len; i++) {
                hash = (hash * p_val + doc_text[i]) % q_val;
            }
            min_hash = hash;
        } else {
            // Use rolling hash technique for efficiency
            unsigned int sum = 0;
            unsigned int p_val = p[hash_id];
            unsigned int q_val = q[hash_id];
            unsigned int r_val = r[hash_id];
            
            // Compute hash for the initial window
            for (int i = 0; i < shingle_len; i++) {
                sum = (sum * p_val + doc_text[i]) % q_val;
            }
            min_hash = sum;
            
            // Compute hash for rolling window
            for (int i = shingle_len; i < text_len; i++) {
                // Rolling hash formula: remove the leftmost character and add the new character
                sum = (sum * p_val + ((unsigned int)doc_text[i - shingle_len]) * r_val + doc_text[i]) % q_val;
                if (sum < min_hash) {
                    min_hash = sum;
                }
            }
        }
        
        signatures[doc_id * num_hash + hash_id] = min_hash;
    }
}

// Generate LSH signatures for Party A's dataset
void generate_signatures_party_a(const std::string& input_file, std::string& output_signatures) {
    generate_signatures(input_file, output_signatures);
}

// Generate LSH signatures for Party B's dataset
void generate_signatures_party_b(const std::string& input_file, std::string& output_signatures) {
    generate_signatures(input_file, output_signatures);
}

// Party A sends buckets to Party B with optimized bucket computation
void send_buckets_party_a(const std::string& signatures_file, std::string& buckets_file) {
    std::ifstream infile(signatures_file, std::ios::binary);
    std::ofstream outfile(buckets_file, std::ios::binary);
    
    // Batch processing for bucket computation
    const int BATCH_SIZE = 1024;
    std::vector<std::vector<unsigned int>> batch_signatures;
    
    while (true) {
        int line_len;
        if (!infile.read(reinterpret_cast<char*>(&line_len), sizeof(int))) break;
        
        std::string line;
        line.resize(line_len);
        infile.read(&line[0], line_len);
        
        std::vector<unsigned int> signature(num_hash);
        infile.read(reinterpret_cast<char*>(signature.data()), sizeof(unsigned int) * num_hash);
        batch_signatures.push_back(signature);
        
        // Process batch when it's full
        if (batch_signatures.size() >= BATCH_SIZE) {
            process_buckets_batch(batch_signatures, outfile);
            batch_signatures.clear();
        }
    }
    
    // Process remaining signatures
    if (!batch_signatures.empty()) {
        process_buckets_batch(batch_signatures, outfile);
    }
    
    infile.close();
    outfile.close();
}

// Process a batch of signatures for bucket computation (from lsh.cu)
void process_buckets_batch(const std::vector<std::vector<unsigned int>>& signatures, std::ofstream& outfile) {
    int batch_size = signatures.size();
    if (batch_size == 0) return;
    
    // Allocate host memory for signatures and buckets
    unsigned int* h_signatures = (unsigned int*)malloc(sizeof(unsigned int) * (num_hash + b) * batch_size);
    
    // Copy signatures to host buffer
    for (int i = 0; i < batch_size; i++) {
        memcpy(&h_signatures[i * (num_hash + b)], signatures[i].data(), sizeof(unsigned int) * num_hash);
    }
    
    auto mem_transfer_start = std::chrono::high_resolution_clock::now();
    
    // Copy signatures to GPU
    cudaMemcpy(hash_result_cuda, h_signatures, sizeof(unsigned int) * (num_hash + b) * batch_size, cudaMemcpyHostToDevice);
    
    auto computation_start = std::chrono::high_resolution_clock::now();
    
    // Launch generate key kernel (from lsh.cu)
    dim3 key_grid(batch_size);
    dim3 key_block(b);
    generate_key_kernel_psi<<<key_grid, key_block>>>(hash_result_cuda, batch_size, num_hash, b, num_key);
    
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    
    auto computation_end = std::chrono::high_resolution_clock::now();
    total_computation_time += std::chrono::duration<double>(computation_end - computation_start).count();
    
    // Copy buckets back to host
    cudaMemcpy(h_signatures, hash_result_cuda, sizeof(unsigned int) * (num_hash + b) * batch_size, cudaMemcpyDeviceToHost);
    
    auto mem_transfer_end = std::chrono::high_resolution_clock::now();
    total_memory_transfer_time += std::chrono::duration<double>(mem_transfer_end - mem_transfer_start).count();
    
    // Write buckets to file
    for (int i = 0; i < batch_size; i++) {
        outfile.write(reinterpret_cast<const char*>(&h_signatures[i * (num_hash + b) + num_hash]), sizeof(unsigned int) * b);
    }
    
    // Clean up
    free(h_signatures);
}

// Party B receives buckets from Party A and finds candidate pairs with optimization (from lsh.cu)
void process_buckets_party_b(const std::string& signatures_file, const std::string& buckets_file, std::string& candidates_file) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::ifstream sig_file(signatures_file, std::ios::binary);
    std::ifstream buck_file(buckets_file, std::ios::binary);
    std::ofstream outfile(candidates_file, std::ios::binary);
    
    // Read Party B's signatures
    std::vector<std::vector<unsigned int>> b_signatures;
    while (true) {
        int line_len;
        if (!sig_file.read(reinterpret_cast<char*>(&line_len), sizeof(int))) break;
        
        std::string line; line.resize(line_len);
        sig_file.read(&line[0], line_len);
        
        std::vector<unsigned int> signature(num_hash);
        sig_file.read(reinterpret_cast<char*>(signature.data()), sizeof(unsigned int) * num_hash);
        b_signatures.push_back(signature);
    }
    
    int b_size = b_signatures.size();
    
    // Precompute Party B's buckets using GPU (from lsh.cu)
    std::vector<std::vector<unsigned int>> b_buckets_all(b_size, std::vector<unsigned int>(b));
    
    // Batch processing for Party B's signatures
    const int BATCH_SIZE = 1024;
    for (int batch_start = 0; batch_start < b_size; batch_start += BATCH_SIZE) {
        int batch_end = std::min(batch_start + BATCH_SIZE, b_size);
        int current_batch_size = batch_end - batch_start;
        
        // Allocate host memory for batch signatures
        unsigned int* h_signatures = (unsigned int*)malloc(sizeof(unsigned int) * (num_hash + b) * current_batch_size);
        
        // Copy signatures to host buffer
        for (int i = 0; i < current_batch_size; i++) {
            int global_idx = batch_start + i;
            memcpy(&h_signatures[i * (num_hash + b)], b_signatures[global_idx].data(), sizeof(unsigned int) * num_hash);
        }
        
        // Copy to GPU and compute buckets
        cudaMemcpy(hash_result_cuda, h_signatures, sizeof(unsigned int) * (num_hash + b) * current_batch_size, cudaMemcpyHostToDevice);
        
        // Launch generate key kernel
        dim3 key_grid(current_batch_size);
        dim3 key_block(b);
        generate_key_kernel_psi<<<key_grid, key_block>>>(hash_result_cuda, current_batch_size, num_hash, b, num_key);
        
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());
        
        // Copy buckets back to host
        cudaMemcpy(h_signatures, hash_result_cuda, sizeof(unsigned int) * (num_hash + b) * current_batch_size, cudaMemcpyDeviceToHost);
        
        // Copy buckets to result vector
        for (int i = 0; i < current_batch_size; i++) {
            int global_idx = batch_start + i;
            memcpy(b_buckets_all[global_idx].data(), &h_signatures[i * (num_hash + b) + num_hash], sizeof(unsigned int) * b);
        }
        
        free(h_signatures);
    }
    
    // Batch processing for Party A's buckets
    std::vector<std::vector<unsigned int>> batch_a_buckets;
    std::vector<int> batch_a_indices;
    int a_idx = 0;
    
    while (true) {
        std::vector<unsigned int> a_buckets(b);
        if (!buck_file.read(reinterpret_cast<char*>(a_buckets.data()), sizeof(unsigned int) * b)) break;
        
        batch_a_buckets.push_back(a_buckets);
        batch_a_indices.push_back(a_idx);
        a_idx++;
        
        // Process batch when it's full
        if (batch_a_buckets.size() >= BATCH_SIZE) {
            process_candidates_batch(batch_a_buckets, batch_a_indices, b_buckets_all, outfile);
            batch_a_buckets.clear();
            batch_a_indices.clear();
        }
    }
    
    // Process remaining buckets
    if (!batch_a_buckets.empty()) {
        process_candidates_batch(batch_a_buckets, batch_a_indices, b_buckets_all, outfile);
    }
    
    sig_file.close();
    buck_file.close();
    outfile.close();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::cout << "Party B bucket processing completed in " << total_time << " seconds" << std::endl;
    }
}

// Process a batch of Party A's buckets to find candidate pairs with GPU optimization (from lsh.cu)
void process_candidates_batch(const std::vector<std::vector<unsigned int>>& a_buckets_batch, 
                             const std::vector<int>& a_indices, 
                             const std::vector<std::vector<unsigned int>>& b_buckets_all, 
                             std::ofstream& outfile) {
    int batch_size = a_buckets_batch.size();
    int b_size = b_buckets_all.size();
    
    // Use a buffer to collect candidates
    std::vector<std::pair<int, int>> candidates;
    
    // Process in parallel with optimized scheduling and reduction
    #pragma omp parallel
    {
        std::vector<std::pair<int, int>> local_candidates;
        
        #pragma omp for schedule(dynamic, 64) // Dynamic scheduling for better load balancing
        for (int i = 0; i < batch_size; i++) {
            const auto& a_buckets = a_buckets_batch[i];
            int a_idx = a_indices[i];
            
            for (int b_idx = 0; b_idx < b_size; b_idx++) {
                const auto& b_buckets = b_buckets_all[b_idx];
                
                // Check if any bucket matches
                bool has_match = false;
                for (int band = 0; band < b; band++) {
                    if (a_buckets[band] == b_buckets[band]) {
                        has_match = true;
                        break;
                    }
                }
                
                if (has_match) {
                    local_candidates.emplace_back(a_idx, b_idx);
                }
            }
        }
        
        // Combine results from all threads
        #pragma omp critical
        {
            candidates.insert(candidates.end(), local_candidates.begin(), local_candidates.end());
        }
    }
    
    // Write candidates to file
    for (const auto& candidate : candidates) {
        outfile.write(reinterpret_cast<const char*>(&candidate.first), sizeof(int));
        outfile.write(reinterpret_cast<const char*>(&candidate.second), sizeof(int));
    }
}

// Optimized GPU version for signature comparison using shared memory tiling (from lsh.cu)
void verify_candidates_gpu_optimized(const std::vector<std::vector<unsigned int>>& a_signatures,
                                     const std::vector<std::vector<unsigned int>>& b_signatures,
                                     const std::vector<std::pair<int, int>>& candidates,
                                     std::vector<std::pair<int, int>>& verified_pairs,
                                     std::vector<double>& similarities) {
    if (candidates.empty()) return;
    
    int th_int = (int)(threshold * num_hash);
    
    // Process candidates in batches for GPU
    const int GPU_BATCH_SIZE = 1024;
    for (int batch_start = 0; batch_start < candidates.size(); batch_start += GPU_BATCH_SIZE) {
        int batch_end = std::min(batch_start + GPU_BATCH_SIZE, (int)candidates.size());
        int current_batch_size = batch_end - batch_start;
        
        // Allocate host memory
        unsigned int* h_a_sigs = (unsigned int*)malloc(sizeof(unsigned int) * num_hash * current_batch_size);
        unsigned int* h_b_sigs = (unsigned int*)malloc(sizeof(unsigned int) * num_hash * current_batch_size);
        
        // Prepare signature data
        for (int i = 0; i < current_batch_size; i++) {
            int global_idx = batch_start + i;
            int a_idx = candidates[global_idx].first;
            int b_idx = candidates[global_idx].second;
            
            memcpy(&h_a_sigs[i * num_hash], a_signatures[a_idx].data(), sizeof(unsigned int) * num_hash);
            memcpy(&h_b_sigs[i * num_hash], b_signatures[b_idx].data(), sizeof(unsigned int) * num_hash);
        }
        
        // Allocate GPU memory
        unsigned int* d_a_sigs;
        unsigned int* d_b_sigs;
        unsigned char* d_results;
        
        cudaMalloc(&d_a_sigs, sizeof(unsigned int) * num_hash * current_batch_size);
        cudaMalloc(&d_b_sigs, sizeof(unsigned int) * num_hash * current_batch_size);
        cudaMalloc(&d_results, sizeof(unsigned char) * current_batch_size * current_batch_size);
        
        // Copy to GPU
        cudaMemcpy(d_a_sigs, h_a_sigs, sizeof(unsigned int) * num_hash * current_batch_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b_sigs, h_b_sigs, sizeof(unsigned int) * num_hash * current_batch_size, cudaMemcpyHostToDevice);
        cudaMemset(d_results, 0, sizeof(unsigned char) * current_batch_size * current_batch_size);
        
        // Launch optimized comparison kernel with shared memory
        const int block = BLOCK_SIZE;
        dim3 numBlocks((current_batch_size + block - 1) / block, (current_batch_size + block - 1) / block);
        dim3 blockSize(THREAD_NUM);
        
        // Use pre-allocated buffer if size permits
        if (current_batch_size <= max_bucket) {
            gpuErrchk(cudaMemcpy(cuda_buf1, h_a_sigs, sizeof(unsigned int) * num_hash * current_batch_size, cudaMemcpyHostToDevice));
            compare_lsh_kernel<<<numBlocks, blockSize>>>(cuda_buf1, cuda_cmp_result, current_batch_size, num_hash, th_int);
            cudaDeviceSynchronize();
            gpuErrchk(cudaGetLastError());
            
            // Perform reduction to count matches
            reduce_compare_result1<<<REDUCE_BLOCK, REDUCE_THREAD>>>(cuda_cmp_result, cuda_reduce_buf, current_batch_size);
            reduce_compare_result2<<<1, REDUCE_BLOCK>>>(cuda_reduce_buf, cuda_reduce_cnt);
            cudaDeviceSynchronize();
            gpuErrchk(cudaGetLastError());
            
            int reduce_cnt;
            gpuErrchk(cudaMemcpy(&reduce_cnt, cuda_reduce_cnt, sizeof(int), cudaMemcpyDeviceToHost));
            
            if (reduce_cnt > 0 && reduce_cnt <= max_bucket * REDUCE_EDGE_TH) {
                // Extract detailed results using optimized reduction
                reduce_compare_result3<<<REDUCE_BLOCK, REDUCE_THREAD>>>(cuda_cmp_result, cuda_reduce_buf, cuda_reduce_result, current_batch_size);
                gpuErrchk(cudaMemcpy(reduce_result, cuda_reduce_result, sizeof(int) * reduce_cnt * 2, cudaMemcpyDeviceToHost));
                
                for (int id = 0; id < reduce_cnt; id++) {
                    int i = reduce_result[id * 2];
                    int j = reduce_result[id * 2 + 1];
                    if (i == j) continue;
                    
                    int global_idx = batch_start + i;
                    int a_idx = candidates[global_idx].first;
                    int b_idx = candidates[global_idx].second;
                    
                    verified_pairs.emplace_back(a_idx, b_idx);
                    similarities.push_back(1.0); // Similar pair
                }
            }
        } else {
            // Fallback to simple kernel for large batches
            int* d_matches;
            cudaMalloc(&d_matches, sizeof(int) * current_batch_size);
            
            dim3 grid(current_batch_size);
            dim3 blk(256);
            compute_matches_kernel<<<grid, blk>>>(d_a_sigs, d_b_sigs, d_matches, current_batch_size, num_hash);
            cudaDeviceSynchronize();
            gpuErrchk(cudaGetLastError());
            
            int* h_matches = (int*)malloc(sizeof(int) * current_batch_size);
            cudaMemcpy(h_matches, d_matches, sizeof(int) * current_batch_size, cudaMemcpyDeviceToHost);
            
            for (int i = 0; i < current_batch_size; i++) {
                int global_idx = batch_start + i;
                int a_idx = candidates[global_idx].first;
                int b_idx = candidates[global_idx].second;
                
                double similarity = static_cast<double>(h_matches[i]) / num_hash;
                if (similarity >= threshold) {
                    verified_pairs.emplace_back(a_idx, b_idx);
                    similarities.push_back(similarity);
                }
            }
            
            free(h_matches);
            cudaFree(d_matches);
        }
        
        // Cleanup
        free(h_a_sigs);
        free(h_b_sigs);
        cudaFree(d_a_sigs);
        cudaFree(d_b_sigs);
        cudaFree(d_results);
    }
}

// Optimized version of verify_candidates_party_a using GPU acceleration
void verify_candidates_party_a_optimized(const std::string& a_signatures_file, const std::string& b_signatures_file, 
                                       const std::string& indices_file, std::string& results_file) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Read Party A's signatures
    std::ifstream a_sig_file(a_signatures_file, std::ios::binary);
    std::vector<std::pair<std::string, std::vector<unsigned int>>> a_signatures;
    
    while (true) {
        int line_len;
        if (!a_sig_file.read(reinterpret_cast<char*>(&line_len), sizeof(int))) break;
        
        std::string line; line.resize(line_len);
        a_sig_file.read(&line[0], line_len);
        
        std::vector<unsigned int> signature(num_hash);
        a_sig_file.read(reinterpret_cast<char*>(signature.data()), sizeof(unsigned int) * num_hash);
        a_signatures.emplace_back(line, signature);
    }
    a_sig_file.close();
    
    // Read Party B's signatures
    std::ifstream b_sig_file(b_signatures_file, std::ios::binary);
    std::vector<std::pair<std::string, std::vector<unsigned int>>> b_signatures;
    
    while (true) {
        int line_len;
        if (!b_sig_file.read(reinterpret_cast<char*>(&line_len), sizeof(int))) break;
        
        std::string line; line.resize(line_len);
        b_sig_file.read(&line[0], line_len);
        
        std::vector<unsigned int> signature(num_hash);
        b_sig_file.read(reinterpret_cast<char*>(signature.data()), sizeof(unsigned int) * num_hash);
        b_signatures.emplace_back(line, signature);
    }
    b_sig_file.close();
    
    // Read candidates
    std::ifstream cand_file(indices_file, std::ios::binary);
    std::vector<std::pair<int, int>> candidates;
    
    while (true) {
        int a_idx, b_idx;
        if (!cand_file.read(reinterpret_cast<char*>(&a_idx), sizeof(int))) break;
        if (!cand_file.read(reinterpret_cast<char*>(&b_idx), sizeof(int))) break;
        candidates.emplace_back(a_idx, b_idx);
    }
    cand_file.close();
    
    // Verify candidates and compute similarity with GPU acceleration
    std::ofstream outfile(results_file);
    int valid_pairs = 0;
    
    // Batch processing for GPU acceleration
    const int BATCH_SIZE = 1024;
    for (int batch_start = 0; batch_start < candidates.size(); batch_start += BATCH_SIZE) {
        int batch_end = std::min(batch_start + BATCH_SIZE, (int)candidates.size());
        int current_batch_size = batch_end - batch_start;
        
        if (current_batch_size == 0) continue;
        
        // Allocate host memory for batch processing
        unsigned int* h_a_signatures = (unsigned int*)malloc(sizeof(unsigned int) * num_hash * current_batch_size);
        unsigned int* h_b_signatures = (unsigned int*)malloc(sizeof(unsigned int) * num_hash * current_batch_size);
        int* h_matches = (int*)malloc(sizeof(int) * current_batch_size);
        
        // Copy signatures to host buffer
        for (int i = 0; i < current_batch_size; i++) {
            int global_idx = batch_start + i;
            int a_idx = candidates[global_idx].first;
            int b_idx = candidates[global_idx].second;
            
            if (a_idx < 0 || a_idx >= a_signatures.size() || b_idx < 0 || b_idx >= b_signatures.size()) {
                h_matches[i] = 0;
                continue;
            }
            
            memcpy(&h_a_signatures[i * num_hash], a_signatures[a_idx].second.data(), sizeof(unsigned int) * num_hash);
            memcpy(&h_b_signatures[i * num_hash], b_signatures[b_idx].second.data(), sizeof(unsigned int) * num_hash);
        }
        
        // Allocate GPU memory
        unsigned int* d_a_signatures;
        unsigned int* d_b_signatures;
        int* d_matches;
        
        cudaMalloc(&d_a_signatures, sizeof(unsigned int) * num_hash * current_batch_size);
        cudaMalloc(&d_b_signatures, sizeof(unsigned int) * num_hash * current_batch_size);
        cudaMalloc(&d_matches, sizeof(int) * current_batch_size);
        
        // Copy data to GPU
        cudaMemcpy(d_a_signatures, h_a_signatures, sizeof(unsigned int) * num_hash * current_batch_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b_signatures, h_b_signatures, sizeof(unsigned int) * num_hash * current_batch_size, cudaMemcpyHostToDevice);
        
        // Launch kernel to compute matches
        dim3 grid(current_batch_size);
        dim3 block(256);
        compute_matches_kernel<<<grid, block>>>(d_a_signatures, d_b_signatures, d_matches, current_batch_size, num_hash);
        
        cudaDeviceSynchronize();
        gpuErrchk(cudaGetLastError());
        
        // Copy results back to host
        cudaMemcpy(h_matches, d_matches, sizeof(int) * current_batch_size, cudaMemcpyDeviceToHost);
        
        // Process results
        for (int i = 0; i < current_batch_size; i++) {
            int global_idx = batch_start + i;
            int a_idx = candidates[global_idx].first;
            int b_idx = candidates[global_idx].second;
            
            if (a_idx < 0 || a_idx >= a_signatures.size() || b_idx < 0 || b_idx >= b_signatures.size()) {
                continue;
            }
            
            double similarity = static_cast<double>(h_matches[i]) / num_hash;
            if (similarity >= threshold) {
                outfile << "Party A index: " << a_idx << ", Party B index: " << b_idx << "\n";
                outfile << "Party A text: " << a_signatures[a_idx].first << "\n";
                outfile << "Party B text: " << b_signatures[b_idx].first << "\n";
                outfile << "Similarity: " << similarity << "\n";
                outfile << "Similarity threshold: " << threshold << "\n";
                outfile << "---\n";
                valid_pairs++;
            }
        }
        
        // Clean up
        free(h_a_signatures);
        free(h_b_signatures);
        free(h_matches);
        cudaFree(d_a_signatures);
        cudaFree(d_b_signatures);
        cudaFree(d_matches);
    }
    
    outfile << "Total candidate pairs found: " << valid_pairs << std::endl;
    outfile.close();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    total_verification_time += total_time;
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::cout << "Verification completed. Found " << valid_pairs << " candidate pairs." << std::endl;
    }
}

// CUDA kernel to compute matches between signatures
__global__ void compute_matches_kernel(unsigned int* a_signatures, unsigned int* b_signatures, int* matches, int batch_size, int num_hash) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    int match_count = 0;
    for (int i = 0; i < num_hash; i++) {
        if (a_signatures[idx * num_hash + i] == b_signatures[idx * num_hash + i]) {
            match_count++;
        }
    }
    matches[idx] = match_count;
}

// Kernel function to compare hash values and determine similarity (from lsh.cu)
__global__ void compare_lsh_kernel(unsigned int *buf, unsigned char* result, int line_num, int num_hash, int th) {
    int block_id1 = blockIdx.x;
    int block_id2 = blockIdx.y;
    if(block_id2 < block_id1) return; 

    __shared__ int t1[BLOCK_SIZE*TILE_SIZE];
    __shared__ int t2[BLOCK_SIZE*TILE_SIZE];

    int x=threadIdx.x;
    int dx = x%BLOCK_SIZE;
    int dy = x/BLOCK_SIZE;

    int bias1=(BLOCK_SIZE*block_id1), lim1 = min(BLOCK_SIZE*(block_id1+1), line_num);
    int bias2=(BLOCK_SIZE*block_id2), lim2 = min(BLOCK_SIZE*(block_id2+1), line_num);
    
    unsigned int *buf1 = &buf[(long long)bias1 * num_hash];
    unsigned int *buf2 = &buf[(long long)bias2 * num_hash];

    unsigned int cnt=0;

    for(int tile=0; tile<num_hash; tile+=TILE_SIZE) {
        if(bias1+dy<lim1) t1[dy*TILE_SIZE+dx] = buf1[dy * num_hash + tile + dx];
        if(bias2+dy<lim2) t2[dy*TILE_SIZE+dx] = buf2[dy * num_hash + tile + dx];
        __syncthreads();

        for(int k = 0; k < TILE_SIZE; k += 4) {
            cnt += (t1[dx*TILE_SIZE + k] == t2[dy*TILE_SIZE + k]);
            cnt += (t1[dx*TILE_SIZE + k + 1] == t2[dy*TILE_SIZE + k + 1]);
            cnt += (t1[dx*TILE_SIZE + k + 2] == t2[dy*TILE_SIZE + k + 2]);
            cnt += (t1[dx*TILE_SIZE + k + 3] == t2[dy*TILE_SIZE + k + 3]);
        }

        __syncthreads();
    }
    if((bias1+dx)<lim1 && (bias2+dy) <lim2 && (bias1+dx)<(bias2+dy)) result[(size_t)(bias1+dx)*line_num+(bias2+dy)]=cnt>th;
}

// Kernel to reduce the comparison results and count the total number of set bits (from lsh.cu)
__global__ void reduce_compare_result1(unsigned char* result, int *cuda_reduce_buf, int line_num) {
    __shared__ int prefix_sum[REDUCE_THREAD];
    int x = threadIdx.x;
    int b = blockDim.x;
    int y = blockIdx.x;
    int cnt = 0;

    size_t total = (size_t)line_num * line_num / 4 * 4;

    size_t l = (total + REDUCE_BLOCK - 1) / REDUCE_BLOCK * y / 4 * 4;
    size_t r = (total + REDUCE_BLOCK - 1) / REDUCE_BLOCK * (y + 1) / 4 * 4;
    if (r > total) r = total;

    unsigned int *result_m = (unsigned int *)result;

    for (size_t i = l + x * 4; i < r; i += b * 4) {
        unsigned int val = result_m[i / 4];
        cnt += __popc(val);
    }
    prefix_sum[x] = cnt;
    __syncthreads();

    for (int k = 1; k < REDUCE_THREAD; k = k + k) {
        if (x >= k) prefix_sum[x] += prefix_sum[x - k];
        __syncthreads();
    }

    if (x == 0) cuda_reduce_buf[y] = prefix_sum[REDUCE_THREAD - 1];
}

// Kernel to compute the total count of set bits across all blocks (from lsh.cu)
__global__ void reduce_compare_result2(int* cuda_reduce_buf, int *total) {
    __shared__ int prefix_sum[REDUCE_BLOCK];
    int x = threadIdx.x;

    prefix_sum[x] = cuda_reduce_buf[x];
    __syncthreads();

    for (int k = 1; k < REDUCE_BLOCK; k = k + k) {
        if (x >= k) prefix_sum[x] += prefix_sum[x - k];
        __syncthreads();
    }

    if (x == 0) {
        *total = prefix_sum[REDUCE_BLOCK - 1];
    }
    cuda_reduce_buf[x] = prefix_sum[x];
}

// Kernel to extract detailed comparison results and store matched pairs in the output (from lsh.cu)
__global__ void reduce_compare_result3(unsigned char* result, int *cuda_reduce_buf, int *output, int line_num) {
    __shared__ int prefix_sum[REDUCE_THREAD];
    int x = threadIdx.x;
    int b = blockDim.x;
    int y = blockIdx.x;
    int cnt = 0;

    size_t total = (size_t)line_num * line_num / 4 * 4;

    size_t l = (total + REDUCE_BLOCK - 1) / REDUCE_BLOCK * y / 4 * 4;
    size_t r = (total + REDUCE_BLOCK - 1) / REDUCE_BLOCK * (y + 1) / 4 * 4;
    if (r > total) r = total;

    unsigned int *result_m = (unsigned int *)result;

    for (size_t i = l + x * 4; i < r; i += b * 4) {
        unsigned int val = result_m[i / 4];
        cnt += __popc(val);
    }

    prefix_sum[x] = cnt;
    __syncthreads();

    for (int k = 1; k < REDUCE_THREAD; k = k + k) {
        if (x >= k) prefix_sum[x] += prefix_sum[x - k];
        __syncthreads();
    }

    cnt = prefix_sum[x] + cuda_reduce_buf[y] - prefix_sum[REDUCE_THREAD - 1];

    for (size_t i = l + x * 4; i < r; i += b * 4) {
        unsigned int val = result_m[i / 4];
        
        if (val & 1) {
            cnt--;
            output[cnt * 2] = (int)(i % line_num);
            output[cnt * 2 + 1] = (int)(i / line_num);
        }
        if (val & 0x100) {
            cnt--;
            output[cnt * 2] = (int)((i + 1) % line_num);
            output[cnt * 2 + 1] = (int)((i + 1) / line_num);
        }
        if (val & 0x10000) {
            cnt--;
            output[cnt * 2] = (int)((i + 2) % line_num);
            output[cnt * 2 + 1] = (int)((i + 2) / line_num);
        }
        if (val & 0x1000000) {
            cnt--;
            output[cnt * 2] = (int)((i + 3) % line_num);
            output[cnt * 2 + 1] = (int)((i + 3) / line_num);
        }
    }
}

// Party B sends candidate indices to Party A
void send_candidates_party_b(const std::string& candidates_file, std::string& indices_file) {
    // Simply copy the candidates file
    std::ifstream infile(candidates_file, std::ios::binary);
    std::ofstream outfile(indices_file, std::ios::binary);
    outfile << infile.rdbuf();
    infile.close();
    outfile.close();
}

// Compute Jaccard similarity between two signatures
double compute_similarity(const std::vector<unsigned int>& sig1, const std::vector<unsigned int>& sig2) {
    int matches = 0;
    for (int i = 0; i < sig1.size(); i++) {
        if (sig1[i] == sig2[i]) {
            matches++;
        }
    }
    return static_cast<double>(matches) / sig1.size();
}

// Party A verifies candidates and computes final results (using optimized GPU version)
void verify_candidates_party_a(const std::string& a_signatures_file, const std::string& b_signatures_file, const std::string& indices_file, std::string& results_file) {
    // Use the optimized GPU version
    verify_candidates_party_a_optimized(a_signatures_file, b_signatures_file, indices_file, results_file);
}

// Run the complete PSI protocol
void run_psi(const std::string& party_a_file, const std::string& party_b_file, const std::string& output_dir, int role) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Create output directory if it doesn't exist
    fs::create_directories(output_dir);
    
    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);
    
    // First run Party A to generate signatures and buckets
    if (role == 0 || role == 1) {
        std::string a_signatures = output_dir + "/a_signatures.bin";
        std::string buckets = output_dir + "/buckets.bin";
        
        // Party A: Generate signatures and buckets
        if (rank == 0) {
            std::cout << "Party A: Generating signatures..." << std::endl;
        }
        generate_signatures_party_a(party_a_file, a_signatures);
        
        auto comm_start = std::chrono::high_resolution_clock::now();
        
        if (rank == 0) {
            std::cout << "Party A: Sending buckets to Party B..." << std::endl;
        }
        send_buckets_party_a(a_signatures, buckets);
        
        auto comm_end = std::chrono::high_resolution_clock::now();
        total_communication_time += std::chrono::duration<double>(comm_end - comm_start).count();
    }
    
    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Then run Party B to process buckets and generate candidates
    if (role == 0 || role == 1) {
        std::string b_signatures = output_dir + "/b_signatures.bin";
        std::string buckets = output_dir + "/buckets.bin";
        std::string candidates = output_dir + "/candidates.bin";
        std::string indices = output_dir + "/candidate_indices.bin";
        
        // Party B: Generate signatures and process buckets
        if (rank == 0) {
            std::cout << "Party B: Generating signatures..." << std::endl;
        }
        generate_signatures_party_b(party_b_file, b_signatures);
        
        if (rank == 0) {
            std::cout << "Party B: Processing buckets and finding candidates..." << std::endl;
        }
        process_buckets_party_b(b_signatures, buckets, candidates);
        
        auto comm_start = std::chrono::high_resolution_clock::now();
        
        if (rank == 0) {
            std::cout << "Party B: Sending candidate indices to Party A..." << std::endl;
        }
        send_candidates_party_b(candidates, indices);
        
        auto comm_end = std::chrono::high_resolution_clock::now();
        total_communication_time += std::chrono::duration<double>(comm_end - comm_start).count();
        
        if (rank == 0) {
            std::cout << "Party B: PSI completed." << std::endl;
        }
    }
    
    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Finally run Party A to verify candidates
    if (role == 0 || role == 1) {
        std::string a_signatures = output_dir + "/a_signatures.bin";
        std::string b_signatures = output_dir + "/b_signatures.bin";
        std::string indices = output_dir + "/candidate_indices.bin";
        std::string results = output_dir + "/psi_results.txt";
        
        auto verification_start = std::chrono::high_resolution_clock::now();
        
        if (rank == 0) {
            std::cout << "Party A: Verifying candidates..." << std::endl;
        }
        verify_candidates_party_a(a_signatures, b_signatures, indices, results);
        
        auto verification_end = std::chrono::high_resolution_clock::now();
        total_verification_time += std::chrono::duration<double>(verification_end - verification_start).count();
        
        if (rank == 0) {
            std::cout << "Party A: PSI completed. Results saved to " << results << std::endl;
        }
    }
    
    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();
    
    if (rank == 0) {
        std::cout << "\nTotal PSI execution time: " << total_time << " seconds" << std::endl;
    }
}

// Print performance statistics
void print_performance_stats() {
    std::cout << "\nPSI Performance Statistics:" << std::endl;
    std::cout << "  - File read time: " << total_file_read_time << " seconds" << std::endl;
    std::cout << "  - Computation time: " << total_computation_time << " seconds" << std::endl;
    std::cout << "  - Memory transfer time: " << total_memory_transfer_time << " seconds" << std::endl;
    std::cout << "  - Communication time: " << total_communication_time << " seconds" << std::endl;
    std::cout << "  - Verification time: " << total_verification_time << " seconds" << std::endl;
    std::cout << "  - Total time: " << total_file_read_time + total_computation_time + total_memory_transfer_time + total_communication_time + total_verification_time << " seconds" << std::endl;
}

// Clean up resources
void finalize_psi() {
    free(p);
    free(q);
    free(r);
    cudaFree(_p);
    cudaFree(_q);
    cudaFree(_r);
    
    // Free GPU memory
    if (hash_result_cuda) cudaFree(hash_result_cuda);
    if (bias_cuda) cudaFree(bias_cuda);
    if (buf_cuda) cudaFree(buf_cuda);
    
    // Print performance statistics
    print_performance_stats();
}
