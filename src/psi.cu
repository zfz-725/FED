#include <fstream>
#include <filesystem>
#include <iostream>
#include <random>
#include <chrono>
#include <vector>
#include <algorithm>
#include "mpi.h"
#include "param.h"
#include "util.h"
#include "psi.h"

namespace fs = std::filesystem;

// Parameters
static int num_hash = 128;          // Number of hash functions
static int len_shingle = 5;         // Length of each shingle
static int b = 16;                  // Number of bands
static double threshold = 0.8;       // Similarity threshold

// Hashing variables
static unsigned int *p, *q, *r;
static unsigned int *_p, *_q, *_r;

// Initialize PSI with LSH parameters
void init_psi(int _num_hash, int _shingle_len, int _bucket, double _threshold) {
    num_hash = _num_hash;
    len_shingle = _shingle_len;
    b = _bucket;
    threshold = _threshold;
    
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
    cudaMalloc((void**)&_p, sizeof(unsigned int) * num_hash);
    cudaMalloc((void**)&_q, sizeof(unsigned int) * num_hash);
    cudaMalloc((void**)&_r, sizeof(unsigned int) * num_hash);
    
    cudaMemcpy(_p, p, sizeof(unsigned int) * num_hash, cudaMemcpyHostToDevice);
    cudaMemcpy(_q, q, sizeof(unsigned int) * num_hash, cudaMemcpyHostToDevice);
    cudaMemcpy(_r, r, sizeof(unsigned int) * num_hash, cudaMemcpyHostToDevice);
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

// Compute hash for a shingle
__device__ unsigned int hash_shingle(const char* shingle, int len, unsigned int p, unsigned int q, unsigned int r) {
    unsigned int hash = 0;
    for (int i = 0; i < len; i++) {
        hash = (hash * p + shingle[i]) % q;
    }
    return hash;
}

// Generate MinHash signature for a document
__global__ void generate_minhash(const char* text, int text_len, int shingle_len, 
                               unsigned int* p, unsigned int* q, unsigned int* r, 
                               int num_hash, unsigned int* signature) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_hash) {
        unsigned int min_hash = UINT_MAX;
        if (text_len < shingle_len) {
            // For short texts, use the entire text as a single shingle
            unsigned int hash_val = hash_shingle(text, text_len, p[tid], q[tid], r[tid]);
            min_hash = hash_val;
        } else {
            for (int i = 0; i <= text_len - shingle_len; i++) {
                unsigned int hash_val = hash_shingle(&text[i], shingle_len, p[tid], q[tid], r[tid]);
                if (hash_val < min_hash) {
                    min_hash = hash_val;
                }
            }
        }
        signature[tid] = min_hash;
    }
}

// Generate LSH signatures for a dataset
void generate_signatures(const std::string& input_file, const std::string& output_signatures) {
    std::ifstream infile(input_file);
    std::ofstream outfile(output_signatures, std::ios::binary);
    std::string line;
    
    while (std::getline(infile, line)) {
        // Generate shingles
        auto shingles = generate_shingles(line);
        
        // Allocate memory for signature
        unsigned int* signature = (unsigned int*)malloc(sizeof(unsigned int) * num_hash);
        unsigned int* d_signature;
        cudaMalloc((void**)&d_signature, sizeof(unsigned int) * num_hash);
        
        // Copy text to device
        char* d_text;
        cudaMalloc((void**)&d_text, line.length() + 1);
        cudaMemcpy(d_text, line.c_str(), line.length() + 1, cudaMemcpyHostToDevice);
        
        // Launch kernel
        dim3 blocks((num_hash + 255) / 256);
        dim3 threads(256);
        generate_minhash<<<blocks, threads>>>(d_text, line.length(), len_shingle, 
                                           _p, _q, _r, num_hash, d_signature);
        
        // Copy result back
        cudaMemcpy(signature, d_signature, sizeof(unsigned int) * num_hash, cudaMemcpyDeviceToHost);
        
        // Write signature to file
        int line_len = line.length();
        outfile.write(reinterpret_cast<const char*>(&line_len), sizeof(int));
        outfile.write(line.c_str(), line_len);
        outfile.write(reinterpret_cast<const char*>(signature), sizeof(unsigned int) * num_hash);
        
        // Clean up
        free(signature);
        cudaFree(d_signature);
        cudaFree(d_text);
    }
    
    infile.close();
    outfile.close();
}

// Generate LSH signatures for Party A's dataset
void generate_signatures_party_a(const std::string& input_file, std::string& output_signatures) {
    generate_signatures(input_file, output_signatures);
}

// Generate LSH signatures for Party B's dataset
void generate_signatures_party_b(const std::string& input_file, std::string& output_signatures) {
    generate_signatures(input_file, output_signatures);
}

// Party A sends buckets to Party B
void send_buckets_party_a(const std::string& signatures_file, std::string& buckets_file) {
    std::ifstream infile(signatures_file, std::ios::binary);
    std::ofstream outfile(buckets_file, std::ios::binary);
    
    while (true) {
        int line_len;
        if (!infile.read(reinterpret_cast<char*>(&line_len), sizeof(int))) break;
        
        std::string line;
        line.resize(line_len);
        infile.read(&line[0], line_len);
        
        unsigned int signature[num_hash];
        infile.read(reinterpret_cast<char*>(signature), sizeof(unsigned int) * num_hash);
        
        // Compute bucket for each band
        int band_size = num_hash / b;
        for (int i = 0; i < b; i++) {
            unsigned int bucket_hash = 0;
            for (int j = 0; j < band_size; j++) {
                bucket_hash = bucket_hash * 31 + signature[i * band_size + j];
            }
            outfile.write(reinterpret_cast<const char*>(&bucket_hash), sizeof(unsigned int));
        }
    }
    
    infile.close();
    outfile.close();
}

// Party B receives buckets from Party A and finds candidate pairs
void process_buckets_party_b(const std::string& signatures_file, const std::string& buckets_file, std::string& candidates_file) {
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
    
    // Read Party A's buckets and find candidates
    int a_idx = 0;
    while (true) {
        std::vector<unsigned int> a_buckets(b);
        if (!buck_file.read(reinterpret_cast<char*>(a_buckets.data()), sizeof(unsigned int) * b)) break;
        
        // Check each of Party B's items
        for (int b_idx = 0; b_idx < b_signatures.size(); b_idx++) {
            // Compute buckets for Party B's item
            int band_size = num_hash / b;
            std::vector<unsigned int> b_buckets(b);
            for (int i = 0; i < b; i++) {
                unsigned int bucket_hash = 0;
                for (int j = 0; j < band_size; j++) {
                    bucket_hash = bucket_hash * 31 + b_signatures[b_idx][i * band_size + j];
                }
                b_buckets[i] = bucket_hash;
            }
            
            // Check if any bucket matches
            bool has_match = false;
            for (int i = 0; i < b; i++) {
                if (a_buckets[i] == b_buckets[i]) {
                    has_match = true;
                    break;
                }
            }
            
            if (has_match) {
                outfile.write(reinterpret_cast<const char*>(&a_idx), sizeof(int));
                outfile.write(reinterpret_cast<const char*>(&b_idx), sizeof(int));
            }
        }
        
        a_idx++;
    }
    
    sig_file.close();
    buck_file.close();
    outfile.close();
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

// Party A verifies candidates and computes final results
void verify_candidates_party_a(const std::string& a_signatures_file, const std::string& b_signatures_file, const std::string& indices_file, std::string& results_file) {
    std::ifstream a_sig_file(a_signatures_file, std::ios::binary);
    std::ifstream b_sig_file(b_signatures_file, std::ios::binary);
    std::ifstream idx_file(indices_file, std::ios::binary);
    std::ofstream outfile(results_file);
    
    if (!a_sig_file.is_open()) {
        std::cerr << "Failed to open Party A signatures file: " << a_signatures_file << std::endl;
        return;
    }
    
    if (!b_sig_file.is_open()) {
        std::cerr << "Failed to open Party B signatures file: " << b_signatures_file << std::endl;
        return;
    }
    
    if (!idx_file.is_open()) {
        std::cerr << "Failed to open indices file: " << indices_file << std::endl;
        return;
    }
    
    if (!outfile.is_open()) {
        std::cerr << "Failed to open results file: " << results_file << std::endl;
        return;
    }
    
    // Read all signatures from Party A
    std::vector<std::pair<std::string, std::vector<unsigned int>>> a_signatures;
    while (true) {
        int line_len;
        if (!a_sig_file.read(reinterpret_cast<char*>(&line_len), sizeof(int))) break;
        
        std::string line;
        line.resize(line_len);
        a_sig_file.read(&line[0], line_len);
        
        std::vector<unsigned int> signature(num_hash);
        a_sig_file.read(reinterpret_cast<char*>(signature.data()), sizeof(unsigned int) * num_hash);
        a_signatures.emplace_back(line, signature);
    }
    
    // Read all signatures from Party B
    std::vector<std::pair<std::string, std::vector<unsigned int>>> b_signatures;
    while (true) {
        int line_len;
        if (!b_sig_file.read(reinterpret_cast<char*>(&line_len), sizeof(int))) break;
        
        std::string line;
        line.resize(line_len);
        b_sig_file.read(&line[0], line_len);
        
        std::vector<unsigned int> signature(num_hash);
        b_sig_file.read(reinterpret_cast<char*>(signature.data()), sizeof(unsigned int) * num_hash);
        b_signatures.emplace_back(line, signature);
    }
    
    // Process candidate pairs
    int count = 0;
    while (true) {
        int a_idx, b_idx;
        if (!idx_file.read(reinterpret_cast<char*>(&a_idx), sizeof(int))) break;
        if (!idx_file.read(reinterpret_cast<char*>(&b_idx), sizeof(int))) break;
        
        // Check if indices are within bounds
        if (a_idx >= 0 && a_idx < a_signatures.size() && b_idx >= 0 && b_idx < b_signatures.size()) {
            // Compute similarity
            double similarity = compute_similarity(a_signatures[a_idx].second, b_signatures[b_idx].second);
            
            // Only include pairs above threshold
            if (similarity >= threshold) {
                outfile << "Party A index: " << a_idx << ", Party B index: " << b_idx << std::endl;
                outfile << "Party A text: " << a_signatures[a_idx].first << std::endl;
                outfile << "Party B text: " << b_signatures[b_idx].first << std::endl;
                outfile << "Similarity: " << similarity << std::endl;
                outfile << "Similarity threshold: " << threshold << std::endl;
                outfile << "---" << std::endl;
                count++;
            }
        }
    }
    
    outfile << "Total candidate pairs found: " << count << std::endl;
    
    a_sig_file.close();
    b_sig_file.close();
    idx_file.close();
    outfile.close();
    
    std::cout << "Verification completed. Found " << count << " candidate pairs." << std::endl;
}

// Run the complete PSI protocol
void run_psi(const std::string& party_a_file, const std::string& party_b_file, const std::string& output_dir, int role) {
    // Create output directory if it doesn't exist
    fs::create_directories(output_dir);
    
    // First run Party A to generate signatures and buckets
    if (role == 0 || role == 1) {
        std::string a_signatures = output_dir + "/a_signatures.bin";
        std::string buckets = output_dir + "/buckets.bin";
        
        // Party A: Generate signatures and buckets
        std::cout << "Party A: Generating signatures..." << std::endl;
        generate_signatures_party_a(party_a_file, a_signatures);
        
        std::cout << "Party A: Sending buckets to Party B..." << std::endl;
        send_buckets_party_a(a_signatures, buckets);
    }
    
    // Then run Party B to process buckets and generate candidates
    if (role == 0 || role == 1) {
        std::string b_signatures = output_dir + "/b_signatures.bin";
        std::string buckets = output_dir + "/buckets.bin";
        std::string candidates = output_dir + "/candidates.bin";
        std::string indices = output_dir + "/candidate_indices.bin";
        
        // Party B: Generate signatures and process buckets
        std::cout << "Party B: Generating signatures..." << std::endl;
        generate_signatures_party_b(party_b_file, b_signatures);
        
        std::cout << "Party B: Processing buckets and finding candidates..." << std::endl;
        process_buckets_party_b(b_signatures, buckets, candidates);
        
        std::cout << "Party B: Sending candidate indices to Party A..." << std::endl;
        send_candidates_party_b(candidates, indices);
        
        std::cout << "Party B: PSI completed." << std::endl;
    }
    
    // Finally run Party A to verify candidates
    if (role == 0 || role == 1) {
        std::string a_signatures = output_dir + "/a_signatures.bin";
        std::string b_signatures = output_dir + "/b_signatures.bin";
        std::string indices = output_dir + "/candidate_indices.bin";
        std::string results = output_dir + "/psi_results.txt";
        
        std::cout << "Party A: Verifying candidates..." << std::endl;
        verify_candidates_party_a(a_signatures, b_signatures, indices, results);
        
        std::cout << "Party A: PSI completed. Results saved to " << results << std::endl;
    }
}

// Clean up resources
void finalize_psi() {
    free(p);
    free(q);
    free(r);
    cudaFree(_p);
    cudaFree(_q);
    cudaFree(_r);
}
