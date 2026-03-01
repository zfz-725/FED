#include <string>
#include <vector>

// Initialize PSI with LSH parameters
void init_psi(int num_hash, int shingle_len, int bucket, double threshold);

// Generate LSH signatures for Party A's dataset
void generate_signatures_party_a(const std::string& input_file, std::string& output_signatures);

// Generate LSH signatures for Party B's dataset
void generate_signatures_party_b(const std::string& input_file, std::string& output_signatures);

// Party A sends buckets to Party B
void send_buckets_party_a(const std::string& signatures_file, std::string& buckets_file);

// Party B receives buckets from Party A and finds candidate pairs
void process_buckets_party_b(const std::string& signatures_file, const std::string& buckets_file, std::string& candidates_file);

// Party B sends candidate indices to Party A
void send_candidates_party_b(const std::string& candidates_file, std::string& indices_file);

// Party A verifies candidates and computes final results
void verify_candidates_party_a(const std::string& a_signatures_file, const std::string& b_signatures_file, const std::string& indices_file, std::string& results_file);

// Run the complete PSI protocol
void run_psi(const std::string& party_a_file, const std::string& party_b_file, const std::string& output_dir, int role);

// Clean up resources
void finalize_psi();
