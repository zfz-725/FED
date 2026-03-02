#include <iostream>
#include <string>
#include <filesystem>
#include "mpi.h"
#include "psi.h"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    if (argc != 4) {
        std::cerr << "Usage: ./psi_test <party_a_file> <party_b_file> <output_dir>" << std::endl;
        MPI_Finalize();
        return 1;
    }
    
    std::string party_a_file = argv[1];
    std::string party_b_file = argv[2];
    std::string output_dir = argv[3];
    
    // Create output directory if it doesn't exist
    fs::create_directories(output_dir);
    
    // Initialize PSI with parameters for fuzzy matching
    init_psi(128, 3, 20, 0.6);
    
    // Run Party A's side
    std::cout << "Running Party A..." << std::endl;
    run_psi(party_a_file, party_b_file, output_dir, 0);
    
    // Run Party B's side
    std::cout << "\nRunning Party B..." << std::endl;
    run_psi(party_a_file, party_b_file, output_dir, 1);
    
    // Clean up
    finalize_psi();
    
    // Finalize MPI
    MPI_Finalize();
    
    std::cout << "\nPSI test completed!" << std::endl;
    return 0;
}
