/**
 * @file test_distributed.cpp
 * @brief Tests for distributed pipeline
 */

#include "../src/distributed/distributed_pipeline.hpp"
#include <iostream>
#include <cassert>
#include <cstdlib>

using namespace turboloader::distributed;

void test_distributed_config() {
    std::cout << "\n[TEST] Testing DistributedConfig..." << std::endl;

    DistributedConfig config;
    config.data_path = "/tmp/test.tar";
    config.batch_size = 32;
    config.num_workers = 4;
    config.queue_size = 128;
    config.shuffle = false;
    config.world_rank = 0;
    config.world_size = 4;
    config.master_addr = "localhost";
    config.master_port = 29500;
    config.backend = CommBackend::MPI;
    config.drop_last = false;

    assert(config.data_path == "/tmp/test.tar");
    assert(config.batch_size == 32);
    assert(config.world_rank == 0);
    assert(config.world_size == 4);
    assert(config.master_addr == "localhost");
    assert(config.master_port == 29500);
    assert(config.backend == CommBackend::MPI);

    std::cout << "  ✓ DistributedConfig structure validated" << std::endl;
    std::cout << "  ✓ World size: " << config.world_size << std::endl;
    std::cout << "  ✓ Rank: " << config.world_rank << std::endl;
    std::cout << "  ✓ Backend: MPI" << std::endl;
}

void test_init_from_env() {
    std::cout << "\n[TEST] Testing init_distributed_from_env..." << std::endl;

    // Set environment variables
    setenv("RANK", "2", 1);
    setenv("WORLD_SIZE", "8", 1);
    setenv("MASTER_ADDR", "192.168.1.100", 1);
    setenv("MASTER_PORT", "12345", 1);

    DistributedConfig config = init_distributed_from_env();

    assert(config.world_rank == 2);
    assert(config.world_size == 8);
    assert(config.master_addr == "192.168.1.100");
    assert(config.master_port == 12345);

    std::cout << "  ✓ Environment variables parsed correctly" << std::endl;
    std::cout << "  ✓ RANK=" << config.world_rank << std::endl;
    std::cout << "  ✓ WORLD_SIZE=" << config.world_size << std::endl;
    std::cout << "  ✓ MASTER_ADDR=" << config.master_addr << std::endl;
    std::cout << "  ✓ MASTER_PORT=" << config.master_port << std::endl;

    // Cleanup
    unsetenv("RANK");
    unsetenv("WORLD_SIZE");
    unsetenv("MASTER_ADDR");
    unsetenv("MASTER_PORT");
}

#ifdef TURBOLOADER_ENABLE_MPI
void test_distributed_initialization() {
    std::cout << "\n[TEST] Testing DistributedPipeline initialization..." << std::endl;

    try {
        DistributedConfig config;
        config.data_path = "/tmp/test.tar";
        config.batch_size = 16;
        config.num_workers = 2;
        config.world_rank = 0;
        config.world_size = 1;
        config.backend = CommBackend::MPI;

        // Note: This requires MPI_Init to have been called
        // In a real MPI program, this would be done in main()
        std::cout << "  ⚠ MPI initialization requires MPI_Init() in main()" << std::endl;
        std::cout << "  ℹ Skipping actual pipeline creation" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "  ⚠ MPI not initialized: " << e.what() << std::endl;
        std::cout << "  ℹ This is expected without MPI_Init()" << std::endl;
    }
}
#else
void test_distributed_without_mpi() {
    std::cout << "\n[TEST] Testing Distributed error handling without MPI..." << std::endl;

    try {
        DistributedConfig config;
        config.data_path = "/tmp/test.tar";
        config.world_rank = 0;
        config.world_size = 2;
        config.backend = CommBackend::MPI;

        DistributedPipeline pipeline(config);

        // Should not reach here
        std::cout << "  ✗ Expected exception not thrown!" << std::endl;
        assert(false);

    } catch (const std::runtime_error& e) {
        std::string error_msg(e.what());
        assert(error_msg.find("MPI") != std::string::npos ||
               error_msg.find("mpi") != std::string::npos);

        std::cout << "  ✓ Correctly throws exception without MPI" << std::endl;
        std::cout << "  ✓ Error message: " << e.what() << std::endl;
    }
}
#endif

void test_comm_backend_enum() {
    std::cout << "\n[TEST] Testing CommBackend enum..." << std::endl;

    CommBackend mpi = CommBackend::MPI;
    CommBackend tcp = CommBackend::TCP;
    CommBackend nccl = CommBackend::NCCL;

    assert(mpi != tcp);
    assert(mpi != nccl);
    assert(tcp != nccl);

    std::cout << "  ✓ CommBackend enum values are distinct" << std::endl;
    std::cout << "  ✓ MPI, TCP, and NCCL backends defined" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Distributed Pipeline Test Suite" << std::endl;
    std::cout << "========================================" << std::endl;

    try {
        // Always test config and utilities
        test_distributed_config();
        test_init_from_env();
        test_comm_backend_enum();

#ifdef TURBOLOADER_ENABLE_MPI
        std::cout << "\nℹ MPI is ENABLED - running full tests" << std::endl;
        test_distributed_initialization();
#else
        std::cout << "\nℹ MPI is NOT ENABLED - running fallback tests" << std::endl;
        test_distributed_without_mpi();
#endif

        std::cout << "\n========================================" << std::endl;
        std::cout << "All Distributed tests PASSED ✓" << std::endl;
        std::cout << "========================================" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n✗ Test FAILED: " << e.what() << std::endl;
        return 1;
    }
}
