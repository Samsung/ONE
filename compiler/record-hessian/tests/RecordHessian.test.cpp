#include "record-hessian/RecordHessian.h"
#include <gtest/gtest.h>
#include <luci/IR/Nodes/CircleInput.h>
#include <luci/IR/Nodes/CircleOutput.h>
#include <luci/IR/Nodes/CircleFullyConnected.h>
#include <luci/IR/Nodes/CircleConst.h>
#include <luci/IR/Module.h>
#include <luci/Importer.h>
#include <luci_interpreter/Interpreter.h>

using namespace record_hessian;

TEST(RecordHessianTest, profileDataInvalidInputPath_NEG)
{
    // Create a module and a graph
    auto m = luci::make_module();

    // Initialize RecordHessian
    RecordHessian rh;
    rh.initialize(m.get());

    // Provide an invalid input_data_path
    std::string invalid_input_data_path = "invalid_h5_file";

    // Call profileData and expect an exception
    EXPECT_ANY_THROW({
        std::unique_ptr<HessianMap> hessian_map = rh.profileData(invalid_input_data_path);
    });
}

TEST(RecordHessianTest, profileDataNonexistingFile_NEG)
{
    // Create a module and a graph
    auto m = luci::make_module();

    // Initialize RecordHessian
    RecordHessian rh;
    rh.initialize(m.get());

    // // Provide an invalid input_data_path
    std::string non_existing_h5 = "non_existing.h5";

    // // Call profileData and expect an exception
    EXPECT_ANY_THROW({
        std::unique_ptr<HessianMap> hessian_map = rh.profileData(non_existing_h5);
    });
}
