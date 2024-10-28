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
    auto graph = m->graph(0);

    // Initialize RecordHessian
    RecordHessian record_hessian;
    record_hessian.initialize(m.get());

    // Provide an invalid input_data_path
    std::string invalid_input_data_path = "nonexistent_h5_file";

    // Call profileData and expect an exception
    EXPECT_ANY_THROW({
        std::unique_ptr<HessianMap> hessian_map = record_hessian.profileData(invalid_input_data_path);
    });
}
