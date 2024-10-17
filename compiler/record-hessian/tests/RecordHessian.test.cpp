// #include "record-hessian/RecordHessian.h"

// #include <gtest/gtest.h>
// #include <luci/IR/CircleNodes.h>
// #include <luci/IR/Graph.h>
// #include <luci/IR/Module.h>
// #include <luci_interpreter/Interpreter.h>

// #include <fstream>
// #include <random>
// #include <sys/stat.h>
// #include <unistd.h>

// using namespace record_hessian;

// class RecordHessianTest : public ::testing::Test
// {
// protected:
//   void SetUp() override
//   {
//     // Create a luci::Module and a graph
//     module = std::make_unique<luci::Module>();
//     graph = module->createGraph();

//     // Create an input node
//     input_node = graph->nodes()->create<luci::CircleInput>();
//     input_node->name("input_0");
//     input_node->index(0);
//     input_node->rank(4);
//     input_node->dim(0) = 1;
//     input_node->dim(1) = 28;
//     input_node->dim(2) = 28;
//     input_node->dim(3) = 1;
//     input_node->dtype(loco::DataType::FLOAT32);

//     // Create a constant node
//     constant_node = graph->nodes()->create<luci::CircleConst>();
//     constant_node->dtype(loco::DataType::FLOAT32);
//     constant_node->size<loco::DataType::FLOAT32>(1);
//     constant_node->rank(1);
//     constant_node->dim(0) = 1;
//     constant_node->scalar<float>() = 1.0f;

//     // Create an Add operator
//     add_node = graph->nodes()->create<luci::CircleAdd>();
//     add_node->x(input_node);
//     add_node->y(constant_node);
//     add_node->dtype(loco::DataType::FLOAT32);
//     add_node->rank(4);
//     add_node->dim(0) = input_node->dim(0);
//     add_node->dim(1) = input_node->dim(1);
//     add_node->dim(2) = input_node->dim(2);
//     add_node->dim(3) = input_node->dim(3);

//     // Create an output node
//     output_node = graph->nodes()->create<luci::CircleOutput>();
//     output_node->from(add_node);
//     output_node->index(0);

//     // Set the graph inputs and outputs
//     graph->inputs()->push_back(input_node);
//     graph->outputs()->push_back(output_node);

//     // Create a temporary directory for input data
//     char temp_dir_template[] = "/tmp/record_hessian_test_XXXXXX";
//     temp_dir = mkdtemp(temp_dir_template);
//     ASSERT_FALSE(temp_dir.empty()) << "Failed to create temporary directory";

//     // Create an HDF5 file with input data
//     input_data_file = std::string(temp_dir) + "/input_data.h5";
//     createInputDataFile(input_data_file);
//   }

//   void TearDown() override
//   {
//     // Clean up temporary files and directories
//     if (!input_data_file.empty())
//       std::remove(input_data_file.c_str());
//     if (!temp_dir.empty())
//       rmdir(temp_dir.c_str());
//   }

//   void createInputDataFile(const std::string &filename)
//   {
//     // Create an HDF5 file and write random input data
//     H5::H5File file(filename, H5F_ACC_TRUNC);

//     // Create a group named "value"
//     H5::Group group = file.createGroup("/value");

//     // Create a dataset for the input tensor
//     hsize_t dims[4] = {1, 28, 28, 1};
//     H5::DataSpace dataspace(4, dims);
//     H5::DataSet dataset = group.createDataSet("input_0", H5::PredType::NATIVE_FLOAT, dataspace);

//     // Generate random data
//     std::vector<float> data(1 * 28 * 28 * 1);
//     std::generate(data.begin(), data.end(), []() { return static_cast<float>(rand()) / RAND_MAX; });

//     // Write data to the dataset
//     dataset.write(data.data(), H5::PredType::NATIVE_FLOAT);

//     file.close();
//   }

//   std::unique_ptr<luci::Module> module;
//   luci::Graph *graph;
//   luci::CircleInput *input_node;
//   luci::CircleConst *constant_node;
//   luci::CircleAdd *add_node;
//   luci::CircleOutput *output_node;

//   std::string input_data_file;
//   std::string temp_dir;
// };

// // Positive Test: Test that profileData works with valid input data
// TEST_F(RecordHessianTest, ProfileDataValidInput)
// {
//   RecordHessian recorder;
//   recorder.initialize(module.get());

//   EXPECT_NO_THROW({
//     auto hessian_map = recorder.profileData(input_data_file);
//     ASSERT_TRUE(hessian_map != nullptr);
//   });
// }

// // Negative Test: Test that profileData throws an exception with invalid input data path
// TEST_F(RecordHessianTest, ProfileDataInvalidInputPath)
// {
//   RecordHessian recorder;
//   recorder.initialize(module.get());

//   std::string invalid_path = "/invalid/path/input_data.h5";
//   EXPECT_THROW({
//     auto hessian_map = recorder.profileData(invalid_path);
//   },
//                std::runtime_error);
// }

// // Negative Test: Test that profileData throws an exception when input data is missing
// TEST_F(RecordHessianTest, ProfileDataMissingInput)
// {
//   // Remove the input data file to simulate missing data
//   std::remove(input_data_file.c_str());

//   RecordHessian recorder;
//   recorder.initialize(module.get());

//   EXPECT_THROW({
//     auto hessian_map = recorder.profileData(input_data_file);
//   },
//                std::runtime_error);
// }

// // Negative Test: Test that profileData throws an exception when input tensor shape mismatches
// TEST_F(RecordHessianTest, ProfileDataShapeMismatch)
// {
//   // Modify the input node's shape to mismatch with the data
//   input_node->dim(1) = 32;

//   RecordHessian recorder;
//   recorder.initialize(module.get());

//   EXPECT_THROW({
//     auto hessian_map = recorder.profileData(input_data_file);
//   },
//                std::runtime_error);
// }

// // Negative Test: Test that profileData throws an exception when input tensor type mismatches
// TEST_F(RecordHessianTest, ProfileDataTypeMismatch)
// {
//   // Modify the input node's data type to mismatch with the data
//   input_node->dtype(loco::DataType::S32);

//   RecordHessian recorder;
//   recorder.initialize(module.get());

//   EXPECT_THROW({
//     auto hessian_map = recorder.profileData(input_data_file);
//   },
//                std::runtime_error);
// }

// // Positive Test: Test profileData with multiple records in the input data
// TEST_F(RecordHessianTest, ProfileDataMultipleRecords)
// {
//   // Create an HDF5 file with multiple records
//   std::string multi_record_file = std::string(temp_dir) + "/multi_input_data.h5";

//   H5::H5File file(multi_record_file, H5F_ACC_TRUNC);
//   H5::Group group = file.createGroup("/value");

//   const int num_records = 5;
//   for (int i = 0; i < num_records; ++i)
//   {
//     std::string dataset_name = "input_0_" + std::to_string(i);
//     hsize_t dims[4] = {1, 28, 28, 1};
//     H5::DataSpace dataspace(4, dims);
//     H5::DataSet dataset = group.createDataSet(dataset_name, H5::PredType::NATIVE_FLOAT, dataspace);

//     // Generate random data
//     std::vector<float> data(1 * 28 * 28 * 1);
//     std::generate(data.begin(), data.end(),
//                   []() { return static_cast<float>(rand()) / RAND_MAX; });

//     // Write data to the dataset
//     dataset.write(data.data(), H5::PredType::NATIVE_FLOAT);
//   }
//   file.close();

//   RecordHessian recorder;
//   recorder.initialize(module.get());

//   EXPECT_NO_THROW({
//     auto hessian_map = recorder.profileData(multi_record_file);
//     ASSERT_TRUE(hessian_map != nullptr);
//   });

//   // Clean up
//   std::remove(multi_record_file.c_str());
// }

// // Negative Test: Test that initialize throws an exception when module is nullptr
// TEST(RecordHessianStandaloneTest, InitializeWithNullModule)
// {
//   RecordHessian recorder;
//   EXPECT_THROW({
//     recorder.initialize(nullptr);
//   },
//                std::runtime_error);
// }