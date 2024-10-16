// #include "record-hessian/RecordHessian.h"
// #include <gtest/gtest.h>
// #include <luci/IR/Module.h>
// #include <luci/IR/Graph.h>
// #include <luci/IR/CircleNodes.h>
// #include <luci_interpreter/Interpreter.h>
// #include <fstream>
// #include <cstdio>
// #include <cstdlib>
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
//   }

//   void TearDown() override
//   {
//     // Clean up temporary files and directories
//     if (!input_data_file.empty())
//       std::remove(input_data_file.c_str());
//     if (!input_list_file.empty())
//       std::remove(input_list_file.c_str());
//     if (!temp_dir.empty())
//       rmdir(temp_dir.c_str());
//   }

//   std::unique_ptr<luci::Module> module;
//   luci::Graph *graph;
//   luci::CircleInput *input_node;
//   luci::CircleConst *constant_node;
//   luci::CircleAdd *add_node;
//   luci::CircleOutput *output_node;

//   std::string input_data_file;
//   std::string input_list_file;
//   std::string temp_dir;
// };

// TEST_F(RecordHessianTest, InitializeValidModule)
// {
//   RecordHessian recorder;

//   EXPECT_NO_THROW(recorder.initialize(module.get()));
// }

// TEST_F(RecordHessianTest, InitializeNullModule_NEG)
// {
//   RecordHessian recorder;

//   EXPECT_ANY_THROW(recorder.initialize(nullptr));
// }

// // TEST_F(RecordHessianTest, ProfileRawDataDirectoryValidInput)
// // {
// //   RecordHessian recorder;
// //   recorder.initialize(module.get());

// //   // Create temporary directory
// //   char dir_template[] = "/tmp/testdirXXXXXX";
// //   char *dir_name = mkdtemp(dir_template);
// //   ASSERT_NE(dir_name, nullptr);
// //   temp_dir = dir_name;

// //   // Create a data file in the directory
// //   input_data_file = temp_dir + "/input_data.bin";

// //   // Write some data to the file
// //   std::vector<float> input_data(28 * 28, 1.0f);
// //   std::ofstream ofs(input_data_file, std::ios::binary);
// //   ofs.write(reinterpret_cast<const char *>(input_data.data()),
// //             input_data.size() * sizeof(float));
// //   ofs.close();

// //   // Now call profileRawDataDirectory
// //   EXPECT_NO_THROW(recorder.profileRawDataDirectory(temp_dir));
// // }

// // TEST_F(RecordHessianTest, ProfileRawDataDirectoryEmptyDir_NEG)
// // {
// //   RecordHessian recorder;
// //   recorder.initialize(module.get());

// //   // Create temporary directory
// //   char dir_template[] = "/tmp/testdirXXXXXX";
// //   char *dir_name = mkdtemp(dir_template);
// //   ASSERT_NE(dir_name, nullptr);
// //   temp_dir = dir_name;

// //   // Directory is empty, expect an exception
// //   EXPECT_ANY_THROW(recorder.profileRawDataDirectory(temp_dir));
// // }

// // TEST_F(RecordHessianTest, ProfileRawDataValidInput)
// // {
// //   RecordHessian recorder;
// //   recorder.initialize(module.get());

// //   // Create a temporary file for the list of input files
// //   char list_template[] = "/tmp/testlistXXXXXX";
// //   int list_fd = mkstemp(list_template);
// //   ASSERT_NE(list_fd, -1);
// //   input_list_file = list_template;

// //   // Create an input data file
// //   char data_template[] = "/tmp/testdataXXXXXX";
// //   int data_fd = mkstemp(data_template);
// //   ASSERT_NE(data_fd, -1);
// //   input_data_file = data_template;

// //   // Write some data to the input data file
// //   std::vector<float> input_data(28 * 28, 1.0f);
// //   write(data_fd, input_data.data(), input_data.size() * sizeof(float));
// //   close(data_fd);

// //   // Write the data file path to the list file
// //   std::ofstream list_ofs(input_list_file);
// //   list_ofs << input_data_file << std::endl;
// //   list_ofs.close();

// //   // Now call profileRawData
// //   EXPECT_NO_THROW(recorder.profileRawData(input_list_file));
// // }

// // TEST_F(RecordHessianTest, ProfileRawDataInvalidFile_NEG)
// // {
// //   RecordHessian recorder;
// //   recorder.initialize(module.get());

// //   // Create a temporary file for the list of input files
// //   char list_template[] = "/tmp/testlistXXXXXX";
// //   int list_fd = mkstemp(list_template);
// //   ASSERT_NE(list_fd, -1);
// //   input_list_file = list_template;
// //   close(list_fd);

// //   // Write an invalid data file path to the list file
// //   std::ofstream list_ofs(input_list_file);
// //   list_ofs << "/invalid/path/to/data.bin" << std::endl;
// //   list_ofs.close();

// //   // Now call profileRawData, expect an exception
// //   EXPECT_ANY_THROW(recorder.profileRawData(input_list_file));
// // }

// // TEST_F(RecordHessianTest, ProfileRawDataWrongInputSize_NEG)
// // {
// //   RecordHessian recorder;
// //   recorder.initialize(module.get());

// //   // Create a temporary file for the list of input files
// //   char list_template[] = "/tmp/testlistXXXXXX";
// //   int list_fd = mkstemp(list_template);
// //   ASSERT_NE(list_fd, -1);
// //   input_list_file = list_template;

// //   // Create an input data file with wrong size
// //   char data_template[] = "/tmp/testdataXXXXXX";
// //   int data_fd = mkstemp(data_template);
// //   ASSERT_NE(data_fd, -1);
// //   input_data_file = data_template;

// //   // Write incorrect data size
// //   std::vector<float> input_data(10, 1.0f); // Incorrect size
// //   write(data_fd, input_data.data(), input_data.size() * sizeof(float));
// //   close(data_fd);

// //   // Write the data file path to the list file
// //   std::ofstream list_ofs(input_list_file);
// //   list_ofs << input_data_file << std::endl;
// //   list_ofs.close();

// //   // Now call profileRawData, expect an exception
// //   EXPECT_ANY_THROW(recorder.profileRawData(input_list_file));
// // }

// // TEST_F(RecordHessianTest, ProfileRawDataMultipleInputs)
// // {
// //   // Modify the graph to have two inputs
// //   auto input_node2 = graph->nodes()->create<luci::CircleInput>();
// //   input_node2->name("input_1");
// //   input_node2->index(1);
// //   input_node2->rank(4);
// //   input_node2->dim(0) = 1;
// //   input_node2->dim(1) = 28;
// //   input_node2->dim(2) = 28;
// //   input_node2->dim(3) = 1;
// //   input_node2->dtype(loco::DataType::FLOAT32);

// //   // Modify the add_node to add two inputs
// //   add_node->y(input_node2);

// //   // Add the second input to graph inputs
// //   graph->inputs()->push_back(input_node2);

// //   RecordHessian recorder;
// //   recorder.initialize(module.get());

// //   // Create a temporary file for the list of input files
// //   char list_template[] = "/tmp/testlistXXXXXX";
// //   int list_fd = mkstemp(list_template);
// //   ASSERT_NE(list_fd, -1);
// //   input_list_file = list_template;

// //   // Create two input data files
// //   char data_template1[] = "/tmp/testdata1XXXXXX";
// //   int data_fd1 = mkstemp(data_template1);
// //   ASSERT_NE(data_fd1, -1);
// //   std::string input_data_file1 = data_template1;

// //   char data_template2[] = "/tmp/testdata2XXXXXX";
// //   int data_fd2 = mkstemp(data_template2);
// //   ASSERT_NE(data_fd2, -1);
// //   std::string input_data_file2 = data_template2;

// //   // Write data to the input data files
// //   std::vector<float> input_data(28 * 28, 1.0f);
// //   write(data_fd1, input_data.data(), input_data.size() * sizeof(float));
// //   close(data_fd1);

// //   write(data_fd2, input_data.data(), input_data.size() * sizeof(float));
// //   close(data_fd2);

// //   // Write the data file paths to the list file
// //   std::ofstream list_ofs(input_list_file);
// //   list_ofs << input_data_file1 << " " << input_data_file2 << std::endl;
// //   list_ofs.close();

// //   // Now call profileRawData
// //   EXPECT_NO_THROW(recorder.profileRawData(input_list_file));

// //   // Clean up additional files
// //   std::remove(input_data_file1.c_str());
// //   std::remove(input_data_file2.c_str());
// // }

// // TEST_F(RecordHessianTest, ProfileRawDataWrongNumberOfInputs_NEG)
// // {
// //   // Modify the graph to have two inputs
// //   auto input_node2 = graph->nodes()->create<luci::CircleInput>();
// //   input_node2->name("input_1");
// //   input_node2->index(1);
// //   input_node2->rank(4);
// //   input_node2->dim(0) = 1;
// //   input_node2->dim(1) = 28;
// //   input_node2->dim(2) = 28;
// //   input_node2->dim(3) = 1;
// //   input_node2->dtype(loco::DataType::FLOAT32);

// //   // Modify the add_node to add two inputs
// //   add_node->y(input_node2);

// //   // Add the second input to graph inputs
// //   graph->inputs()->push_back(input_node2);

// //   RecordHessian recorder;
// //   recorder.initialize(module.get());

// //   // Create a temporary file for the list of input files
// //   char list_template[] = "/tmp/testlistXXXXXX";
// //   int list_fd = mkstemp(list_template);
// //   ASSERT_NE(list_fd, -1);
// //   input_list_file = list_template;

// //   // Create only one input data file
// //   char data_template1[] = "/tmp/testdata1XXXXXX";
// //   int data_fd1 = mkstemp(data_template1);
// //   ASSERT_NE(data_fd1, -1);
// //   std::string input_data_file1 = data_template1;

// //   // Write data to the input data file
// //   std::vector<float> input_data(28 * 28, 1.0f);
// //   write(data_fd1, input_data.data(), input_data.size() * sizeof(float));
// //   close(data_fd1);

// //   // Write the data file path to the list file
// //   std::ofstream list_ofs(input_list_file);
// //   list_ofs << input_data_file1 << std::endl; // Only one input instead of two
// //   list_ofs.close();

// //   // Now call profileRawData, expect an exception
// //   EXPECT_ANY_THROW(recorder.profileRawData(input_list_file));

// //   // Clean up additional file
// //   std::remove(input_data_file1.c_str());
// // }