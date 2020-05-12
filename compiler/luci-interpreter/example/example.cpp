#include <luci/Importer.h>
#include <luci_interpreter/Interpreter.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

static void readDataFromFile(const std::string &filename, char *data, size_t data_size)
{
  std::ifstream fs(filename, std::ifstream::binary);
  if (fs.fail())
    throw std::runtime_error("Cannot open file \"" + filename + "\".\n");
  if (fs.read(data, data_size).fail())
    throw std::runtime_error("Failed to read data from file \"" + filename + "\".\n");
}

static void writeDataToFile(const std::string &filename, const char *data, size_t data_size)
{
  std::ofstream fs(filename, std::ofstream::binary);
  if (fs.fail())
    throw std::runtime_error("Cannot open file \"" + filename + "\".\n");
  if (fs.write(data, data_size).fail())
  {
    throw std::runtime_error("Failed to write data to file \"" + filename + "\".\n");
  }
}

static std::unique_ptr<luci::Module> importModel(const std::string &filename)
{
  std::ifstream fs(filename, std::ifstream::binary);
  if (fs.fail())
    throw std::runtime_error("Cannot open model file \"" + filename + "\".\n");
  std::vector<char> model_data((std::istreambuf_iterator<char>(fs)),
                               std::istreambuf_iterator<char>());
  return luci::Importer().importModule(circle::GetModel(model_data.data()));
}

template <typename NodeT> size_t getTensorSize(const NodeT *node)
{
  uint32_t tensor_size = loco::size(node->dtype());
  for (uint32_t i = 0; i < node->rank(); ++i)
    tensor_size *= node->dim(i).value();
  return tensor_size;
}

int main(int argc, char *argv[])
{
  if (argc != 4)
  {
    std::cerr << "Usage: " << argv[0]
              << " <path/to/circle/model> <path/to/input/file> <path/to/output/file>\n";
    return EXIT_FAILURE;
  }

  const char *filename = argv[1];
  const char *input_file = argv[2];
  const char *output_file = argv[3];

  std::unique_ptr<luci::Module> module = importModel(filename);

  // Create interpreter.
  luci_interpreter::Interpreter interpreter(module.get());

  // Set input data.
  const auto *input_node =
      dynamic_cast<const luci::CircleInput *>(loco::input_nodes(module->graph())[0]);
  std::vector<char> input_data(getTensorSize(input_node));
  readDataFromFile(input_file, input_data.data(), input_data.size());
  interpreter.writeInputTensor(input_node, input_data.data(), input_data.size());

  // Do inference.
  interpreter.interpret();

  // Get output data.
  const auto *output_node =
      dynamic_cast<const luci::CircleOutput *>(loco::output_nodes(module->graph())[0]);
  std::vector<char> output_data(getTensorSize(output_node));
  interpreter.readOutputTensor(output_node, output_data.data(), output_data.size());
  writeDataToFile(output_file, output_data.data(), output_data.size());

  return EXIT_SUCCESS;
}
