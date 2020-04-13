#include "Model.h"

#include <luci/Importer.h>
#include <luci/Pass/ShapeInferencePass.h>
#include <luci/Pass/TypeInferencePass.h>
#include <luci/Service/Validate.h>
#include <luci/CircleExporter.h>
#include <oops/InternalExn.h>

#include <fstream>
#include <iostream>
#include <map>
#include <string>

namespace
{

void show_help_message(const char *progname, std::ostream &os)
{
  os << "USAGE: " << progname << " circlefile_in circlefile_out" << std::endl << std::endl;
}

void show_error_message(const char *progname, std::ostream &os, const std::string &msg)
{
  os << "ERROR: " << msg << std::endl;
  os << std::endl;

  show_help_message(progname, os);
}

struct CircleExpContract : public luci::CircleExporter::Contract
{
public:
  CircleExpContract(loco::Graph *graph, const std::string &filename)
      : _graph(graph), _filepath(filename)
  {
    // NOTHING TO DO
  }
  virtual ~CircleExpContract() = default;

public:
  loco::Graph *graph(void) const final { return _graph; }

public:
  bool store(const char *ptr, const size_t size) const final;

private:
  loco::Graph *_graph;
  const std::string _filepath;
};

bool CircleExpContract::store(const char *ptr, const size_t size) const
{
  if (!ptr)
    INTERNAL_EXN("Graph was not serialized by FlatBuffer for some reason");

  std::ofstream fs(_filepath.c_str(), std::ofstream::binary);
  fs.write(ptr, size);

  return fs.good();
}

} // namespace

/*
 * @brief WriteTester main
 *
 *        Give two Circle file as an argument
 *
 *        This will use luci_import to read the first file and get loco graph
 *        With the graph, this will use luci_export to write to the second file
 *        Like ReadTester, LUCI_LOG=1 environment variable is available to dump the graph
 */
int main(int argc, char **argv)
{
  if (argc != 3)
  {
    show_error_message(argv[0], std::cerr, "In/Out Circle file path is not specified");
    return 255;
  }

  std::string input_path = argv[1];
  std::string output_path = argv[2];

  std::cout << "[INFO] Circle from '" << input_path << "' to '" << output_path << "'" << std::endl;

  // Load model from the file
  std::unique_ptr<luci::Model> model = luci::load_model(input_path);
  if (model == nullptr)
  {
    std::cerr << "ERROR: Failed to load '" << input_path << "'" << std::endl;
    return 255;
  }

  const circle::Model *input_model = model->model();
  if (input_model == nullptr)
  {
    std::cerr << "ERROR: Failed to read '" << input_path << "'" << std::endl;
    return 255;
  }

  // Import from input Circle file
  luci::Importer importer;
  auto graph = importer.import(input_model);

  if (graph.get() == nullptr)
    return 255;

  {
    luci::ShapeInferencePass pass;
    while (pass.run(graph.get()) == true)
      ;
  }
  {
    luci::TypeInferencePass pass;
    while (pass.run(graph.get()) == true)
      ;
  }

  if (!luci::validate(graph.get()))
    return 255;

  // Export to output Circle file
  luci::CircleExporter exporter;

  CircleExpContract contract(graph.get(), output_path);

  return exporter.invoke(&contract) ? 0 : 255;
}
