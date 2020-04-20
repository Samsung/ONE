#include "Model.h"

#include <luci/Importer.h>
#include <luci/Service/Validate.h>
#include <luci/Pass/ShapeInferencePass.h>
#include <luci/Pass/TypeInferencePass.h>

#include <iostream>
#include <map>
#include <string>

namespace
{

void show_help_message(const char *progname, std::ostream &os)
{
  os << "USAGE: " << progname << " circlefile" << std::endl << std::endl;
}

void show_error_message(const char *progname, std::ostream &os, const std::string &msg)
{
  os << "ERROR: " << msg << std::endl;
  os << std::endl;

  show_help_message(progname, os);
}

} // namespace

/*
 * @brief ReadTest main
 *
 *        Give one Circle file as an argument
 *
 *        This will use luci_import to read the file and get loco graph
 *        In luci_import, LUCI_LOG environment will be checked and will
 *        dump graph to console if set.
 *        i.e. "LUCI_LOG=1 luci_readtester mymodel.circle"
 */
int main(int argc, char **argv)
{
  if (argc != 2)
  {
    show_error_message(argv[0], std::cerr, "Circle file is not specified");
    return 255;
  }

  std::string input_path = argv[1];

  std::cout << "[INFO] Circle is '" << input_path << "'" << std::endl;

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

  luci::Importer importer;
  auto module = importer.importModule(input_model);
  assert(module->size() > 0);

  for (size_t g = 0; g < module->size(); ++g)
  {
    auto graph = module->graph(g);
    if (graph == nullptr)
      return 255;

    {
      luci::ShapeInferencePass pass;
      while (pass.run(graph) == true)
        ;
    }
    {
      luci::TypeInferencePass pass;
      while (pass.run(graph) == true)
        ;
    }

    if (!luci::validate(graph))
      return 255;
  }
  return 0;
}
