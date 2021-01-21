#include "LuciCodegen.h"
#include "luci/Importer.h"
#include "luci/CircleExporter.h"
#include "luci/CircleFileExpContract.h"

#include <iostream>
#include <string>

int main(int argc, char **argv)
{
  if (argc != 3)
  {
    std::cout << "Usage: ./circle_codegen <input circle file> <output package name>\n";
    return 1;
  }
  std::string input_circle_name = argv[1];
  std::string output_package_name = argv[2];
  luci::Importer importer;
  const circle::Model *circle_module;
  std::unique_ptr<luci::Module> luci_module = importer.importModule(circle_module);

  luci_codegen::Options options;
  // set options if needed
  luci_codegen::LuciCodegen codegen(options);
  codegen.process(*luci_module);
  codegen.emit_code(output_package_name);

  luci::CircleExporter exporter;
  luci::CircleFileExpContract contract(luci_module.get(), output_package_name + ".circle");
  exporter.invoke(&contract);
  return 0;
}
