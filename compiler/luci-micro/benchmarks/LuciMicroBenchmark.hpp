#ifndef LUCIMICROBENCHMARK_HPP
#define LUCIMICROBENCHMARK_HPP

#include "include/abstract/ICore.hpp"
#include "include/abstract/ILogger.hpp"
#include <luci_interpreter/Interpreter.h>
#include <lib/import/include/luci/Importer.h>
#include <luci/IR/Module.h>
#include <loco/IR/DataTypeTraits.h>
#include "mbed.h"
#include <iostream>
#include "circlemodel.h"
#include "resources/mio/circle/schema_generated.h"
namespace LuciMicroBenchmarks
{

class LuciMicroBenchmark : public LuciMicro::ICore
{
public:
  LuciMicroBenchmark(const osPriority_t thread_priority, const char *name, LuciMicro::ILogger &log);
  virtual ~LuciMicroBenchmark();

protected:
  void loop(void) override;
  void setup(void) override;
  LuciMicro::ILogger &log_;

private:
  void run_benchmark();
  void fill_in_tensor(std::vector<char> &data, loco::DataType dtype);
};
};     // namespace LuciMicroBenchmarks
#endif // LUCIMICROBENCHMARK_HPP
