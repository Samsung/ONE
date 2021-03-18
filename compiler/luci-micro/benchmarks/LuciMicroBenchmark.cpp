#include "LuciMicroBenchmark.hpp"
using namespace LuciMicroBenchmarks;
LuciMicroBenchmark::LuciMicroBenchmark(const osPriority_t thread_priority, const char *name,
                                       LuciMicro::ILogger &log)
    : LuciMicro::ICore(thread_priority, name), log_(log)
{
}
LuciMicroBenchmark::~LuciMicroBenchmark() {}
void LuciMicroBenchmark::setup(void)
{
  run_benchmark();
  _is_stopped = true;
}
void LuciMicroBenchmark::fill_in_tensor(std::vector<char> &data, loco::DataType dtype)
{
  switch (dtype)
  {
    case loco::DataType::FLOAT32:
      for (int i = 0; i < data.size() / sizeof(float); ++i)
      {
        reinterpret_cast<float *>(data.data())[i] = 123.f;
      }
      break;
    default:
      assert(false);
  }
}
void LuciMicroBenchmark::run_benchmark()
{
  while (!_is_stopped)
  {
    // #ifdef TEST_LUCIINTERPRETER
    printf("STM32F767 SystemCoreClock %d\n", SystemCoreClock);
    printf("Model NET_0000.circle\n");

    std::vector<char> *buf =
        new std::vector<char>(circle_model_raw, circle_model_raw + sizeof(circle_model_raw) /
                                                                       sizeof(circle_model_raw[0]));
    std::vector<char> &model_data = *buf;
    // Verify flatbuffers
    flatbuffers::Verifier verifier{
        static_cast<const uint8_t *>(static_cast<void *>(model_data.data())), model_data.size()};
    printf("circle::VerifyModelBuffer\n");
    if (!circle::VerifyModelBuffer(verifier))
    {
      printf("ERROR: Failed to verify circle\n");
    }
    printf("OK\n");
    // auto model = circle::GetModel(static_cast<const uint8_t *>(static_cast<void
    // *>(model_data.data()))); printf("%s\n", model->description()->c_str());
    printf("luci::Importer().importModule\n");

    auto module = luci::Importer().importModule(circle::GetModel(model_data.data()));

    if (module == nullptr)
    {
      printf("ERROR: Failed to load \n");
    }
    printf("OK\n");

    printf("luci_interpreter::Interpreter\n");

    auto interpreter = std::make_unique<luci_interpreter::Interpreter>(module.get());
    auto nodes = module->graph()->nodes();
    auto nodes_count = nodes->size();
    printf("nodes_count: %d\n", nodes_count);
    // Fill input tensors with some garbage
    while (true)
    {
      Timer t;
      for (int i = 0; i < nodes_count; ++i)
      {
        auto *node = dynamic_cast<luci::CircleNode *>(nodes->at(i));
        assert(node);
        if (node->opcode() == luci::CircleOpcode::CIRCLEINPUT)
        {
          auto *input_node = static_cast<luci::CircleInput *>(node);
          loco::GraphInput *g_input = module->graph()->inputs()->at(input_node->index());
          const loco::TensorShape *shape = g_input->shape();
          size_t data_size = 1;
          for (int d = 0; d < shape->rank(); ++d)
          {
            assert(shape->dim(d).known());
            data_size *= shape->dim(d).value();
          }
          data_size *= loco::size(g_input->dtype());
          std::vector<char> data(data_size);
          fill_in_tensor(data, g_input->dtype());

          interpreter->writeInputTensor(static_cast<luci::CircleInput *>(node), data.data(),
                                        data_size);
        }
      }
      t.start();
      interpreter->interpret();
      t.stop();
      printf("\rFinished in %dus   ", t.read_us());
      ThisThread::sleep_for(100ms);
    }
    // #endif
    ThisThread::sleep_for(100);
  }
}
