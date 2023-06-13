/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "HEScheduler.h"
#include "../exec/ExecTime.h"

#include <ir/DataType.h>
#include <ir/InternalType.h>
#include <ir/Shape.h>
#include <ir/TypeInfo.h>
#include <ir/operation/BinaryArithmetic.h>
#include <ir/operation/FullyConnected.h>

#include <gtest/gtest.h>

namespace
{
using namespace onert;
using namespace ir;
using namespace backend;
using namespace operation;
using namespace exec;

//
// Mock backends classes
//

struct MockConfigCPU : public IConfig
{
  std::string id() override { return "cpu"; }
  bool initialize() override { return true; };
  bool supportPermutation() override { return false; }
  Layout supportLayout(const IOperation &, Layout) override { return Layout::UNKNOWN; }
  bool supportDynamicTensor() override { return false; }
  bool supportFP16() override { return false; }
};

class MockBackendContext : public BackendContext
{
public:
  using BackendContext::BackendContext;
  ITensorRegistry *genTensors() override { return nullptr; }
  FunctionMap genKernels() override { return {}; }
};

struct MockBackendCPU : public Backend
{
  std::shared_ptr<IConfig> config() const override { return std::make_shared<MockConfigCPU>(); }
  std::unique_ptr<BackendContext> newContext(ContextData &&data) const override
  {
    return std::make_unique<MockBackendContext>(this, std::move(data), nullptr);
  }
};

struct MockConfigGPU : public IConfig
{
  std::string id() override { return "gpu"; }
  bool initialize() override { return true; };
  bool supportPermutation() override { return false; }
  ir::Layout supportLayout(const ir::IOperation &, ir::Layout) override
  {
    return ir::Layout::UNKNOWN;
  }
  bool supportDynamicTensor() override { return false; }
  bool supportFP16() override { return false; }
};

struct MockBackendGPU : public Backend
{
  std::shared_ptr<IConfig> config() const override { return std::make_shared<MockConfigGPU>(); }
  std::unique_ptr<BackendContext> newContext(ContextData &&data) const override
  {
    return std::make_unique<MockBackendContext>(this, std::move(data), nullptr);
  }
};

struct MockConfigNPU : public IConfig
{
  std::string id() override { return "npu"; }
  bool initialize() override { return true; };
  bool supportPermutation() override { return false; }
  ir::Layout supportLayout(const ir::IOperation &, ir::Layout) override
  {
    return ir::Layout::UNKNOWN;
  }
  bool supportDynamicTensor() override { return false; }
  bool supportFP16() override { return false; }
};

struct MockBackendNPU : public Backend
{
  std::shared_ptr<IConfig> config() const override { return std::make_shared<MockConfigNPU>(); }
  std::unique_ptr<BackendContext> newContext(ContextData &&data) const override
  {
    return std::make_unique<MockBackendContext>(this, std::move(data), nullptr);
  }
};

//
// Constants
//

const int OPERAND_ELEMS = 268203;
const int OPERAND_SIZE = OPERAND_ELEMS * 4;
const int OPERATION_SIZE = OPERAND_SIZE * 3;

const std::string LINEAR("Linear");
const std::string DATAFLOW("Dataflow");
const std::string PARALLEL("Parallel");

//
// Helper functions
//

// Set executor through environment variable
void setExecutor(const std::string &executor) { setenv("EXECUTOR", executor.c_str(), true); }

// Set profiling mode through environment variable
void setProfilingMode(const bool value) { setenv("PROFILING_MODE", value ? "1" : "0", true); }

// Calculate operation size by addition sizes of all input and output operands
uint32_t calcOpSize(const std::shared_ptr<Graph> &graph, const OperationIndex &op_idx)
{
  uint32_t size = 0;
  const auto &op = graph->operations().at(op_idx);
  for (const auto &ind : op.getInputs() + op.getOutputs())
    size += graph->operands().at(ind).info().total_size();
  return size;
}

// Set execution operation time. This method is needed since ExecutionTime has only
// 'updateOperationExecTime' method.
void setOperationExecTime(ExecTime &et, const Backend *backend, const std::string &operation,
                          bool quant, uint32_t op_size, int64_t time)
{
  // You shouldn't set negative time with this method since nnfw JSON deserializer can't read it
  assert(time > 0);
  int64_t prev_time = et.getOperationExecTime(backend, operation, quant, op_size);
  int64_t time_to_set = prev_time == ExecTime::NOT_FOUND ? time : 2 * time - prev_time;
  et.updateOperationExecTime(backend, operation, quant, op_size, time_to_set);
  assert(et.getOperationExecTime(backend, operation, quant, op_size) == time);
}

// Set same execution time for all given backends/operations
void setOperationsExecutionTime(const std::vector<const Backend *> &backends,
                                const std::vector<std::string> &op_names,
                                const std::vector<uint32_t> &op_sizes, int64_t exec_time)
{
  assert(op_names.size() == op_sizes.size());
  ExecTime et(backends);
  for (int i = 0; i < op_names.size(); ++i)
  {
    for (const auto backend : backends)
      setOperationExecTime(et, backend, op_names[i], false, op_sizes[i], exec_time);
  }
  et.storeOperationsExecTime();
}

// Set permute time from one backend to another. This method is needed since ExecutionTime has only
// 'updatePermuteTime' method.
void setPermutationTime(ExecTime &et, const Backend *from_backend, const Backend *to_backend,
                        bool quant, uint32_t op_size, int64_t time)
{
  // You shouldn't set negative time with this method since nnfw JSON deserializer can't read it
  assert(time > 0);
  int64_t prev_time = et.getPermuteTime(from_backend, to_backend, quant, op_size);
  int64_t time_to_set = prev_time == ExecTime::NOT_FOUND ? time : 2 * time - prev_time;
  et.updatePermuteTime(from_backend, to_backend, quant, op_size, time_to_set);
  assert(et.getPermuteTime(from_backend, to_backend, quant, op_size) == time);
}

// Set same permutation time between all given backends
void setPermutationsExecutionTime(const std::vector<const Backend *> &backends,
                                  const int operand_size, const int64_t exec_time)
{
  ExecTime et(backends);
  for (const auto &backend : backends)
  {
    for (const auto other_backend : backends)
    {
      if (backend == other_backend)
        continue;
      setPermutationTime(et, backend, other_backend, false, operand_size, exec_time);
    }
  }
  et.storeOperationsExecTime();
}

//
// Functions for creating graphs
//

using OIS = OperandIndexSequence;

template <typename NodeT, typename... Types>
OperationIndex create(std::shared_ptr<Graph> graph, Types &&... args)
{
  auto op = std::make_unique<NodeT>(std::forward<Types>(args)...);
  auto op_idx = graph->addOperation(std::move(op));
  // For now in scheduler test all operations in tested graphs has same size (for simplicity)
  assert(calcOpSize(graph, op_idx) == OPERATION_SIZE);
  return op_idx;
}

// Create straight graph: Add->Sub->Mul
std::shared_ptr<Graph> createStraightGraph()
{
  auto graph = std::make_shared<Graph>();
  const TypeInfo float_op(DataType::FLOAT32);

  // Create add node
  auto add_lhs_idx = graph->addOperand(ir::Shape{OPERAND_ELEMS}, float_op);
  auto add_rhs_idx = graph->addOperand(ir::Shape{OPERAND_ELEMS}, float_op);
  auto add_out_idx = graph->addOperand(ir::Shape{OPERAND_ELEMS}, float_op);
  BinaryArithmetic::Param add_op_params{BinaryArithmetic::ArithmeticType::ADD, Activation::NONE};
  create<BinaryArithmetic>(graph, OIS{add_lhs_idx, add_rhs_idx}, OIS{add_out_idx}, add_op_params);

  // Create sub node
  auto sub_const_idx = graph->addOperand(ir::Shape{OPERAND_ELEMS}, float_op);
  auto sub_out_idx = graph->addOperand(ir::Shape{OPERAND_ELEMS}, float_op);
  BinaryArithmetic::Param sub_op_params{BinaryArithmetic::ArithmeticType::SUB, Activation::NONE};
  create<BinaryArithmetic>(graph, OIS{add_out_idx, sub_const_idx}, OIS{sub_out_idx}, sub_op_params);

  // Create mul node
  auto mul_const_idx = graph->addOperand(ir::Shape{OPERAND_ELEMS}, float_op);
  auto mul_out_idx = graph->addOperand(ir::Shape{OPERAND_ELEMS}, float_op);
  BinaryArithmetic::Param mul_op_params{BinaryArithmetic::ArithmeticType::MUL, Activation::NONE};
  create<BinaryArithmetic>(graph, OIS{sub_out_idx, mul_const_idx}, OIS{mul_out_idx}, mul_op_params);

  graph->verify();
  return graph;
}

/* Create branched graph:
 *       [Add]
 *      //   \\
 *   [Mul1]  [FC2]
 *     ||     ||
 *   [Mul2]  [FC2]
 *      \\   //
 *       [Sub]
 */
std::shared_ptr<Graph> createBranchedGraph()
{
  auto graph = std::make_shared<Graph>();
  const TypeInfo float_op(DataType::FLOAT32);

  // Create add node
  auto add_lhs_idx = graph->addOperand(ir::Shape{OPERAND_ELEMS}, float_op);
  auto add_rhs_idx = graph->addOperand(ir::Shape{OPERAND_ELEMS}, float_op);
  auto add_out_idx = graph->addOperand(ir::Shape{OPERAND_ELEMS}, float_op);
  BinaryArithmetic::Param add_op_params{BinaryArithmetic::ArithmeticType::ADD, Activation::NONE};
  create<BinaryArithmetic>(graph, OIS{add_lhs_idx, add_rhs_idx}, OIS{add_out_idx}, add_op_params);

  // Create mul1 node
  auto mul1_const_idx = graph->addOperand(ir::Shape{OPERAND_ELEMS}, float_op);
  auto mul1_out_idx = graph->addOperand(ir::Shape{OPERAND_ELEMS}, float_op);
  BinaryArithmetic::Param mul1_op_params{BinaryArithmetic::ArithmeticType::MUL, Activation::NONE};
  create<BinaryArithmetic>(graph, OIS{add_out_idx, mul1_const_idx}, OIS{mul1_out_idx},
                           mul1_op_params);

  // Create mul2 node
  auto mul2_const_idx = graph->addOperand(ir::Shape{OPERAND_ELEMS}, float_op);
  auto mul2_out_idx = graph->addOperand(ir::Shape{OPERAND_ELEMS}, float_op);
  BinaryArithmetic::Param mul2_op_params{BinaryArithmetic::ArithmeticType::MUL, Activation::NONE};
  create<BinaryArithmetic>(graph, OIS{mul1_out_idx, mul2_const_idx}, OIS{mul2_out_idx},
                           mul2_op_params);

  // Create fc1 node
  auto fc1_const_idx = graph->addOperand(ir::Shape{OPERAND_ELEMS}, float_op);
  auto fc1_out_idx = graph->addOperand(ir::Shape{OPERAND_ELEMS}, float_op);
  FullyConnected::Param fc1_op_params{Activation::NONE};
  create<FullyConnected>(graph, OIS{add_out_idx, fc1_const_idx}, OIS{fc1_out_idx}, fc1_op_params);

  // Create fc2 node
  auto fc2_const_idx = graph->addOperand(ir::Shape{OPERAND_ELEMS}, float_op);
  auto fc2_out_idx = graph->addOperand(ir::Shape{OPERAND_ELEMS}, float_op);
  FullyConnected::Param fc2_op_params{Activation::NONE};
  create<FullyConnected>(graph, OIS{fc1_out_idx, fc2_const_idx}, OIS{fc2_out_idx}, fc2_op_params);

  // Create sub node
  auto sub_out_idx = graph->addOperand(ir::Shape{OPERAND_ELEMS}, float_op);
  BinaryArithmetic::Param sub_op_params{BinaryArithmetic::ArithmeticType::SUB, Activation::NONE};
  create<BinaryArithmetic>(graph, OIS{mul2_out_idx, fc2_out_idx}, OIS{sub_out_idx}, sub_op_params);

  graph->verify();
  return graph;
}

//
// Tests setup/teardown
//

// SetUp/TearDown methods runs before/after each test and performs actions common for each test
class HESchedulerTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    // Initialize mock backends
    _cpu_backend = new MockBackendCPU();
    _gpu_backend = new MockBackendGPU();
    _npu_backend = new MockBackendNPU();
    _mock_backends = {_cpu_backend, _gpu_backend, _npu_backend};

    // Remove previous profile data if it exists
    if (!remove("exec_time.json"))
    {
      // DO NOTHING (no profile data)
    }

    // Remember original value of 'EXECUTOR' environment variable
    char *executor = std::getenv("EXECUTOR");
    _original_executor = executor == nullptr ? "" : executor;

    // Remember original value of 'PROFILING_MODE' environment variable
    char *profiling_mode = std::getenv("PROFILING_MODE");
    _original_profiling_mode = profiling_mode == nullptr ? "" : profiling_mode;
  }

  void TearDown() override
  {
    delete _cpu_backend;
    delete _gpu_backend;
    delete _npu_backend;
    EXPECT_EQ(remove("exec_time.json"), 0);
    setenv("EXECUTOR", _original_executor.c_str(), true);
    setenv("PROFILING_MODE", _original_profiling_mode.c_str(), true);
  }

  const MockBackendCPU *_cpu_backend{nullptr};
  const MockBackendGPU *_gpu_backend{nullptr};
  const MockBackendNPU *_npu_backend{nullptr};
  std::vector<const Backend *> _mock_backends;

  std::string _original_executor;
  std::string _original_profiling_mode;
};

//
// HEScheduler tests
//

class HESchedulerTestWithExecutorParam : public HESchedulerTest,
                                         public testing::WithParamInterface<std::string>
{
};

// SchedulerTestWithExecutorParam tests are parameterized with executor name and runs three times -
// one time for each executor
INSTANTIATE_TEST_SUITE_P(AllExecutors, HESchedulerTestWithExecutorParam,
                         testing::Values(LINEAR, DATAFLOW, PARALLEL));

// Test scheduler behavior for straight graph with known execution time of all nodes and permutes.
TEST_P(HESchedulerTestWithExecutorParam, straight_graph_known_exec_time)
{
  setExecutor(GetParam());

  // Prepare graph
  ir::Model model;
  auto graph(createStraightGraph());
  model.push(ir::SubgraphIndex{0}, graph);
  OperationIndex add_op_idx(0), sub_op_idx(1), mul_op_idx(2);

  // Set default execution and transfer time
  setPermutationsExecutionTime(_mock_backends, OPERAND_SIZE, 1);
  setOperationsExecutionTime(_mock_backends, {"Add", "Sub", "Mul"},
                             {OPERATION_SIZE, OPERATION_SIZE, OPERATION_SIZE}, 1e4);

  // Test 1
  // Expected behaviour: scheduler assigns different backend to each node
  {
    // For each backend reduce execution time of one node
    ExecTime et(_mock_backends);
    setOperationExecTime(et, _cpu_backend, "Add", false, OPERATION_SIZE, 1);
    setOperationExecTime(et, _gpu_backend, "Sub", false, OPERATION_SIZE, 1);
    setOperationExecTime(et, _npu_backend, "Mul", false, OPERATION_SIZE, 1);
    et.storeOperationsExecTime();

    // Test scheduler
    auto coptions = *onert::compiler::CompilerOptions::fromGlobalConfig();
    auto scheduler = compiler::HEScheduler(_mock_backends, coptions);
    const auto br = scheduler.schedule(*graph);
    ASSERT_EQ(br->getBackend(add_op_idx)->config()->id(), "cpu");
    ASSERT_EQ(br->getBackend(sub_op_idx)->config()->id(), "gpu");
    ASSERT_EQ(br->getBackend(mul_op_idx)->config()->id(), "npu");
  }

  // Test 2
  // Expected behaviour: scheduler assigns single backend to all nodes because of big transfer time
  {
    // Increase transfer time
    setPermutationsExecutionTime(_mock_backends, OPERAND_SIZE, 1e5);

    // Test scheduler
    auto coptions = *onert::compiler::CompilerOptions::fromGlobalConfig();
    auto scheduler = compiler::HEScheduler(_mock_backends, coptions);
    const auto br = scheduler.schedule(*graph);
    ASSERT_EQ(br->getBackend(add_op_idx)->config()->id(), "cpu");
    ASSERT_EQ(br->getBackend(sub_op_idx)->config()->id(), "cpu");
    ASSERT_EQ(br->getBackend(mul_op_idx)->config()->id(), "cpu");
  }
}

// Test scheduler behavior for branched graph with known execution time of all nodes and permutes
TEST_P(HESchedulerTestWithExecutorParam, branched_graph_known_exec_time)
{
  const int64_t NPU_ET = 5000;
  setExecutor(GetParam());

  // Prepare graph
  ir::Model model;
  auto graph(createBranchedGraph());
  model.push(ir::SubgraphIndex{0}, graph);
  OperationIndex add_op_idx(0), mul1_op_idx(1), mul2_op_idx(2), fc1_op_idx(3), fc2_op_idx(4),
    sub_op_idx(5);

  // Set default execution and transfer time
  setPermutationsExecutionTime(_mock_backends, OPERAND_SIZE, 1000);
  setOperationsExecutionTime(_mock_backends, {"Add", "Sub", "Mul", "FullyConnected"},
                             {OPERATION_SIZE, OPERATION_SIZE, OPERATION_SIZE, OPERATION_SIZE}, 1e4);

  // Test 1
  // Expected behaviour: for dataflow and linear executors scheduler assigns fastest backend to all
  // nodes, in case of parallel executor scheduler assigns different backends to branches.
  {
    // Reduce execution time
    ExecTime et(_mock_backends);
    setOperationExecTime(et, _npu_backend, "Add", false, OPERATION_SIZE, NPU_ET);
    setOperationExecTime(et, _npu_backend, "Mul", false, OPERATION_SIZE, NPU_ET);
    setOperationExecTime(et, _npu_backend, "Sub", false, OPERATION_SIZE, NPU_ET);
    setOperationExecTime(et, _npu_backend, "FullyConnected", false, OPERATION_SIZE, NPU_ET);
    setOperationExecTime(et, _gpu_backend, "Mul", false, OPERATION_SIZE, NPU_ET + 1000);
    setOperationExecTime(et, _gpu_backend, "FullyConnected", false, OPERATION_SIZE, NPU_ET + 1000);
    et.storeOperationsExecTime();

    // Test scheduler
    auto coptions = *onert::compiler::CompilerOptions::fromGlobalConfig();
    auto scheduler = compiler::HEScheduler(_mock_backends, coptions);
    const auto br = scheduler.schedule(*graph);

    std::string branch1_expected_backend("npu"), branch2_expected_backend("npu");
    if (GetParam() == PARALLEL)
    {
      branch1_expected_backend =
        br->getBackend(mul1_op_idx)->config()->id() == "npu" ? "npu" : "gpu";
      branch2_expected_backend = branch1_expected_backend == "npu" ? "gpu" : "npu";
    }

    ASSERT_EQ(br->getBackend(add_op_idx)->config()->id(), "npu");
    ASSERT_EQ(br->getBackend(mul1_op_idx)->config()->id(), branch1_expected_backend);
    ASSERT_EQ(br->getBackend(mul2_op_idx)->config()->id(), branch1_expected_backend);
    ASSERT_EQ(br->getBackend(fc1_op_idx)->config()->id(), branch2_expected_backend);
    ASSERT_EQ(br->getBackend(fc2_op_idx)->config()->id(), branch2_expected_backend);
    ASSERT_EQ(br->getBackend(sub_op_idx)->config()->id(), "npu");
  }

  // Test 2
  // Expected behaviour: scheduler assigns single backend to all nodes
  {
    // Increase execution time for GPU backend
    ExecTime et(_mock_backends);
    /* for parallel executor: set a time, that is larger than sum_of_other_branches_nodes_cnt *
     * npu_exec_time so that npu is prefered: the ith branch will wait for npu until it finishes the
     * [0;i-1] branches nodes in DFS order. In each branch it goes deep intul doesn't encounter
     * branching or scheduler assigns another backend to a node*/
    setOperationExecTime(et, _gpu_backend, "Mul", false, OPERATION_SIZE, NPU_ET * 3 + 1);
    setOperationExecTime(et, _gpu_backend, "FullyConnected", false, OPERATION_SIZE, NPU_ET * 3 + 1);
    et.storeOperationsExecTime();

    // Test scheduler
    auto coptions = *onert::compiler::CompilerOptions::fromGlobalConfig();
    auto scheduler = compiler::HEScheduler(_mock_backends, coptions);
    const auto br = scheduler.schedule(*graph);
    ASSERT_EQ(br->getBackend(add_op_idx)->config()->id(), "npu");
    ASSERT_EQ(br->getBackend(mul1_op_idx)->config()->id(), "npu");
    ASSERT_EQ(br->getBackend(mul2_op_idx)->config()->id(), "npu");
    ASSERT_EQ(br->getBackend(fc1_op_idx)->config()->id(), "npu");
    ASSERT_EQ(br->getBackend(fc2_op_idx)->config()->id(), "npu");
    ASSERT_EQ(br->getBackend(sub_op_idx)->config()->id(), "npu");
  }
}

// Test scheduler behavior for branched graph and enabled profiling mode
TEST_F(HESchedulerTest, branched_graph_profiling_mode)
{
  const int ET = 1e5;

  // Turn on profiling mode
  setProfilingMode(true);
  setExecutor(DATAFLOW);

  // Prepare graph
  ir::Model model;
  auto graph(createBranchedGraph());
  model.push(ir::SubgraphIndex{0}, graph);
  OperationIndex add_op_idx(0), mul1_op_idx(1), mul2_op_idx(2), fc1_op_idx(3), fc2_op_idx(4),
    sub_op_idx(5);

  // Test 1
  // Expected behaviour: scheduler assigns backends to nodes with unknown execution time
  {
    // Set execution time for all backends/nodes except for cpu/Sub, npu/Mul, gpu/FC
    ExecTime et(_mock_backends);
    setOperationExecTime(et, _cpu_backend, "Add", false, OPERATION_SIZE, ET);
    setOperationExecTime(et, _cpu_backend, "Mul", false, OPERATION_SIZE, ET + 1);
    setOperationExecTime(et, _cpu_backend, "FullyConnected", false, OPERATION_SIZE, ET);
    setOperationExecTime(et, _npu_backend, "Add", false, OPERATION_SIZE, ET);
    setOperationExecTime(et, _npu_backend, "FullyConnected", false, OPERATION_SIZE, ET);
    setOperationExecTime(et, _npu_backend, "Sub", false, OPERATION_SIZE, ET);
    setOperationExecTime(et, _gpu_backend, "Add", false, OPERATION_SIZE, ET);
    setOperationExecTime(et, _gpu_backend, "Mul", false, OPERATION_SIZE, ET + 1);
    setOperationExecTime(et, _gpu_backend, "Sub", false, OPERATION_SIZE, ET);
    et.storeOperationsExecTime();

    // Test scheduler
    auto coptions = *onert::compiler::CompilerOptions::fromGlobalConfig();
    auto scheduler = compiler::HEScheduler(_mock_backends, coptions);
    const auto br = scheduler.schedule(*graph);
    ASSERT_EQ(br->getBackend(mul1_op_idx)->config()->id(), "npu");
    ASSERT_EQ(br->getBackend(mul2_op_idx)->config()->id(), "npu");
    ASSERT_EQ(br->getBackend(fc1_op_idx)->config()->id(), "gpu");
    ASSERT_EQ(br->getBackend(fc2_op_idx)->config()->id(), "gpu");
    ASSERT_EQ(br->getBackend(sub_op_idx)->config()->id(), "cpu");
  }

  // Test 2
  // Expected behaviour: scheduler shuffling backends, so different backends are assigned to
  // neighbor nodes
  {
    // Set execution time for rest backends/nodes (cpu/Sub, npu/Mul, gpu/FC)
    ExecTime et(_mock_backends);
    setOperationExecTime(et, _cpu_backend, "Sub", false, OPERATION_SIZE, ET);
    setOperationExecTime(et, _npu_backend, "Mul", false, OPERATION_SIZE, ET + 1);
    setOperationExecTime(et, _gpu_backend, "FullyConnected", false, OPERATION_SIZE, ET);
    et.storeOperationsExecTime();

    // Test scheduler
    auto coptions = *onert::compiler::CompilerOptions::fromGlobalConfig();
    auto scheduler = compiler::HEScheduler(_mock_backends, coptions);
    const auto br = scheduler.schedule(*graph);
    ASSERT_NE(br->getBackend(add_op_idx)->config()->id(),
              br->getBackend(mul1_op_idx)->config()->id());
    ASSERT_NE(br->getBackend(add_op_idx)->config()->id(),
              br->getBackend(fc1_op_idx)->config()->id());
    ASSERT_NE(br->getBackend(mul1_op_idx)->config()->id(),
              br->getBackend(mul2_op_idx)->config()->id());
    ASSERT_NE(br->getBackend(fc1_op_idx)->config()->id(),
              br->getBackend(fc2_op_idx)->config()->id());
    ASSERT_NE(br->getBackend(mul2_op_idx)->config()->id(),
              br->getBackend(sub_op_idx)->config()->id());
    ASSERT_NE(br->getBackend(fc2_op_idx)->config()->id(),
              br->getBackend(sub_op_idx)->config()->id());
  }
}

// TODO: Add tests with unknown execution and permutation time

} // unnamed namespace
