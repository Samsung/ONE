/*
 * Copyright (c) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "pass/PassData.h"

#include "passes/transformations/DataFormatSwitcher.h"
#include "passes/transformations/LowerConv2D.h"

#include "backends/interpreter/InterpreterBackend.h"
#include "backends/soft_backend/CPPGenerator.h"
#include "passes/dot_dumper/DumperPass.h"
#include "backends/acl_soft_backend/AclCppGenerator.h"

#include "passes/optimizations/CombineTransposes.h"
#include "passes/optimizations/ConstantFoldTranspose.h"
#include "passes/optimizations/DeadCodeElimination.h"
#include "passes/optimizations/FuseArithmeticOps.h"
#include "passes/optimizations/SinkRelu.h"
#include "passes/optimizations/SinkTranspose.h"

#include "support/CommandLine.h"
#include "Definitions.h"
#include "Options.h"
#include "Driver.h"

#ifdef NNC_FRONTEND_CAFFE2_ENABLED
#include <caffe2_importer.h>
#endif // NNC_FRONTEND_CAFFE2_ENABLED
#ifdef NNC_FRONTEND_CAFFE_ENABLED
#include <caffe_importer.h>
#endif // NNC_FRONTEND_CAFFE_ENABLED
#ifdef NNC_FRONTEND_TFLITE_ENABLED
#include <tflite_importer.h>
#endif // NNC_FRONTEND_TFLITE_ENABLED
#ifdef NNC_FRONTEND_ONNX_ENABLED
#include <ONNXImporterImpl.h>
#endif // NNC_FRONTEND_ONNX_ENABLED

#include <memory>

namespace nnc
{

static std::string getFrontendOptionsString()
{
  std::string res;

  if (!cli::caffeFrontend.isDisabled())
    res += "'" + cli::caffeFrontend.getNames()[0] + "' ";

  if (!cli::caffe2Frontend.isDisabled())
    res += "'" + cli::caffe2Frontend.getNames()[0] + "' ";

  if (!cli::onnxFrontend.isDisabled())
    res += "'" + cli::onnxFrontend.getNames()[0] + "' ";

  if (!cli::tflFrontend.isDisabled())
    res += "'" + cli::tflFrontend.getNames()[0] + "'";

  return res;
}

static std::unique_ptr<mir::Graph> importModel()
{
  // For bool, the value false is converted to zero and the value true is converted to one
  if (cli::caffeFrontend + cli::caffe2Frontend + cli::tflFrontend + cli::onnxFrontend != 1)
    throw DriverException("One and only one of the following options are allowed and have to be set"
                          "in the same time: " +
                          getFrontendOptionsString());

  if (cli::caffeFrontend)
  {
#ifdef NNC_FRONTEND_CAFFE_ENABLED
    return mir_caffe::loadModel(cli::inputFile.getRawValue());
#endif // NNC_FRONTEND_CAFFE_ENABLED
  }
  else if (cli::caffe2Frontend)
  {
#ifdef NNC_FRONTEND_CAFFE2_ENABLED
    // FIXME: caffe2 input shapes are not provided by model and must be set from cli
    // current 'inputShapes' could provide only one shape, while model could has several inputs
    return mir_caffe2::loadModel(cli::inputFile.getRawValue(), cli::initNet.getRawValue(),
                                 {cli::inputShapes.getRawValue()});
#endif // NNC_FRONTEND_CAFFE2_ENABLED
  }
  else if (cli::onnxFrontend)
  {
#ifdef NNC_FRONTEND_ONNX_ENABLED
    return mir_onnx::loadModel(cli::inputFile.getRawValue());
#endif // NNC_FRONTEND_ONNX_ENABLED
  }
  else if (cli::tflFrontend)
  {
#ifdef NNC_FRONTEND_TFLITE_ENABLED
    return mir_tflite::loadModel(cli::inputFile.getRawValue());
#endif // NNC_FRONTEND_TFLITE_ENABLED
  }

  assert(false);
  return nullptr;
}

static void backend(mir::Graph *graph)
{
  if (cli::target == NNC_TARGET_ARM_CPP || cli::target == NNC_TARGET_X86_CPP)
  {
    CPPCodeGenerator(cli::artifactDir, cli::artifactName).run(graph);
  }
  else if (cli::target == NNC_TARGET_ARM_GPU_CPP)
  {
    AclCppCodeGenerator(cli::artifactDir, cli::artifactName).run(graph);
  }
  else if (cli::target == NNC_TARGET_INTERPRETER)
  {
    InterpreterBackend(cli::interInputDataDir, cli::artifactDir).run(graph);
  }
  else
  {
    assert(false && "invalid option value");
  }
}

/**
 * @brief run all registered passes
 * @throw PassException, if errors occured
 */
void Driver::runPasses()
{
  auto graph = importModel();
  PassData pass_data(graph.get());

  for (const auto &pass : _passManager.getPasses())
  {
    pass_data = pass->run(pass_data);
    if (cli::dumpGraph && static_cast<mir::Graph *>(pass_data))
    {
      DumperPass d(pass->getName());
      d.run(pass_data);
    }
  }

  backend(pass_data);

  // NOTE. Now we destroy data of all passes when PassManager is destroyed.
  // In future to reduce memory consumption we can destory it when passes are being performed

} // runPasses

/**
 * @brief Register backend specific passes
 * @throw DriverException if errors occurred
 */
void Driver::registerBackendSpecificPasses()
{
  std::unique_ptr<Pass> data_format_pass;

  if (cli::target == NNC_TARGET_ARM_CPP || cli::target == NNC_TARGET_X86_CPP)
  {
    _passManager.registerPass(std::make_unique<LowerConv2D>());
    _passManager.registerPass(std::make_unique<DataFormatSwitcher>(mir::DataFormat::NHWC));
  }
  else if (cli::target == NNC_TARGET_ARM_GPU_CPP)
  {
    _passManager.registerPass(std::make_unique<LowerConv2D>());
    _passManager.registerPass(std::make_unique<ConstantFoldTranspose>());
    // TODO Change to DataFormat::NCHW when fix it in ACL
    _passManager.registerPass(std::make_unique<DataFormatSwitcher>(mir::DataFormat::NHWC));
  }
  else if (cli::target == NNC_TARGET_INTERPRETER)
  {
    _passManager.registerPass(std::make_unique<DataFormatSwitcher>(mir::DataFormat::NHWC));
  }
  else
  {
    assert(false && "invalid option value");
  }
}

void Driver::registerOptimizationPass()
{
  if (cli::doOptimizationPass)
  {
    // TODO: maybe we should start managing the optimizations more intelligently?
    _passManager.registerPass(std::unique_ptr<Pass>(new CombineTransposes()));
    _passManager.registerPass(std::unique_ptr<Pass>(new SinkTranspose()));
    _passManager.registerPass(std::unique_ptr<Pass>(new SinkRelu()));
#if 0
    // TODO Support broadcasting.
    _passManager.registerPass(std::unique_ptr<Pass>(new FuseArithmeticOps()));
#endif
    _passManager.registerPass(std::unique_ptr<Pass>(new DeadCodeElimination()));
  }
} // registerOptimizationPass

void Driver::runDriver()
{
  registerOptimizationPass();
  registerBackendSpecificPasses();

  runPasses();
}

} // namespace nnc
