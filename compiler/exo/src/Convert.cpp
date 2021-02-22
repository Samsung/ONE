/*
 * Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "Convert.h"

#include "Conversions.h"
#include "Pass/ShapeInferencePass.h"
#include "Pass/TypeInferencePass.h"
#include "ProgressReporter.h"
#include "Knob.h"

#include <loco.h>
#include <loco/Service/ShapeInference.h>
#include <loco/Service/CanonicalShapeInferenceRule.h>
#include <loco/Service/TypeInference.h>

#include <logo/SimplifyDomainConversionPass.h>
#include <logo/RemoveDeadNodePass.h>
#include <logo/RemoveForwardNodePass.h>

#include <logo/Phase.h>
#include <memory>

namespace exo
{

void convert_to_TFLNodes(loco::Graph *graph)
{
  // run Shape and Type inference must be run before conversion
  loco::CanonicalShapeInferenceRule shape_rule;
  loco::apply(&shape_rule).to(graph);

  loco::CanonicalTypeInferenceRule type_rule;
  loco::apply(&type_rule).to(graph);

  logo::Phase phase;
  {
    // prepare type and shape before conversion
    phase.emplace_back(std::make_unique<TypeInferencePass>());
    phase.emplace_back(std::make_unique<ShapeInferencePass>());

    // Add converters for canonical nodes. Note: Not all loco canonical nodes are listed.
    phase.emplace_back(std::make_unique<AvgPool2DConverter>());
    phase.emplace_back(std::make_unique<ConstGenConverter>());
    phase.emplace_back(std::make_unique<Conv2DConverter>());
    phase.emplace_back(std::make_unique<DepthwiseConv2DConverter>());
    // TODO loco::DepthwiseFilterEncode
    phase.emplace_back(std::make_unique<EltwiseAddConverter>());
    phase.emplace_back(std::make_unique<EltwiseDivConverter>());
    phase.emplace_back(std::make_unique<EltwiseMaxConverter>());
    phase.emplace_back(std::make_unique<EltwiseMulConverter>());
    phase.emplace_back(std::make_unique<EltwiseSqrtConverter>());
    phase.emplace_back(std::make_unique<EltwiseSubConverter>());
    phase.emplace_back(std::make_unique<FeatureBiasAddConverter>());
    // TODO loco::FixedReshape
    phase.emplace_back(std::make_unique<MatMulConverter>());
    phase.emplace_back(std::make_unique<MaxPool2DConverter>());
    phase.emplace_back(std::make_unique<ReluConverter>());
    phase.emplace_back(std::make_unique<Relu6Converter>());
    // TODO loco::Tanh
    phase.emplace_back(std::make_unique<TensorConcatConverter>());
    // TODO loco::TensorBiasAdd
    phase.emplace_back(std::make_unique<TensorBroadcastConverter>());
    phase.emplace_back(std::make_unique<TensorReduceConverter>());
    // TODO loco::TensorSoftmax
    phase.emplace_back(std::make_unique<TensorTransposeConverter>());
    phase.emplace_back(std::make_unique<TransposedConv2DConverter>());

    // Add optimization below
    phase.emplace_back(std::make_unique<logo::SimplifyDomainConversionPass>());
    phase.emplace_back(std::make_unique<logo::RemoveForwardNodePass>());
    phase.emplace_back(std::make_unique<logo::RemoveDeadNodePass>());
  }

  logo::PhaseRunner<logo::PhaseStrategy::Restart> phase_runner{graph};

  ProgressReporter prog(graph, logo::PhaseStrategy::Restart);
  phase_runner.attach(&prog);
  phase_runner.run(phase);

  // TODO Assert if all canonical nodes are converted to TFL node
}

} // namespace exo
