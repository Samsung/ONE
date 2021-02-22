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

#include "enco/Backend.h"

#include "IRValidator.h"

#include "Session.h"
#include "Pipeline.h"

#include "Code.h"
#include "AsmCode.h"
#include "CppCode.h"

#include "Transforms/Duplicate.h"
#include "Transforms/FeatureUnification.h"
#include "Transforms/AvgPoolLowering.h"
#include "Transforms/IntrinsicSelection.h"
#include "Transforms/DataLayoutConversion.h"
#include "Transforms/IndirectCopyElimination.h"
#include "Transforms/IdenticalObjectReduction.h"
#include "Transforms/DuplicatedObjectReduction.h"
#include "Transforms/DeadObjectElimination.h"
#include "Transforms/ConstantFolding.h"
#include "Transforms/CopyLowering.h"
#include "Transforms/ConcatLowering.h"
#include "Transforms/FreeInstrElimination.h"
#include "Transforms/FreeOpElimination.h"
#include "Transforms/DeadBagElimination.h"
#include "Transforms/Optimizations.h"
#include "Transforms/Split.h"
#include "Transforms/GlobalDataGeneration.h"

#include <memory>
#include <stdexcept>
#include <iostream>
#include <fstream>

using std::make_unique;
using namespace enco;

namespace
{

// has_inout_bag(m) returns true if there is a pair of coco::Input and coco::Output that share
// the same bag as their backing storage
inline bool has_inout_bag(const coco::Module *m)
{
  for (uint32_t n = 0; n < m->entity()->bag()->size(); ++n)
  {
    auto bag = m->entity()->bag()->at(n);

    if (bag->isInput() && bag->isOutput())
    {
      return true;
    }
  }
  return false;
}

class BackendImpl final : public enco::Backend
{
public:
  BackendImpl(const std::string &prefix) : _prefix{prefix}
  {
    // DO NOTHING
  }

public:
  void compile(coco::Module *m, coco::Data *d) override;

private:
  std::string _prefix;
};

void BackendImpl::compile(coco::Module *m, coco::Data *d)
{
  auto sess = make_session(m, d);

  // validate if IR from frontend is correct
  assert(validate(code(sess)));

  enco::Pipeline pipeline;

  // Configure pipeline

  // As explained below, the current implementation does not work if there is a pair of input/output
  // that share the same bag as their underlying bag.
  //
  // BagDuplicationPass creates a copy of such bags in order to eliminate such a pair.
  pipeline.append(make_unique<BagDuplicationPass>());
  pipeline.append(make_unique<FeatureUnificationPass>());
  pipeline.append(make_unique<AvgPoolLoweringPass>());
  pipeline.append(make_unique<IntrinsicSelectionPass>());
  // Insert data ordering if necessary
  pipeline.append(make_unique<DataLayoutConversionPass>());
  pipeline.append(make_unique<IndirectCopyEliminationPass>());
  pipeline.append(make_unique<IdenticalObjectReductionPass>());
  pipeline.append(make_unique<DuplicatedObjectReductionPass>());
  pipeline.append(make_unique<ConstantFoldingPass>());
  // Eliminate dead object
  //
  // NOTE Dead Object Elimination (DOE) is performed before Copy lowering
  //      in order to reduce compilation overhead.
  pipeline.append(make_unique<DeadObjectEliminationPass>());
  // Lower Copy as Shuffle
  pipeline.append(make_unique<CopyLoweringPass>());
  // Lower ConcatF as Shuffle if it is not delegated to NNAPI yet
  pipeline.append(make_unique<ConcatLoweringPass>());
  pipeline.append(make_unique<BypassGenerationPass>());
  pipeline.append(make_unique<FreeInstrEliminationPass>());
  // NOTE Free Op Elimination should be applied after Free Instr Elimination
  //      - Free Instr Elimination may generate additional free Op(s)
  pipeline.append(make_unique<FreeOpEliminationPass>());
  pipeline.append(make_unique<DeadBagEliminationPass>());
  // Split instructions into a set of phases (each block serves as a phase)
  pipeline.append(make_unique<PhaseConstructionPass>());

  // Apply transforms in the pipeline
  for (uint32_t n = 0; n < pipeline.size(); ++n)
  {
    const auto &pass = pipeline.at(n);

    pass.run(sess);
  }

  // The current implementation will assign memory region for each bag as follows:
  //   Bind input bag to the region provided by Network_input_bind
  //   Bind output bag to the region provided by Network_output_bind
  //   Bind intermediate bag to the region allocated during execution
  //
  // Note that this scheme does not work if there is a pair of input/output
  // that share the same bag as their underlying bag
  assert(!has_inout_bag(code(sess)->module()));

  const std::string data_var = "data";
  const std::string data_filename = _prefix + ".bin";

  // Generate 'bin' file
  {
    std::ofstream ofs{data_filename, std::ios::binary};
    generate_global_data(ofs, code(sess));
  }

  // Generate 'embed.S' file
  {
    std::ofstream ofs{_prefix + ".embed.S"};
    ofs << AsmCode{data_filename, data_var};
  }

  // TODO Run various transforms over enco::Code

  std::ofstream ofs{_prefix + ".cpp"};
  ofs << CppCode{data_var, code(sess)} << std::endl;
}

} // namespace

#include <iostream>

std::unique_ptr<enco::Backend> make_backend(const cmdline::View &cmdline)
{
  return make_unique<::BackendImpl>(cmdline.at(0));
}
