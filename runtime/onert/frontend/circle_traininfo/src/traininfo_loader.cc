/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "traininfo_loader.h"
#include "circle_traininfo_generated.h"
#include "flatbuffers/flatbuffers.h"

namespace onert
{
namespace traininfo_loader
{

namespace
{

template <typename T>
std::unique_ptr<ir::train::OptimizerOption>
getOptimizerOption(const circle::ModelTraining *circle_model);

template <>
std::unique_ptr<ir::train::OptimizerOption>
getOptimizerOption<circle::SGDOptions>(const circle::ModelTraining *circle_model)
{
  assert(circle_model->optimizer_opt_type() ==
         circle::OptimizerOptions::OptimizerOptions_SGDOptions);

  const circle::SGDOptions *sgd_option = circle_model->optimizer_opt_as_SGDOptions();
  std::unique_ptr<ir::train::SGDOption> ir_sgd_option(new ir::train::SGDOption);
  {
    ir_sgd_option->learning_rate = sgd_option->learning_rate();
  }
  return std::move(ir_sgd_option);
}

template <>
std::unique_ptr<ir::train::OptimizerOption>
getOptimizerOption<circle::AdamOptions>(const circle::ModelTraining *circle_model)
{
  assert(circle_model->optimizer_opt_type() ==
         circle::OptimizerOptions::OptimizerOptions_AdamOptions);

  const circle::AdamOptions *adam_option = circle_model->optimizer_opt_as_AdamOptions();
  std::unique_ptr<ir::train::AGDOption> ir_adam_option(new ir::train::AGDOption);
  {
    ir_adam_option->learning_rate = adam_option->learning_rate();
    ir_adam_option->beta_1 = adam_option->beta_1();
    ir_adam_option->beta_2 = adam_option->beta_2();
    ir_adam_option->epsilon = adam_option->epsilon();
  }
  return std::move(ir_adam_option);
}

ir::train::OptimizerInfo getOptimizerInfo(const circle::ModelTraining *circle_model)
{
  ir::train::OptimizerInfo ir_optimizer_info;

  const circle::Optimizer circle_optimizer = circle_model->optimizer();
  switch (circle_optimizer)
  {
    case circle::Optimizer_SGD:
      ir_optimizer_info.optim_code = ir::train::OptimizerCode::SGD;
      ir_optimizer_info.optim_option = getOptimizerOption<circle::SGDOptions>(circle_model);
      // TODO deprecate
      ir_optimizer_info.learning_rate =
        dynamic_cast<const ir::train::SGDOption *>(ir_optimizer_info.optim_option.get())
          ->learning_rate;
      break;
    case circle::Optimizer_ADAM:
      ir_optimizer_info.optim_code = ir::train::OptimizerCode::Adam;
      ir_optimizer_info.optim_option = getOptimizerOption<circle::AdamOptions>(circle_model);
      // TODO deprecate
      ir_optimizer_info.learning_rate =
        dynamic_cast<const ir::train::AGDOption *>(ir_optimizer_info.optim_option.get())
          ->learning_rate;
      break;
    default:
      throw std::runtime_error("unknown optimzer");
  }
  return ir_optimizer_info;
}

ir::train::LossInfo getLossInfo(const circle::ModelTraining *circle_model)
{
  ir::train::LossInfo ir_loss_info;

  const circle::LossFn circle_lossfn = circle_model->lossfn();

  // TODO add circle::LossFnOption parsing
  switch (circle_lossfn)
  {
    case circle::LossFn::LossFn_CATEGORICAL_CROSSENTROPY:
      ir_loss_info.type = ir::operation::Loss::Type::CATEGORICAL_CROSSENTROPY;
      break;
    case circle::LossFn::LossFn_SPARSE_CATEGORICAL_CROSSENTROPY:
      // TODO Enable this case after 'sparse_cateogrial_crossentropy' implemented
      throw std::runtime_error{"not supported yet"};
    case circle::LossFn::LossFn_MEAN_SQUARED_ERROR:
      ir_loss_info.type = ir::operation::Loss::Type::MEAN_SQUARED_ERROR;
      break;
    default:
      throw std::runtime_error{"unknown loss function"};
  }
  return ir_loss_info;
}
} // namespace

std::unique_ptr<ir::train::TrainingInfo> loadTrainInfo(const uint8_t *buffer, const size_t size)
{
  // verify
  flatbuffers::Verifier v(buffer, size);
  bool verify_result = circle::VerifyModelTrainingBuffer(v);
  if (verify_result == false)
  {
    throw std::runtime_error{"TrainingInfo is something wrong"};
  }

  const circle::ModelTraining *circle_model = circle::GetModelTraining((void *)buffer);

  std::unique_ptr<ir::train::TrainingInfo> ir_model(new ir::train::TrainingInfo);
  {
    ir_model->setBatchSize(circle_model->batch_size());
    ir_model->setOptimizerInfo(getOptimizerInfo(circle_model));
    ir_model->setLossInfo(getLossInfo(circle_model));
    ir_model->setEpoch(circle_model->epochs());
  }
  return std::move(ir_model);
}

} // namespace traininfo_loader
} // namespace onert
