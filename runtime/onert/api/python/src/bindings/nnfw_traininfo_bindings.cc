/*
 * Copyright (c) 2025 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "nnfw_traininfo_bindings.h"

#include "nnfw_api_wrapper.h"

namespace py = pybind11;

using namespace onert::api::python;

// Declare binding train enums
void bind_nnfw_train_enums(py::module_ &m)
{
  // Bind NNFW_TRAIN_LOSS
  py::enum_<NNFW_TRAIN_LOSS>(m, "loss", py::module_local())
    .value("UNDEFINED", NNFW_TRAIN_LOSS_UNDEFINED)
    .value("MEAN_SQUARED_ERROR", NNFW_TRAIN_LOSS_MEAN_SQUARED_ERROR)
    .value("CATEGORICAL_CROSSENTROPY", NNFW_TRAIN_LOSS_CATEGORICAL_CROSSENTROPY);

  // Bind NNFW_TRAIN_LOSS_REDUCTION
  py::enum_<NNFW_TRAIN_LOSS_REDUCTION>(m, "loss_reduction", py::module_local())
    .value("UNDEFINED", NNFW_TRAIN_LOSS_REDUCTION_UNDEFINED)
    .value("SUM_OVER_BATCH_SIZE", NNFW_TRAIN_LOSS_REDUCTION_SUM_OVER_BATCH_SIZE)
    .value("SUM", NNFW_TRAIN_LOSS_REDUCTION_SUM);

  // Bind NNFW_TRAIN_OPTIMIZER
  py::enum_<NNFW_TRAIN_OPTIMIZER>(m, "optimizer", py::module_local())
    .value("UNDEFINED", NNFW_TRAIN_OPTIMIZER_UNDEFINED)
    .value("SGD", NNFW_TRAIN_OPTIMIZER_SGD)
    .value("ADAM", NNFW_TRAIN_OPTIMIZER_ADAM);

  // Bind NNFW_TRAIN_NUM_OF_TRAINABLE_OPS_SPECIAL_VALUES
  py::enum_<NNFW_TRAIN_NUM_OF_TRAINABLE_OPS_SPECIAL_VALUES>(m, "trainable_ops", py::module_local())
    .value("INCORRECT_STATE", NNFW_TRAIN_TRAINABLE_INCORRECT_STATE)
    .value("ALL", NNFW_TRAIN_TRAINABLE_ALL)
    .value("NONE", NNFW_TRAIN_TRAINABLE_NONE);
}

// Declare binding loss info
void bind_nnfw_loss_info(py::module_ &m)
{
  py::class_<nnfw_loss_info>(m, "lossinfo", py::module_local())
    .def(py::init<>()) // Default constructor
    .def_readwrite("loss", &nnfw_loss_info::loss, "Loss type")
    .def_readwrite("reduction_type", &nnfw_loss_info::reduction_type, "Reduction type");
}

// Declare binding train info
void bind_nnfw_train_info(py::module_ &m)
{
  py::class_<nnfw_train_info>(m, "traininfo", py::module_local())
    .def(py::init<>()) // Default constructor
    .def_readwrite("learning_rate", &nnfw_train_info::learning_rate, "Learning rate")
    .def_readwrite("batch_size", &nnfw_train_info::batch_size, "Batch size")
    .def_readwrite("loss_info", &nnfw_train_info::loss_info, "Loss information")
    .def_readwrite("opt", &nnfw_train_info::opt, "Optimizer type")
    .def_readwrite("num_of_trainable_ops", &nnfw_train_info::num_of_trainable_ops,
                   "Number of trainable operations");
}
