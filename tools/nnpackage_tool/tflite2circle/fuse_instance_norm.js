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

'use strict'

// read json and parse
const fs = require('fs')
let inputfile = "./03_2k.json"
if (process.argv.length == 3)
  inputfile = process.argv[2]
let raw = fs.readFileSync(inputfile)
let model = JSON.parse(raw)

// 0. prepare shortcut variables with object destructuring
const { operators, tensors } = model.subgraphs[0]

//--------------------------------------------------------------------------
// 0. construct infra

// List : opcode index (number) => op name (string)
let opcodeIdxToOpName = []
for (const opcode of model.operator_codes) {
  opcodeIdxToOpName.push(opcode.builtin_code)
}

// List: tensor index (number) => producing operator's index (number)
// assume there is only one op that produces given output tensor.
let defOp = []
for (let i = 0; i < operators.length; ++i) {
  let op = operators[i]
  if (op.outputs.length !== 1) {
    console.debug("Assumption failed. Multiple output operator exists.")
    process.exit(-1);
  }
  defOp[op.outputs[0]] = i
}

// List: tensor index (number) => consuming operator indices (list of number)
// Note that there may be multiple consumer ops for a given tensor index
let useOps = []
for (let i = 0; i < operators.length; ++i) {
  let op = operators[i]
  for (let inTensorIdx of op.inputs) {
    if (useOps[inTensorIdx])
      useOps[inTensorIdx].push(i)
    else
      useOps[inTensorIdx] = [ i ]
  }
}

// return operator that defines the given tensor index
function getDefOp(iTensor) {
  return defOp[iTensor] === undefined ? undefined : operators[defOp[iTensor]]
}

function getUseOps(iTensor) {
  if (useOps[iTensor] === undefined)
    return undefined
  let ret = []
  for (let i of useOps[iTensor])
    ret.push(operators[i])
  return ret
}

function opeq(op, str) {
  return op === undefined ? undefined : opcodeIdxToOpName[op.opcode_index] === str
}

function hasUndefined() {
  for (let arg of arguments)
    if (arg === undefined)
      return true
  return false
}

//--------------------------------------------------------------------------
// find SquaredDifference as starting point
let squaredDifferenceIdxList = []
for (let i = 0; i < operators.length; ++i) {
  if (opeq(operators[i], "SQUARED_DIFFERENCE"))
    squaredDifferenceIdxList.push(i)
}

let instanceNormList = [ ]
for (let idx of squaredDifferenceIdxList) {
  const sqd1 = operators[idx]
  const findMean0AndInstanceNormInputTensor = function(sqd1) {
    let mean0, iInstanceNormInputTensor
    for (let i = 0; i < sqd1.inputs.length; ++i) {
      let op = getDefOp(sqd1.inputs[i])
      if (opeq(op, "MEAN")) {
        mean0 = op
        // let's check one of inputs are instance_norm
        // the other input is axis of mean operator.
        for (let j = 0; j < mean0.inputs.length; ++j) {
          // 1 - i means the other input of squared_difference.
          if (mean0.inputs[j] === sqd1.inputs[1 - i]) {
            iInstanceNormInputTensor = mean0.inputs[j]
          }
          if (!hasUndefined(iInstanceNormInputTensor)) break // found instance_norm
        }
      }
      if (!hasUndefined(mean0, iInstanceNormInputTensor)) break
    }
    return [mean0, iInstanceNormInputTensor]
  }
  const [mean0, iInstanceNormInputTensor] = findMean0AndInstanceNormInputTensor(sqd1)
  if (hasUndefined(mean0, iInstanceNormInputTensor)) continue

  const findConsumer = function(op, expectedOp) {
    let ops = getUseOps(op.outputs[0])
    if (ops === undefined || ops.length !== 1 || !opeq(ops[0], expectedOp))
      return undefined
    return ops[0]
  }
  const mean2 = findConsumer(sqd1, "MEAN")
  if (hasUndefined(mean2)) continue

  const add3 = findConsumer(mean2, "ADD")
  if (hasUndefined(add3)) continue

  const isScalar  = function(tsr) { return tsr.shape.length === 0 }
  const is1D      = function(tsr) { return tsr.shape.length === 1 }
  const isFloat32 = function(tsr) { return tsr.type === "FLOAT32" }
  const asFloat32 = function(arr) { return new Float32Array(new Uint8Array(arr).buffer)[0]; }
  const getFloatScalarValueFromInputsOf = function(op) {
    for (let i of op.inputs) {
      if (isScalar(tensors[i]) && isFloat32(tensors[i])) {
        let buf = model.buffers[tensors[i].buffer]
        if (buf.data && buf.data.length === 4)
          return asFloat32(buf.data)
      }
    }
    return undefined
  }
  const epsilon = getFloatScalarValueFromInputsOf(add3)
  if (hasUndefined(epsilon)) continue

  const rsqrt4 = findConsumer(add3, "RSQRT")
  if (hasUndefined(rsqrt4)) continue

  const mul5 = findConsumer(rsqrt4, "MUL")
  if (hasUndefined(mul5)) continue

  const getFloat1DTensorIdxFromInputsOf = function(op) {
    for (let i of op.inputs) {
      if (is1D(tensors[i]) && isFloat32(tensors[i]))
        return i
    }
    return undefined
  }
  const iGamma = getFloat1DTensorIdxFromInputsOf(mul5)
  if (hasUndefined(iGamma)) continue

  let mul6, mul7
  for (let i of useOps[mul5.outputs[0]]) {
    const op = operators[i]
    if (opcodeIdxToOpName[op.opcode_index] !== "MUL")
      break;
    const otherInput = op.inputs[0] === mul5.outputs[0] ? op.inputs[1] : op.inputs[0]
    if (otherInput === iInstanceNormInputTensor)
      mul6 = op
    else if (otherInput === mean0.outputs[0])
      mul7 = op
  }
  if (hasUndefined(mul6, mul7)) continue

  const sub8 = findConsumer(mul7, "SUB")
  if (hasUndefined(sub8)) continue

  const iBeta = getFloat1DTensorIdxFromInputsOf(sub8)
  if (hasUndefined(iBeta)) continue

  const add9 = findConsumer(sub8, "ADD")
  if (hasUndefined(add9)) continue

  const add9_2 = findConsumer(mul6, "ADD")
  if (hasUndefined(add9_2)) continue

  if (add9 !== add9_2)
    continue

  const getActivation = function(op) {
    return op.builtin_options.fused_activation_function
  }
  const activation = getActivation(add9)
  if (hasUndefined(activation)) continue

  //--------------------------------------------------------------------------
  // convert to instance norm
  let instanceNormOpcodeIdx = model.operator_codes.findIndex(o => { return o.builtin_code === "INSTANCE_NORM" })
  opcodeIdxToOpName.indexOf('INSTANCE_NORM')
  if (instanceNormOpcodeIdx === -1) {
    model.operator_codes.push( { "builtin_code": "INSTANCE_NORM", "version": 1 } )
    instanceNormOpcodeIdx = model.operator_codes.length - 1;
  }
  // construct instance norm operator
  let instanceNorm = {
    "opcode_index": instanceNormOpcodeIdx,
    "inputs": [ iInstanceNormInputTensor, iGamma, iBeta ],
    "outputs": [ add9.outputs[0] ],
    "builtin_options": { "epsilon": epsilon, "fused_activation_function": activation },
    "builtin_options_type": "InstanceNormOptions",
    "custom_options_format": "FLEXBUFFERS",
    "mutating_variable_inputs": [],
  }
  // add instance norm after removing 0~9 nodes
  instanceNormList.push(instanceNorm)
} // end of sqd1
let adjust = 0
for (let i = 0; i < squaredDifferenceIdxList.length; ++i) {
  let idx = squaredDifferenceIdxList[i] + adjust
  operators.splice(idx - 1, 10, instanceNormList[i])
  adjust += -9
}
let raw_fused = JSON.stringify(model)
fs.writeFileSync(inputfile+".fused", raw_fused);
