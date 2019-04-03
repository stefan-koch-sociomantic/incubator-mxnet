/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2016 by Contributors
 * \file c_api_ndarray.cc
 * \brief C API of mxnet
 */

#include <mxnet/base.h>
#include <mxnet/c_api.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <mxnet/op_attr_types.h>
#include <mxnet/imperative.h>
#include <nnvm/node.h>
#include <nnvm/op_attr_types.h>
#include <string>
#include "./c_api_common.h"
#include "../common/utils.h"
#include "../common/exec_utils.h"
#include "../imperative/imperative_utils.h"
#include "../imperative/cached_op.h"

using namespace mxnet;

void SetNDInputsOutputs(const nnvm::Op* op,
                        std::vector<NDArray*>* ndinputs,
                        std::vector<NDArray*>* ndoutputs,
                        int num_inputs,
                        const NDArrayHandle *inputs,
                        int *num_outputs,
                        int infered_num_outputs,
                        int num_visible_outputs,
                        NDArrayHandle **outputs) {
  NDArray** out_array = *reinterpret_cast<NDArray***>(outputs);

  ndinputs->clear();
  ndinputs->reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    ndinputs->emplace_back(reinterpret_cast<NDArray*>(inputs[i]));
  }

  ndoutputs->clear();
  ndoutputs->reserve(infered_num_outputs);
  if (out_array == nullptr) {
    for (int i = 0; i < infered_num_outputs; ++i) {
      ndoutputs->emplace_back(new NDArray());
    }
    *num_outputs = num_visible_outputs;
  } else {
    CHECK(*num_outputs == infered_num_outputs || *num_outputs == num_visible_outputs)
      << "Operator expects " << infered_num_outputs << " (all) or "
      << num_visible_outputs << " (visible only) outputs, but got "
      << *num_outputs << " instead.";
    for (int i = 0; i < *num_outputs; ++i) {
      ndoutputs->emplace_back(out_array[i]);
    }
    for (int i = *num_outputs; i < infered_num_outputs; ++i) {
      ndoutputs->emplace_back(new NDArray());
    }
  }
}

void MXImperativeInvokeImpl(AtomicSymbolCreator creator,
                            int num_inputs,
                            NDArrayHandle *inputs,
                            int *num_outputs,
                            NDArrayHandle **outputs,
                            int num_params,
                            const char **param_keys,
                            const char **param_vals) {

  char input_ptrs_[512];
  input_ptrs_[0] = '{';
  char* input_ptrs = &input_ptrs_[1];
  for(unsigned int i = 0; i < num_inputs; i++)
  {
      input_ptrs += sprintf(input_ptrs, "%p, ", inputs[i]);
  }

  input_ptrs[-2] = '}';
  input_ptrs[-1] = '\0';

  const char* sname;
  MXSymbolGetAtomicSymbolName(creator, &sname);

  char keys_string_[512] = "[";
  char vals_string_[512] = "[";
  char* keys_string = &keys_string_[1];
  char* vals_string = &vals_string_[1];

  if (num_params) 
    for(int i = 0; i < num_params; i++)
    {
      keys_string += sprintf(keys_string, "\"%s\", ", param_keys[i]);
      vals_string += sprintf(vals_string, "\"%s\", ", param_vals[i]);
    }

  keys_string[-2] = ']';
  vals_string[-2] = ']';
  keys_string[-1] = '\0';
  vals_string[-1] = '\0';


  std::cout << __FUNCTION__ << " (" 
        << "\"" << sname << "\"" << ", "
        << "num_inputs=" << num_inputs << ", "
        << input_ptrs_ << ", "
        << "num_params=" << num_params << ", "
        << keys_string_
        << vals_string_
        << std::endl;

      const nnvm::Op* op = static_cast<nnvm::Op*>(creator);
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();

  nnvm::NodeAttrs attrs = imperative::ParseAttrs(op, num_inputs, num_params,
                                                 param_keys, param_vals);

  int infered_num_outputs;
  int num_visible_outputs;
  imperative::SetNumOutputs(op, attrs, num_inputs, &infered_num_outputs, &num_visible_outputs);

  std::vector<NDArray*> ndinputs, ndoutputs;
  SetNDInputsOutputs(op, &ndinputs, &ndoutputs, num_inputs, inputs,
      num_outputs, infered_num_outputs, num_visible_outputs, outputs);

  auto state = Imperative::Get()->Invoke(Context::CPU(), attrs, ndinputs, ndoutputs);
  if (Imperative::Get()->is_recording()) {
    Imperative::Get()->RecordOp(std::move(attrs), ndinputs, ndoutputs, state);
  }

  for (int i = *num_outputs; i < infered_num_outputs; ++i) delete ndoutputs[i];

  if (*outputs == nullptr) {
    ret->ret_handles.clear();
    ret->ret_handles.reserve(*num_outputs);
    for (int i = 0; i < *num_outputs; ++i) ret->ret_handles.push_back(ndoutputs[i]);
    *outputs = reinterpret_cast<NDArrayHandle*>(dmlc::BeginPtr(ret->ret_handles));
  }


  char output_ptrs_[512];
  output_ptrs_[0] = '{';
  char* output_ptrs = &output_ptrs_[1];

  for(unsigned int i = 0; i < *num_outputs; i++)
  {
      output_ptrs += sprintf(output_ptrs, "%p, ", *outputs[i]);
  }

  output_ptrs[-2] = '}';
  output_ptrs[-1] = '\0';

  std::cout << __FUNCTION__ << " output: " 
    << output_ptrs_ << std::endl;  

}

int MXImperativeInvoke(AtomicSymbolCreator creator,
                       int num_inputs,
                       NDArrayHandle *inputs,
                       int *num_outputs,
                       NDArrayHandle **outputs,
                       int num_params,
                       const char **param_keys,
                       const char **param_vals) {


  API_BEGIN();
  MXImperativeInvokeImpl(creator, num_inputs, inputs, num_outputs, outputs,
                         num_params, param_keys, param_vals);
  API_END();

}

int MXImperativeInvokeEx(AtomicSymbolCreator creator,
                         int num_inputs,
                         NDArrayHandle *inputs,
                         int *num_outputs,
                         NDArrayHandle **outputs,
                         int num_params,
                         const char **param_keys,
                         const char **param_vals,
                         const int **out_stypes) {  // outputs storage types
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();
  MXImperativeInvokeImpl(creator, num_inputs, inputs, num_outputs, outputs,
                         num_params, param_keys, param_vals);
  NDArray** out_array = *reinterpret_cast<NDArray***>(outputs);
  ret->out_types.clear();
  ret->out_types.reserve(*num_outputs);
  for (int i = 0; i < *num_outputs; ++i) {
    ret->out_types.emplace_back(out_array[i]->storage_type());
  }
  *out_stypes = dmlc::BeginPtr(ret->out_types);
  API_END();
}

int MXCreateCachedOp(SymbolHandle handle,
                     CachedOpHandle *out) {
  nnvm::Symbol* sym = static_cast<nnvm::Symbol*>(handle);

  API_BEGIN();
  auto inputs = sym->ListInputs(nnvm::Symbol::kAll);
  std::vector<std::string> input_names;
  input_names.reserve(inputs.size());
  for (const auto& i : inputs) input_names.push_back(i->attrs.name);
  *out = new CachedOpPtr(new CachedOp(
      *sym, std::vector<std::pair<std::string, std::string> >()));
  API_END();
}

int MXCreateCachedOpEx(SymbolHandle handle,
                       int num_flags,
                       const char** keys,
                       const char** vals,
                       CachedOpHandle *out) {
  nnvm::Symbol* sym = static_cast<nnvm::Symbol*>(handle);

  API_BEGIN();
  std::vector<std::pair<std::string, std::string> > flags;
  for (int i = 0; i < num_flags; ++i) {
    flags.emplace_back(keys[i], vals[i]);
  }
  *out = new CachedOpPtr(new CachedOp(*sym, flags));

  char keys_string_[512] = "{";
  char vals_string_[512] = "{";
  char* keys_string = &keys_string_[1];
  char* vals_string = &vals_string_[1];

  for(int i = 0; i < num_flags; i++)
  {
    keys_string += sprintf(keys_string, "\"%s\", ", keys[i]);
    vals_string += sprintf(vals_string, "\"%s\", ", vals[i]);
  }

  keys_string[-2] = '}';
  vals_string[-2] = '}';
  keys_string[-1] = '\0';
  vals_string[-1] = '\0';

  std::cout << __FUNCTION__ << " ("
    << handle << ", "
    << num_flags << ", "
    << keys_string_ << ", "
    << vals_string_ << ", "
    << *out
    << ")" << std::endl;

  API_END();
}

int MXFreeCachedOp(CachedOpHandle handle) {
  CachedOpPtr* g = static_cast<CachedOpPtr*>(handle);
  API_BEGIN();
  delete g;
  API_END();
}

int MXInvokeCachedOp(CachedOpHandle handle,
                     int num_inputs,
                     NDArrayHandle *inputs,
                     int *num_outputs,
                     NDArrayHandle **outputs) {
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();

  API_BEGIN();
  CachedOpPtr op = *static_cast<CachedOpPtr*>(handle);
  std::vector<NDArray*> ndinputs;
  ndinputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    ndinputs.push_back(reinterpret_cast<NDArray*>(inputs[i]));
  }

  std::vector<NDArray*> ndoutputs;
  ndoutputs.reserve(op->num_outputs());
  if (*outputs == nullptr) {
    *num_outputs = op->num_outputs();
    for (int i = 0; i < *num_outputs; ++i) ndoutputs.push_back(new NDArray());
  } else {
    CHECK_EQ(*num_outputs, op->num_outputs())
        << "CachedOp expects " << op->num_outputs() << " outputs, but "
        << *num_outputs << " was given.";
    for (int i = 0; i < *num_outputs; ++i) {
      ndoutputs.push_back(reinterpret_cast<NDArray*>((*outputs)[i]));
    }
  }

  op->Forward(op, ndinputs, ndoutputs);

  if (*outputs == nullptr) {
    ret->ret_handles.clear();
    ret->ret_handles.reserve(*num_outputs);
    for (int i = 0; i < *num_outputs; ++i) {
      ret->ret_handles.push_back(ndoutputs[i]);
    }
    *outputs = dmlc::BeginPtr(ret->ret_handles);
  }

  API_END();
}

int MXInvokeCachedOpEx(CachedOpHandle handle,
                       int num_inputs,
                       NDArrayHandle *inputs,
                       int *num_outputs,
                       NDArrayHandle **outputs,
                       const int **out_stypes) {  // outputs storage types
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  int err = MXInvokeCachedOp(handle, num_inputs, inputs, num_outputs, outputs);
  if (err != 0) return err;
  API_BEGIN();
  NDArray** out_array = reinterpret_cast<NDArray**>(*outputs);
  ret->out_types.clear();
  ret->out_types.reserve(*num_outputs);
  for (int i = 0; i < *num_outputs; ++i) {
    ret->out_types.emplace_back(out_array[i]->storage_type());
  }
  *out_stypes = dmlc::BeginPtr(ret->out_types);

  char input_string_[512];
  input_string_[0] = '{';
  char* input_string = &input_string_[1];
  for(unsigned int i = 0; i < num_inputs; i++)
  {
      input_string += sprintf(input_string, "%p, ", inputs[i]);
  }

  input_string[-2] = '}';
  input_string[-1] = '\0';

  char output_string_[512];
  output_string_[0] = '{';
  char* output_string = &output_string_[1];
  for(unsigned int i = 0; i < *num_outputs; i++)
  {
      output_string += sprintf(output_string, "%p, ", (*outputs)[i]);
  }

  output_string[-2] = '}';
  output_string[-1] = '\0';

  std::cout << __FUNCTION__ << " (" 
        << handle << ", "
        << num_inputs << ", "
        << input_string_
        << *num_outputs << ", "
        << output_string_ << ", "
        << "output_stypes omitted"
        << std::endl;


  API_END();
}

int MXAutogradIsTraining(bool* curr) {
  API_BEGIN();
  *curr = Imperative::Get()->is_training();
  API_END();
}

int MXAutogradSetIsTraining(int is_training, int* prev) {
  API_BEGIN();
  *prev = Imperative::Get()->set_is_training(static_cast<bool>(is_training));
  API_END();
}

int MXAutogradIsRecording(bool* curr) {
  API_BEGIN();
  *curr = Imperative::Get()->is_recording();
  API_END();
}

int MXAutogradSetIsRecording(int is_recording, int* prev) {
  API_BEGIN();
  *prev = Imperative::Get()->set_is_recording(static_cast<bool>(is_recording));
  API_END();
}

int MXAutogradMarkVariables(mx_uint num_var,
                            NDArrayHandle *var_handles,
                            mx_uint *reqs_array,
#define BUFFER_DEF(VAR, DIM) \
  char VAR##_string_[DIM] = "{"; \
  char* VAR##_string = &VAR##_string_[1];

#define FOREACH(VAR, END_VAR, FORMAT) \
  for(unsigned int i = 0; i < END_VAR;i++) \
  { VAR##_string += sprintf(VAR##_string, FORMAT, VAR[i]); } \
  if (END_VAR) \
  { VAR##_string[-2] = '}'; VAR##_string[-1] = ','; } \
  VAR##_string[0] = END_VAR ? ' ' : '}';  VAR##_string[1] = '\0';
                          
NDArrayHandle *grad_handles) {
  BUFFER_DEF(var_handles, 512);
  BUFFER_DEF(reqs_array, 255);
  BUFFER_DEF(grad_handles, 512);

  FOREACH(var_handles, num_var, "%p, ");
  FOREACH(reqs_array, num_var, "%d, ");
  FOREACH(grad_handles, num_var, "%p, ");

  std::cout << __FUNCTION__ << " ("
        << num_var << ", "
        << var_handles_string_
        << reqs_array_string_
        << grad_handles_string_
        << std::endl;

  API_BEGIN();
  std::vector<NDArray*> variables, gradients;
  std::vector<mx_uint> grad_reqs;
  variables.reserve(num_var);
  gradients.reserve(num_var);
  grad_reqs.reserve(num_var);
  for (mx_uint i = 0; i < num_var; ++i) {
    variables.emplace_back(static_cast<NDArray*>(var_handles[i]));
    gradients.emplace_back(static_cast<NDArray*>(grad_handles[i]));
    grad_reqs.emplace_back(reqs_array[i]);
  }
  Imperative::Get()->MarkVariables(variables, grad_reqs, gradients);
#undef FOREACH
#undef BUFFER_DEF
  API_END();
}

int MXAutogradComputeGradient(mx_uint num_output,
                              NDArrayHandle *output_handles) {
  return MXAutogradBackward(num_output, output_handles, nullptr, 0);
}

int MXAutogradBackward(mx_uint num_output,
                       NDArrayHandle *output_handles,
                       NDArrayHandle *ograd_handles,
                       int retain_graph) {
  return MXAutogradBackwardEx(num_output, output_handles, ograd_handles,
                              0, nullptr, retain_graph, false, true,
                              nullptr, nullptr);
}

int MXAutogradBackwardEx(mx_uint num_output,
                         NDArrayHandle *output_handles,
                         NDArrayHandle *ograd_handles,
                         mx_uint num_variables,
                         NDArrayHandle *var_handles,
                         int retain_graph,
                         int create_graph,
                         int is_train,
                         NDArrayHandle **grad_handles,
                         int **grad_stypes) {
  MXAPIThreadLocalEntry *ret = MXAPIThreadLocalStore::Get();
  API_BEGIN();

  std::vector<NDArray*> outputs, ograds, variables;
  outputs.reserve(num_output);
  for (mx_uint i = 0; i < num_output; ++i) {
    outputs.emplace_back(reinterpret_cast<NDArray*>(output_handles[i]));
  }

  ograds.reserve(num_output);
  for (mx_uint i = 0; i < num_output; ++i) {
    if (ograd_handles != nullptr) {
      ograds.emplace_back(reinterpret_cast<NDArray*>(ograd_handles[i]));
    } else {
      ograds.emplace_back(nullptr);
    }
  }

  variables.reserve(num_variables);
  for (mx_uint i = 0; i < num_variables; ++i) {
    variables.emplace_back(reinterpret_cast<NDArray*>(var_handles[i]));
  }

  auto grads = Imperative::Get()->Backward(outputs, ograds, variables, is_train,
                                                  retain_graph, create_graph);
  if (num_variables != 0) {
    ret->ret_handles.clear();
    ret->out_types.clear();
    ret->ret_handles.reserve(grads.size());
    ret->out_types.reserve(grads.size());
    for (const auto& i : grads) {
      ret->ret_handles.push_back(i);
      ret->out_types.push_back(i->storage_type());
    }
    *grad_handles = dmlc::BeginPtr(ret->ret_handles);
    *grad_stypes = dmlc::BeginPtr(ret->out_types);
  }
  API_END();
}

int MXAutogradGetSymbol(NDArrayHandle handle, SymbolHandle *out) {
  API_BEGIN();
  NDArray *head = reinterpret_cast<NDArray*>(handle);
  auto sym = new nnvm::Symbol(head->get_autograd_symbol());
  *out = reinterpret_cast<SymbolHandle>(sym);
  API_END();
}
