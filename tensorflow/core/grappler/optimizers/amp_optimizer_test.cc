/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/grappler/clusters/single_machine.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/devices.h"
#include "tensorflow/core/grappler/graph_view.h"
#include "tensorflow/core/grappler/optimizers/amp_optimizer.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace grappler {
namespace {

class AMPOptimizerTest : public GrapplerTest {
 protected:
  void SetUp() override {
    gpu_available_ = GetNumAvailableGPUs() > 0;

    if (gpu_available_) {
      virtual_cluster_.reset(new SingleMachine(/* timeout_s = */ 10, 1, 1));
    } else {
      DeviceProperties device_properties;
      device_properties.set_type("GPU");
      device_properties.mutable_environment()->insert({"architecture", "6"});
      virtual_cluster_.reset(
          new VirtualCluster({{"/GPU:1", device_properties}}));
    }
    TF_CHECK_OK(virtual_cluster_->Provision());
  }

  void TearDown() override { TF_CHECK_OK(virtual_cluster_->Shutdown()); }

  void AddSimpleNode(const string& name, const string& op,
                     const std::vector<string>& inputs, GraphDef* graph) const {
    std::vector<std::pair<string, AttrValue>> attributes;
    if (op == "AddN") {
      AttrValue num_inputs;
      num_inputs.set_i(inputs.size());
      attributes.emplace_back("N", num_inputs);
    }
    AttrValue type;
    type.set_type(DT_FLOAT);
    if (op == "Const" || op == "Placeholder") {
      attributes.emplace_back("dtype", type);
    } else if (op == "SparseMatMul") {
      attributes.emplace_back("Ta", type);
      attributes.emplace_back("Tb", type);
    } else if (op == "IdentityN") {
      AttrValue type_list;
      for (int i = 0; i < (int)inputs.size(); ++i) {
        type_list.mutable_list()->add_type(DT_FLOAT);
      }
      attributes.emplace_back("T", type_list);
    } else {
      attributes.emplace_back("T", type);
    }
    AddNode(name, op, inputs, attributes, graph);
  }

  std::unique_ptr<Cluster> virtual_cluster_;
  bool gpu_available_;
};

void VerifyGraphsEquivalent(const GraphDef& original_graph,
                            const GraphDef& optimized_graph,
                            const string& func) {
  EXPECT_EQ(original_graph.node_size(), optimized_graph.node_size()) << func;
  GraphView optimized_view(&optimized_graph);
  for (int i = 0; i < original_graph.node_size(); ++i) {
    const NodeDef& original = original_graph.node(i);
    const NodeDef& optimized = *optimized_view.GetNode(original.name());
    EXPECT_EQ(original.name(), optimized.name()) << func;
    EXPECT_EQ(original.op(), optimized.op()) << func;
    EXPECT_EQ(original.input_size(), optimized.input_size()) << func;
    if (original.input_size() == optimized.input_size()) {
      for (int j = 0; j < original.input_size(); ++j) {
        EXPECT_EQ(original.input(j), optimized.input(j)) << func;
      }
    }
  }
}

TEST_F(AMPOptimizerTest, NoOp) {
  GraphDef graph;
  AddSimpleNode("In", "Placeholder", {}, &graph);
  AddSimpleNode("B1", "Exp", {"In"}, &graph);
  AddSimpleNode("C1", "Relu", {"B1"}, &graph);
  AddSimpleNode("G1", "Sqrt", {"C1"}, &graph);
  AddSimpleNode("C2", "Relu", {"G1"}, &graph);

  GrapplerItem item;
  item.graph = graph;
  AMPOptimizer optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  VerifyGraphsEquivalent(item.graph, output, __FUNCTION__);

  GraphView output_view(&output);
  EXPECT_EQ(output_view.GetNode("In")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("B1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("C1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("G1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("C2")->attr().at("T").type(), DT_FLOAT);
}

TEST_F(AMPOptimizerTest, Simple1) {
  GraphDef graph;
  AddSimpleNode("In", "Placeholder", {}, &graph);
  AddSimpleNode("B1", "Exp", {"In"}, &graph);
  AddSimpleNode("C1", "Relu", {"B1"}, &graph);
  AddSimpleNode("G1", "Sqrt", {"C1"}, &graph);
  AddSimpleNode("C2", "Relu", {"G1"}, &graph);
  AddSimpleNode("W1", "MatMul", {"C2", "C2"}, &graph);
  AddSimpleNode("C3", "Relu", {"W1"}, &graph);
  AddSimpleNode("B2", "Exp", {"C3"}, &graph);
  AddSimpleNode("C4", "Relu", {"B2"}, &graph);
  AddSimpleNode("B4", "SparseMatMul", {"C4", "C4"}, &graph);
  AddSimpleNode("C5", "Relu", {"B4"}, &graph);

  GrapplerItem item;
  item.graph = graph;
  AMPOptimizer optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), graph.node_size() + 2);
  EXPECT_EQ(output_view.GetNode("In")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("B1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("C1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("G1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("C2")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("W1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("C3")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("B2")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("C4")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("B4")->attr().at("Ta").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("B4")->attr().at("Tb").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("C5")->attr().at("T").type(), DT_FLOAT);
}

TEST_F(AMPOptimizerTest, PreserveFetches) {
  GraphDef graph;
  AddSimpleNode("In", "Placeholder", {}, &graph);
  AddSimpleNode("Const1", "Const", {}, &graph);
  AddSimpleNode("W1", "MatMul", {"In", "Const1"}, &graph);
  AddSimpleNode("C1", "Relu", {"W1"}, &graph);
  AddSimpleNode("G1", "Sqrt", {"C1"}, &graph);
  AddSimpleNode("B1", "Exp", {"G1"}, &graph);
  AddSimpleNode("C2", "Relu", {"B1"}, &graph);
  AddSimpleNode("W2", "MatMul", {"C2", "C2"}, &graph);
  AddSimpleNode("C3", "Relu", {"W2"}, &graph);
  AddSimpleNode("B2", "Exp", {"C3"}, &graph);
  AddSimpleNode("C4", "Relu", {"B2"}, &graph);

  GrapplerItem item;
  item.graph = graph;
  item.fetch.push_back("W1");
  item.fetch.push_back("C2");
  item.fetch.push_back("C3");
  AMPOptimizer optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), graph.node_size() + 2);
  EXPECT_EQ(output_view.GetNode("In")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("Const1")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("W1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("C1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("G1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("B1")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("C2")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("W2")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("C3")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("B2")->attr().at("T").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("C4")->attr().at("T").type(), DT_FLOAT);
}

TEST_F(AMPOptimizerTest, FusedBatchNorm) {
  GraphDef graph;
  AddSimpleNode("X", "Placeholder", {}, &graph);
  AddSimpleNode("Const1", "Const", {}, &graph);
  AddSimpleNode("Scale", "Placeholder", {}, &graph);
  AddSimpleNode("Offset", "Placeholder", {}, &graph);
  AddSimpleNode("Mean", "Placeholder", {}, &graph);
  AddSimpleNode("Variance", "Placeholder", {}, &graph);
  AddSimpleNode("W1", "Conv2D", {"X", "Const1"}, &graph);
  AddSimpleNode("BN1", "FusedBatchNorm",
                {"W1", "Scale", "Offset", "Mean", "Variance"}, &graph);
  AddSimpleNode("BNG1", "FusedBatchNormGrad",
                {"BN1", "W1", "Scale", "Mean", "Variance"}, &graph);
  AddSimpleNode("G1", "Add", {"BN1", "BNG1"}, &graph);
  AddSimpleNode("W2", "Conv2D", {"G1", "Const1"}, &graph);

  GrapplerItem item;
  item.graph = graph;
  AMPOptimizer optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), graph.node_size() + 2);
  EXPECT_EQ(output_view.GetNode("W1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("BN1")->op(), "FusedBatchNormV2");
  EXPECT_EQ(output_view.GetNode("BN1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("BN1")->attr().at("U").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("BNG1")->op(), "FusedBatchNormGradV2");
  EXPECT_EQ(output_view.GetNode("BNG1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("BNG1")->attr().at("U").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("G1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("W2")->attr().at("T").type(), DT_HALF);
}

TEST_F(AMPOptimizerTest, ListTypeAttrs) {
  GraphDef graph;
  AddSimpleNode("In", "Placeholder", {}, &graph);
  AddSimpleNode("W1", "MatMul", {"In", "In"}, &graph);
  AddSimpleNode("ID", "IdentityN", {"W1", "W1", "W1"}, &graph);
  AddSimpleNode("G1", "AddN", {"ID:0", "ID:1", "ID:2"}, &graph);
  AddSimpleNode("W2", "MatMul", {"G1", "G1"}, &graph);

  GrapplerItem item;
  item.graph = graph;
  AMPOptimizer optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(virtual_cluster_.get(), item, &output));

  VLOG(1) << output.DebugString();

  GraphView output_view(&output);
  EXPECT_EQ(output.node_size(), graph.node_size() + 1);
  EXPECT_EQ(output_view.GetNode("In")->attr().at("dtype").type(), DT_FLOAT);
  EXPECT_EQ(output_view.GetNode("W1")->attr().at("T").type(), DT_HALF);
  for (auto type : output_view.GetNode("ID")->attr().at("T").list().type()) {
    EXPECT_EQ(type, DT_HALF);
  }
  EXPECT_EQ(output_view.GetNode("G1")->attr().at("T").type(), DT_HALF);
  EXPECT_EQ(output_view.GetNode("W2")->attr().at("T").type(), DT_HALF);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
