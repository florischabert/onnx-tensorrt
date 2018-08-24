/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "BoxNMS.hpp"

#include <thrust/gather.h>
#include <thrust/tabulate.h>
#include <cassert>

nvinfer1::Dims BoxNMSPlugin::getOutputDimensions(int index,
                                                 const nvinfer1::Dims *inputDims,
                                                 int nbInputs) {
  assert(nbInputs == 3);
  assert(index < this->getNbOutputs());
  switch( index ) {
    case 1: // boxes
      return {2, {_detections_per_im, 4}};
    default:// scores, classes
      return {1, {_detections_per_im}};
  }
}

int BoxNMSPlugin::initialize() {
  return 0;
}

int BoxNMSPlugin::enqueue(int batchSize,
                          const void *const *inputs, void **outputs,
                          void *workspace, cudaStream_t stream) {
  size_t count = this->getInputDims(0).d[1];

  for( int batch = 0; batch < batchSize; batch++ ) {
    size_t in_offset = batch * count;
    auto scores_ptr = static_cast<float *>(inputs[0] + in_offset);
    auto boxes_ptr = static_cast<float4 *>(inputs[1] + in_offset);
    auto classes_ptr = static_cast<float *>(inputs[2] + in_offset);

    size_t out_offset = batch * _detections_per_im;
    auto nms_scores_ptr = static_cast<float *>(outputs[0] + out_offset);
    auto nms_boxes_ptr = static_cast<float4 *>(outputs[1] + out_offset);
    auto nms_classes_ptr = static_cast<float *>(outputs[2] + out_offset);
  
    // Extract actual detections
    thrust::device_vector<int> indices(count);
    auto last_idx = thrust::copy_if(
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(count),
      thrust::device_pointer_cast(scores_ptr),
      indices.begin(),
      thrust::placeholders::_1 > 0);
    indices.resize(thrust::distance(indices.begin(), last_idx));

    // Gather filtered scores
    thrust::device_vector<float> scores(indices.size());
    thrust::gather(indices.begin(), indices.end(),
      thrust::device_pointer_cast(scores_ptr), scores.begin());

    // Sort scores and corresponding indices
    thrust::sort_by_key(scores.begin(), scores.end(), indices.begin(), 
      thrust::greater<float>());
    indices.resize(std::min(indices.size(), static_cast<int>(_detections_per_im)));
    scores.resize(indices.size())
  
    // Gather boxes
    thrust::device_vector<float4> boxes(indices.size());
    thrust::gather(indices.begin(), indices.end(),
      thrust::device_pointer_cast(boxes_ptr), boxes.begin());

    // Gather classes
    thrust::device_vector<float> classes(indices.size());
    thrust::gather(indices.begin(), indices.end(),
      thrust::device_pointer_cast(classes_ptr), classes.begin());

    // Non maximum suppression
    // TODO

     // Copy to output
    thrust::copy(scores.begin(), scores.end(), 
      thrust::device_pointer_cast(nms_scores_ptr));
    thrust::copy(boxes.begin(), boxes.end(),
      thrust::device_pointer_cast(nms_boxes_ptr));
    thrust::copy(classes.begin(), classes.end(), 
      thrust::device_pointer_cast(nms_classes_ptr));

    // Zero fill unused scores
    thrust::fill(
      thrust::device_pointer_cast(nms_scores_ptr + indices.size()), 
      thrust::device_pointer_cast(nms_scores_ptr + _detections_per_im * batch), 0);
  }

  return 0;
}
