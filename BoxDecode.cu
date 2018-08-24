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

#include "BoxDecode.hpp"

#include <thrust/gather.h>
#include <thrust/tabulate.h>
#include <cassert>

nvinfer1::Dims BoxDecodePlugin::getOutputDimensions(int index,
                                                    const nvinfer1::Dims *inputDims,
                                                    int nbInputs) {
  assert(nbInputs == );
  assert(index < this->getNbOutputs());
  switch( index ) {
    case 1: // boxes
      return {2, {_top_n, 4}};
    default:// scores, classes
      return {1, {_top_n}};
  }
}

int BoxDecodePlugin::initialize() {
  _anchors_d.resize(_anchors.size());
  thrust::copy(_anchors.begin(), _anchors.end(), _anchors_d.begin());

  return 0;
}

int BoxDecodePlugin::enqueue(int batchSize,
                             const void *const *inputs, void **outputs,
                             void *workspace, cudaStream_t stream) {
  auto const& scores_dims = this->getInputDims(0);
  auto const& boxes_dims = this->getInputDims(1);
  size_t height = scores_dims.d[1];
  size_t width = scores_dims.d[2];
  size_t num_anchors = boxes_dims.d[0] / 4; 
  size_t num_classes = boxes_dims.d[0] / num_anchors;
  size_t scores_size = num_anchors * num_classes * height * width;

  for( int batch = 0; batch < batchSize; batch++ ) {
    size_t in_offset = batch * scores_size;
    auto scores_ptr = static_cast<const float *>(inputs[0]) + in_offset;
    auto boxes_ptr = static_cast<const float4 *>(inputs[1]) + in_offset / num_classes;

    size_t out_offset = batch * _top_n;
    auto out_scores_ptr = static_cast<float *>(outputs[0]) + out_offset;
    auto out_boxes_ptr = static_cast<float4 *>(outputs[1]) + out_offset;
    auto out_classes_ptr = static_cast<float *>(outputs[2]) + out_offset;
  
    // // Filter scores above threshold 
    thrust::device_vector<int> indices(scores_size);
    auto last_idx = thrust::copy_if(
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(scores_size),
      thrust::device_pointer_cast(scores_ptr),
      indices.begin(),
      thrust::placeholders::_1 > _score_thresh);
    indices.resize(thrust::distance(indices.begin(), last_idx));
    
    // Gather filtered scores
    thrust::device_vector<float> scores(indices.size());
    thrust::gather(indices.begin(), indices.end(),
      thrust::device_pointer_cast(scores_ptr), scores.begin());

    // Sort scores and corresponding indices
    thrust::sort_by_key(scores.begin(), scores.end(), indices.begin(), 
      thrust::greater<float>());
    indices.resize(
      std::min(indices.size(), static_cast<size_t>(_top_n)));
    scores.resize(indices.size());

    // Gather boxes
    thrust::device_vector<float4> boxes(indices.size());
    thrust::gather(indices.begin(), indices.end(),
      thrust::device_pointer_cast(boxes_ptr), boxes.begin());

    // Get classes
    thrust::device_vector<float> classes(indices.size());
    thrust::transform(indices.begin(), indices.end(), classes.begin(),
      (thrust::placeholders::_1 / height / width) % num_classes);

    if( !_anchors.empty() ) {
      // Add anchors offsets to deltas
      auto anchors_ptr_d = thrust::raw_pointer_cast(_anchors_d.data());
      thrust::transform(
        boxes.begin(), boxes.end(), indices.begin(), boxes.begin(),
        [=] __device__ (float4 b, int i) {
          float x = (i % width) * _scale;
          float y = ((i / width)  % height) * _scale;
          int a = (i / num_classes / height / width) % num_anchors;
          float *d = anchors_ptr_d + 4*a;
          return float4{x+d[0]+b.x, y+d[1]+b.y, x+d[2]+b.z, y+d[3]+b.w};
        });
    }

    // Copy to output
    thrust::copy(scores.begin(), scores.end(), 
      thrust::device_pointer_cast(out_scores_ptr));
    thrust::copy(boxes.begin(), boxes.end(),
      thrust::device_pointer_cast(out_boxes_ptr));
    thrust::copy(classes.begin(), classes.end(), 
      thrust::device_pointer_cast(out_classes_ptr));

    // Zero fill unused scores
    thrust::fill(
      thrust::device_pointer_cast(out_scores_ptr + indices.size()), 
      thrust::device_pointer_cast(out_scores_ptr + _top_n * batch), 0);
  }

  return 0;
}
