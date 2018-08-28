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

void nms(const thrust::device_vector<float>::iterator& scores_begin, 
         const thrust::device_vector<float>::iterator& scores_end,
         const thrust::device_vector<float4>::iterator& boxes_begin,
         float threshold) {

  auto count = thrust::distance(scores_begin, scores_end);

  // Sort scores
  thrust::device_vector<int> indices(count);
  thrust::copy(
    thrust::make_counting_iterator<int>(0),
    thrust::make_counting_iterator<int>(count),
    indices.begin());

  thrust::device_vector<float> scores(count);
  thrust::copy(scores_begin, scores_end, scores.begin());
  thrust::sort_by_key(scores.begin(), scores.end(), indices.begin());

  while( !indices.empty() ) {
    auto last = indices.back();

    // Compute boxes intersection
    thrust::device_vector<float4> boxes(indices.size() - 1);
    thrust::gather(
      indices.begin(), indices.end() - 1, boxes_begin, boxes.begin());

    thrust::device_vector<float> overlap(boxes.size());
    thrust::transform(boxes.begin(), boxes.end(), overlap.begin(),
      [=] __device__ (float4 box) {
        float4 it = *(boxes_begin + last);
        float x1 = max(it.x, box.x);
        float y1 = max(it.y, box.y);
        float x2 = min(it.z, box.z);
        float y2 = min(it.w, box.w);
        float w = max(0.0f, x2 - x1 + 1);
        float h = max(0.0f, y2 - y1 + 1);
        float area = (box.z - box.x + 1) * (box.w - box.y + 1);
        return (w * h) / area;
      });

    indices.pop_back();

    // Zero out discarded scores
    thrust::device_vector<int> discard(indices.size());
    auto last_idx = thrust::copy_if(
      indices.begin(), indices.end(), 
      overlap.begin(), discard.begin(),
      thrust::placeholders::_1 > threshold);
    discard.resize(thrust::distance(discard.begin(), last_idx));
    thrust::fill(
      thrust::make_permutation_iterator(scores_begin, discard.begin()),
      thrust::make_permutation_iterator(scores_begin, discard.end()), 0);

    // Keep indices under overlap threshold
    last_idx = thrust::copy_if(
      indices.begin(), indices.end(), 
      overlap.begin(), indices.begin(),
      thrust::placeholders::_1 <= threshold);
    indices.resize(thrust::distance(indices.begin(), last_idx));
  }
}

nvinfer1::Dims BoxNMSPlugin::getOutputDimensions(int index,
                                                 const nvinfer1::Dims *inputDims,
                                                 int nbInputs) {
  assert(nbInputs == 3);
  assert(index < this->getNbOutputs());
  switch( index ) {
    case 1: // boxes
      return {2, {_detections_per_im, 4}};
    default: // scores, classes
      return {1, {_detections_per_im}};
  }
}

int BoxNMSPlugin::initialize() {
  return 0;
}

int BoxNMSPlugin::enqueue(int batchSize,
                          const void *const *inputs, void **outputs,
                          void *workspace, cudaStream_t stream) {
  size_t count = this->getInputDims(0).d[0];

  for( int batch = 0; batch < batchSize; batch++ ) {
    size_t in_offset = batch * count;
    auto scores_ptr = static_cast<const float *>(inputs[0]) + in_offset;
    auto boxes_ptr = static_cast<const float4 *>(inputs[1]) + in_offset;
    auto classes_ptr = static_cast<const float *>(inputs[2]) + in_offset;

    size_t out_offset = batch * _detections_per_im;
    auto nms_scores_ptr = static_cast<float *>(outputs[0]) + out_offset;
    auto nms_boxes_ptr = static_cast<float4 *>(outputs[1]) + out_offset;
    auto nms_classes_ptr = static_cast<float *>(outputs[2]) + out_offset;
  
    // Extract actual detections
    thrust::device_vector<int> indices(count);
    auto last_idx = thrust::copy_if(
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(count),
      thrust::device_pointer_cast(scores_ptr),
      indices.begin(),
      thrust::placeholders::_1 > 0);
    indices.resize(thrust::distance(indices.begin(), last_idx));

    // Sort by class
    thrust::device_vector<float> classes(indices.size());
    thrust::gather(indices.begin(), indices.end(),
      thrust::device_pointer_cast(classes_ptr), classes.begin());
  
    thrust::sort_by_key(classes.begin(), classes.end(), indices.begin());

    // Gather scores, boxes
    thrust::device_vector<float> scores(indices.size());
    thrust::gather(indices.begin(), indices.end(),
      thrust::device_pointer_cast(scores_ptr), scores.begin());

    thrust::device_vector<float4> boxes(indices.size());
    thrust::gather(indices.begin(), indices.end(),
      thrust::device_pointer_cast(boxes_ptr), boxes.begin()); 

    // Per-class non maximum suppression
    auto class_begin = classes.begin();
    while( class_begin < classes.end() ) {
      auto class_end = thrust::find_if(class_begin, classes.end(),
        thrust::placeholders::_1 > *class_begin);

      auto begin = thrust::distance(classes.begin(), class_begin);
      auto end = thrust::distance(classes.begin(), class_end);
      nms(scores.begin() + begin, scores.begin() + end, 
        boxes.begin() + begin, _nms_thresh);
      
      class_begin = class_end;
    }

    // Sort scores and corresponding indices
    indices.resize(boxes.size());
    thrust::copy(
      thrust::make_counting_iterator<int>(0),
      thrust::make_counting_iterator<int>(indices.size()),
      indices.begin());

    thrust::sort_by_key(scores.begin(), scores.end(), indices.begin(), 
      thrust::greater<float>());
    indices.resize(
      std::min(indices.size(), static_cast<size_t>(_detections_per_im)));
    scores.resize(indices.size());
  
    // Gather filtered boxes, classes
    thrust::device_vector<float4> boxes_nms(indices.size());
    thrust::gather(indices.begin(), indices.end(),
      boxes.begin(), boxes_nms.begin()); 

    thrust::device_vector<float> classes_nms(indices.size());
    thrust::gather(indices.begin(), indices.end(),
      classes.begin(), classes_nms.begin());

    // Copy to output
    thrust::copy(scores.begin(), scores.end(), 
      thrust::device_pointer_cast(nms_scores_ptr));
    thrust::copy(boxes_nms.begin(), boxes_nms.end(),
      thrust::device_pointer_cast(nms_boxes_ptr));
    thrust::copy(classes_nms.begin(), classes_nms.end(), 
      thrust::device_pointer_cast(nms_classes_ptr));

    // Zero fill unused scores
    thrust::fill(
      thrust::device_pointer_cast(nms_scores_ptr + indices.size()), 
      thrust::device_pointer_cast(nms_scores_ptr + _detections_per_im), 0);
  }

  return 0;
}
