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
  assert(nbInputs == 3);
  assert(index < this->getNbOutputs());
  switch( index ) {
    case 0: // scores
    case 2: // classes
      return {2, {_detections_per_im, 1}};
    case 1: // boxes
      return {2, {_detections_per_im, 4}};
    case 3: // batch splits
      return {1, {1}};
  }
}

int BoxDecodePlugin::initialize() {
  _offsets.resize(_anchors.size());
  thrust::copy(_anchors.begin(), _anchors.end(), _offsets.begin());

  return 0;
}

int BoxDecodePlugin::enqueue(int batchSize,
                             const void *const *inputs, void **outputs,
                             void *workspace, cudaStream_t stream) {
  auto const& scores_dims = this->getInputDims(0);
  auto scores_ptr = static_cast<const float *>(inputs[0]);
  auto const& boxes_dims = this->getInputDims(1);
  auto boxes_ptr = static_cast<const float4 *>(inputs[1]);
  auto im_info_ptr = static_cast<const float *>(inputs[2]);

  int height = scores_dims.d[1];
  int width = scores_dims.d[2];
  int num_anchors = boxes_dims.d[0] / 4; 
  int num_classes = boxes_dims.d[0] / num_anchors;
  int scores_size = batchSize * num_anchors * num_classes * height * width;

  for( int batch = 0; batch < batchSize; batch++ ) {
    auto all_scores = merge_vectors
  }

  // Filter scores above threshold 
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

  auto top_n = std::min(static_cast<int>(indices.size()), _top_n);
  scores.resize(top_n);
  indices.resize(top_n);

  // Get batches, classes
  thrust::device_vector<float> batches(top_n);
  thrust::transform(indices.begin(), indices.end(), batches.begin(),
    thrust::placeholders::_1 / num_anchors / num_classes / height / width);
  thrust::device_vector<float> classes(top_n);
  thrust::transform(indices.begin(), indices.end(), classes.begin(),
    (thrust::placeholders::_1 / height / width) % num_classes);
  
  // Gather boxes
  thrust::device_vector<float4> boxes(top_n);
  thrust::gather(indices.begin(), indices.end(),
    thrust::device_pointer_cast(boxes_ptr), boxes.begin());

  if( !_anchors.empty() ) {
    // Add anchors offsets to deltas
    const float *offsets = thrust::raw_pointer_cast(_offsets.data());
    thrust::transform(
      boxes.begin(), boxes.end(), indices.begin(), boxes.begin(),
      [=] __device__ (float4 b, int i) {
        float im_scale = im_info_ptr[0] / height;
        float x = (i % width) * im_scale;
        float y = ((i / width)  % height) * im_scale;
        int a = (i / num_classes / height / width) % num_anchors;
        const float *d = offsets + 4*a;
        return float4{x+d[0]+b.x, y+d[1]+b.y, x+d[2]+b.z, y+d[3]+b.w};
      });
  }

  // Reshape to (batch, score, class, x, y, w, h)
  auto detections_tuple = thrust::make_tuple(
    batches.begin(), scores.begin(), classes.begin(), boxes.begin());
  thrust::copy_n(
    thrust::make_zip_iterator(detections_tuple),
    top_n,
    thrust::device_pointer_cast(detections_ptr));

  return 0;
}
