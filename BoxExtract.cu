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

#include "BoxExtract.hpp"

#include <thrust/gather.h>
#include <thrust/tabulate.h>
#include <cassert>
 
nvinfer1::Dims BoxExtractPlugin::getOutputDimensions(int index,
                                                     const nvinfer1::Dims *inputDims,
                                                     int nbInputs) {
  assert(nbInputs == 3);
  assert(index < this->getNbOutputs());
  return {2, {_top_n, 7}}; // (batch, score, class, x, y, w, h)
}
 
int BoxExtractPlugin::initialize() {
  return 0;
}
 
int BoxExtractPlugin::enqueue(int batchSize,
                              const void *const *inputs, void **outputs,
                              void *workspace, cudaStream_t stream) {
  auto const& scores_dims = this->getInputDims(0);
  auto const& boxes_dims = this->getInputDims(1);

  int batch_size = scores_dims.d[0];
  int height = scores_dims.d[2];
  int width = scores_dims.d[3];
  int num_anchors = boxes_dims.d[1] / 4; 
  int num_classes = boxes_dims.d[1] / num_anchors;

  int im_scale = static_cast<const float *>(inputs[2])[0] / height;

  thrust::device_ptr<const float> scores_ptr(static_cast<const float *>(inputs[0]));
  thrust::device_vector<float> scores(
    scores_ptr, scores_ptr + batch_size * scores_dims.d[1] * height * width);

  thrust::device_ptr<const float> boxes_ptr(static_cast<const float *>(inputs[1]));
  thrust::device_vector<float> boxes(
    boxes_ptr, boxes_ptr + batch_size * boxes_dims.d[1] * height * width);

  // Sort and filter scores above threshold
  thrust::device_vector<int> indices(scores.size());
  thrust::tabulate(indices.begin(), indices.end(),
    thrust::placeholders::_1 / num_classes);

  thrust::sort_by_key(scores.begin(), scores.end(), indices.begin());
  auto last_score = thrust::find_if(scores.begin(), scores.end(), 
    thrust::placeholders::_1 < _score_thresh);
  auto num_scores = thrust::distance(scores.begin(), last_score);

  // Gather boxes
  thrust::device_vector<float> kept_boxes(num_scores);
  thrust::gather(indices.begin(), indices.begin() + num_scores,
    boxes.begin(), kept_boxes.begin());

  if( !_anchors.empty() ) {
    // Generate anchors
    thrust::device_vector<float> xywh(4 * height * width);

    // Generate x, y
    auto x_end = xywh.begin() + height * width;
    thrust::tabulate(xywh.begin(), x_end, 
      thrust::placeholders::_1 % width);
    auto y_end = x_end + height * width;
    thrust::tabulate(x_end, y_end,
      thrust::placeholders::_1 / width);

    // Re-scale coordinates
    thrust::device_vector<float> scale(2 * height * width);
    thrust::fill(scale.begin(), scale.end(), im_scale);
    thrust::transform(xywh.begin(), y_end, scale.begin(), scale.begin(),
      thrust::multiplies<float>());

    // Copy to w, h
    auto h_end = y_end + 2 * height * width;
    thrust::copy(xywh.begin(), h_end, y_end);

    thrust::copy(xywh.begin(), xywh.end(), std::ostream_iterator<float>(std::cout, " "));

    // Add to deltas
    
    
  }

  // Reshape to (batch, score, class, x, y, w, h)

  return 0;
}
