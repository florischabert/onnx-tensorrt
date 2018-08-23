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

#pragma once

#include "plugin.hpp"
#include "serialize.hpp"
#include "ShapedWeights.hpp"

#include <cassert>

class BoxDecodePlugin final : public onnx2trt::Plugin {
  float _score_thresh;
  int _pre_nms_top_n;
  float _nms_thresh;
  int _detections_per_im;
  std::vector<float> _anchors;
  std::vector<int> _anchors_counts;
protected:
  void deserialize(void const* serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    deserialize_value(&serialData, &serialLength, &_score_thresh);
    deserialize_value(&serialData, &serialLength, &_pre_nms_top_n);
    deserialize_value(&serialData, &serialLength, &_nms_thresh);
    deserialize_value(&serialData, &serialLength, &_detections_per_im);
    deserialize_value(&serialData, &serialLength, &anchors);
    deserialize_value(&serialData, &serialLength, &_anchors_counts);
  }
  virtual size_t getSerializationSize() override {
    return serialized_size(_score_thresh) + serialized_size(_pre_nms_top_n)
      + serialized_size(_nms_thresh) + serialized_size(_detections_per_im) 
      + serialized_size(_anchors) + serialized_size(_anchors_counts) 
      + getBaseSerializationSize();
  }
  virtual void serialize(void *buffer) override {
    serializeBase(buffer);
    serialize_value(&buffer, _score_thresh);
    serialize_value(&buffer, _pre_nms_top_n);
    serialize_value(&buffer, _nms_thresh);
    serialize_value(&buffer, _detections_per_im);
    serialize_value(&buffer, _anchors_counts);
  }
public:
  BoxDecodePlugin(float score_thresh, int pre_nms_top_n, float nms_thresh, int detections_per_im,
                  ShapedWeights const& anchors)
    : _score_thresh(score_thresh), _pre_nms_top_n(pre_nms_top_n), _nms_thresh(nms_thresh),
      _detections_per_im(detections_per_im), _anchors(anchors) {
    assert(score_thresh >= 0);
    assert(pre_nms_top_n > 0);
    assert(nms_thresh >= 0);
    assert(detections_per_im > 0);
    assert(_anchors.size() == std::accumulate(_anchors_counts.begin(), _anchors_counts.end(), 0));
  }
  BoxDecodePlugin(void const* serialData, size_t serialLength) {
    this->deserialize(serialData, serialLength);
  }
  virtual const char* getPluginType() const override { return "BoxDecode"; }
  virtual int getNbOutputs() const override { return 1; }
  virtual nvinfer1::Dims getOutputDimensions(int index,
                                             const nvinfer1::Dims *inputs, int nbInputDims) override;
  virtual int initialize() override;
  virtual int enqueue(int batchSize,
                      const void *const *inputs, void **outputs,
                      void *workspace, cudaStream_t stream) override;
};
