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

#include <cassert>

class CropPlugin final : public onnx2trt::Plugin {
  int _border[nvinfer1::Dims::MAX_DIMS];
  int _scale[nvinfer1::Dims::MAX_DIMS];
  nvinfer1::Dims _output_dims;
protected:
  void deserialize(void const* serialData, size_t serialLength) {
    deserializeBase(serialData, serialLength);
    deserialize_value(&serialData, &serialLength, &_border);
    deserialize_value(&serialData, &serialLength, &_scale);
  }
  size_t getSerializationSize() override {
    return serialized_size(_border) + serialized_size(_scale) + getBaseSerializationSize();
  }
  void serialize(void *buffer) override {
    serializeBase(buffer);
    serialize_value(&buffer, _border);
    serialize_value(&buffer, _scale);
  }
public:
  CropPlugin(std::vector<int> const& border, std::vector<int> const& scale) {
    assert(border.size() == 4);
    assert(scale.size() == 2);
    std::copy(border.begin(), border.end(), _border);
    std::copy(scale.begin(), scale.end(), _scale);
  }
  CropPlugin(void const* serialData, size_t serialLength) {
    this->deserialize(serialData, serialLength);
  }
  virtual const char* getPluginType() const override { return "Crop"; }
  virtual int getNbOutputs() const override { return 1; }
  virtual nvinfer1::Dims getOutputDimensions(int index,
                                             const nvinfer1::Dims *inputs, int nbInputDims) override;
  virtual int initialize() override;
  int enqueue(int batchSize,
              const void *const *inputs, void **outputs,
              void *workspace, cudaStream_t stream) override;
};
