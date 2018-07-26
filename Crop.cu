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

#include "Crop.hpp"
#include <cuda_fp16.h>
#include <cassert>

// TODO: Move this to a common header
inline bool is_CHW(nvinfer1::Dims const& dims) {
  return (dims.nbDims == 3 &&
          dims.type[0] == nvinfer1::DimensionType::kCHANNEL &&
          dims.type[1] == nvinfer1::DimensionType::kSPATIAL &&
          dims.type[2] == nvinfer1::DimensionType::kSPATIAL);
}

nvinfer1::Dims CropPlugin::getOutputDimensions(int index,
                                               const nvinfer1::Dims *inputDims,
                                               int nbInputs) {
  assert(nbInputs == 1);
  nvinfer1::Dims const& input = inputDims[0];
  assert(is_CHW(input));
  assert(index == 0);
  nvinfer1::Dims output;
  output.nbDims = input.nbDims;
  output.d[0] = input.d[0];
  output.d[1] = (_border[2] - _border[0]) * _scale[0];
  output.d[2] = (_border[3] - _border[1]) * _scale[1];
  return output;
}

int CropPlugin::initialize() {
  _output_dims = this->getOutputDimensions(0, &this->getInputDims(0), 1);
  assert(is_CHW(this->getInputDims(0)));
  assert(is_CHW(_output_dims));
  return 0;
}

template <typename Data>
__global__
void crop_kernel_2d(int nbatch,
                    int4 border,
                    int2 scale,
                    int2 osize,
                    Data const* idata, int istride, int ibatchstride,
                    Data*       odata, int ostride, int obatchstride) {
  int x0 = threadIdx.x + blockIdx.x * blockDim.x;
  int y0 = threadIdx.y + blockIdx.y * blockDim.y;
  int z0 = blockIdx.z;

  int xmin = border.x * scale.x;
  int xmax = border.z * scale.x;
  int ymin = border.y * scale.y;
  int ymax = border.w * scale.y;

  for( int batch=z0; batch<nbatch; batch+=gridDim.z ) {
    for( int oy=y0; oy<osize.y; oy+=blockDim.y*gridDim.y ) {
      for( int ox=x0; ox<osize.x; ox+=blockDim.x*gridDim.x ) {
        if( (ox >= xmin && ox < xmax) || (oy >= ymin && oy < ymax) ) {
          odata[batch * obatchstride + oy * ostride + ox] =
            idata[batch * ibatchstride + oy * istride + ox];
        }
      }
    }
  }
}

int CropPlugin::enqueue(int batchSize,
                                 const void *const *inputs, void **outputs,
                                 void *workspace, cudaStream_t stream) {
  auto const& input_dims = this->getInputDims(0);
  int nchan = input_dims.d[0];
  int4 border = {_border[0], _border[1], _border[2], _border[3]};
  int2 scale = {_scale[1], _scale[0]};
  int2 osize = {_output_dims.d[2], _output_dims.d[1]};
  int istride =   input_dims.d[2];
  int ostride = _output_dims.d[2];
  int ibatchstride =   input_dims.d[1] * istride;
  int obatchstride = _output_dims.d[1] * ostride;
  dim3 block(32, 16);
  dim3 grid((osize.x - 1) / block.x + 1,
            (osize.y - 1) / block.y + 1,
            std::min(batchSize * nchan, 65535));
  if (getDataType()==nvinfer1::DataType::kFLOAT) {				
    crop_kernel_2d<<<grid, block, 0, stream>>>
      (batchSize * nchan, border, scale, osize,
        static_cast<float const*>( inputs[0]), istride, ibatchstride,
        static_cast<float*      >(outputs[0]), ostride, obatchstride);
  } else {
    crop_kernel_2d<<<grid, block, 0, stream>>>
      (batchSize * nchan, border, scale, osize,
        static_cast<__half const*>( inputs[0]), istride, ibatchstride,
        static_cast<__half*      >(outputs[0]), ostride, obatchstride);
  }
  return cudaGetLastError() != cudaSuccess;
}
