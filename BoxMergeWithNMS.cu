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

 #include "BoxMergeWithNMS.hpp"
 #include <cassert>
 
 nvinfer1::Dims BoxMergeWithNMSPlugin::getOutputDimensions(int index,
                                                      const nvinfer1::Dims *inputDims,
                                                      int nbInputs) {
   assert(index < this->getNbOutputs());
   if( index == 1 ) {
      return {2, {_detections_per_im, 4}};
   }
   return {2, {_detections_per_im, 1}};
 }
 
 int BoxMergeWithNMSPlugin::initialize() {
   return 0;
 }
 
 int BoxMergeWithNMSPlugin::enqueue(int batchSize,
                               const void *const *inputs, void **outputs,
                               void *workspace, cudaStream_t stream) {
   auto const& input_dims = this->getInputDims(0);
   return 0;
 }
 