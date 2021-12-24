import os
import argparse
import numpy as np
# PyCUDA
import pycuda.driver as cuda
import pycuda.autoinit
# TensorRT
import tensorrt as trt


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))

        # in case batch dimension is -1 (dynamic)
        if engine.get_binding_shape(binding)[0] < 0:
            size *= -1

        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1, return_host=True):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    if return_host:
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return the outputs
    if return_host:
        # Return only the host outputs.
        return [out.host for out in outputs]
    else:
        return outputs


# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


# This class is generalized for simple run of trt models by containing initialization and post-process.
TRT_LOGGER = trt.Logger()
class TRTModel(object):
    def __init__(self, engine_path, batch_size, imgH, imgW):
        print("Reading engine from file {}".format(engine_path))
        self.return_host = True
        self.batch_size = batch_size

        # load an engine and init the environment
        self.cfx = cuda.Device(0).make_context()
        with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)
        self.context.set_binding_shape(0, (batch_size, 3, imgH, imgW))

    def set_inputs(self, inputs):
        # For a single input
        self.inputs[0].host = inputs

    def postprocess(self, outputs):
        pass

    def __call__(self, inputs):
        # Make self the active context, pushing it on top of the context stack.
        self.cfx.push()

        # Set input data for inference
        self.set_inputs(inputs)

        # do inference
        outputs = do_inference(self.context,
                               self.bindings, self.inputs, self.outputs, self.stream,
                               self.batch_size, self.return_host)

        # do post process
        outputs = self.postprocess(outputs)

        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()

        return outputs


# import packages
import torch
# utils
from utils.torch_utils import time_synchronized

class TRTYOLOv5(TRTModel):
    def __init__(self, engine_path, batch_size, imgH, imgW, device='cuda:0'):
        super(TRTYOLOv5, self).__init__(engine_path, batch_size, imgH, imgW)
        self.return_host = False
        self.device = device

    def postprocess(self, outputs):
        if self.return_host:
            # Need to copy the ndarray output to a GPU tensor for following NMX process
            output = torch.from_numpy(outputs[-1]).to(self.device, non_blocking=True)
            output = output.reshape(self.batch_size, -1, 85)
        
        else:
            # Convert gpuarray to torch tensor
            outH, outD = outputs[-1].host, outputs[-1].device
            byte_size = outH.size * outH.itemsize
            output = torch.zeros(outH.shape, device=self.device).reshape((self.batch_size, -1, 85))
            cuda.memcpy_dtod(output.data_ptr(), outD, byte_size)
        return output


# test an engine file
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, default='yolov5s.pt', help='model.pt path(s)')
    #parser.add_argument('--source', type=str, default='data/images/bus.jpg', help='file/dir/URL/image.png')
    parser.add_argument('--imgH', type=int, default=640, help='inference height size (pixels)')
    parser.add_argument('--imgW', type=int, default=640, help='inference width size (pixels)')
    parser.add_argument('--batch', type=int, default=1, help='size of batch')
    args = parser.parse_args()

    # generate a random input image
    img = np.random.rand(args.batch, 3, args.imgH, args.imgW).astype(dtype=np.float32)

    # load and initialize an engine model
    model = TRTYOLOv5(args.engine, args.batch, args.imgH, args.imgW)

    # run inference
    for j in range(10):
        t1 = time_synchronized()
        output = model(img)
        t2 = time_synchronized()
        print(f'TRT Inference Done({j}): {t2 - t1:.3f}s')
    
    # clean up the context stack after inference
    model.cfx.pop()
