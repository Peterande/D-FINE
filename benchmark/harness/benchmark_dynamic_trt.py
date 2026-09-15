#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import cv2,numpy as np
import onnxruntime as ort
import pycuda.autoinit  # noqa
import pycuda.driver as cuda
import tensorrt as trt

class Allocator(trt.IOutputAllocator):
    def __init__(self): super().__init__(); self.buffers={}; self.shapes={}
    def reallocate_output(self,name,memory,size,alignment):
        self.buffers[name]=cuda.mem_alloc(max(1,size)); return int(self.buffers[name])
    def notify_shape(self,name,shape): self.shapes[name]=tuple(shape)

def main():
    p=argparse.ArgumentParser();p.add_argument('--engine',required=True);p.add_argument('--onnx');p.add_argument('--image');p.add_argument('--out',required=True,type=Path);p.add_argument('--iterations',type=int,default=200);a=p.parse_args()
    logger=trt.Logger(trt.Logger.ERROR);runtime=trt.Runtime(logger);engine=runtime.deserialize_cuda_engine(Path(a.engine).read_bytes());context=engine.create_execution_context();context.set_input_shape('input',(1,3,640,640))
    if a.image:
        im=cv2.imread(a.image);ratio=min(640/im.shape[0],640/im.shape[1]);res=cv2.resize(im,(int(im.shape[1]*ratio),int(im.shape[0]*ratio)));pad=np.full((640,640,3),114,np.uint8);pad[:res.shape[0],:res.shape[1]]=res;x=pad.transpose(2,0,1)[None].astype(np.float32)
    else:x=np.random.default_rng(0).uniform(0,255,(1,3,640,640)).astype(np.float32)
    x=np.ascontiguousarray(x);inp=cuda.mem_alloc(x.nbytes);cuda.memcpy_htod(inp,x);context.set_tensor_address('input',int(inp));alloc=Allocator()
    outputs=[]
    for i in range(engine.num_io_tensors):
        name=engine.get_tensor_name(i)
        if engine.get_tensor_mode(name)==trt.TensorIOMode.OUTPUT: context.set_output_allocator(name,alloc);outputs.append(name)
    stream=cuda.Stream()
    for _ in range(10): assert context.execute_async_v3(stream.handle)
    stream.synchronize();times=[]
    for _ in range(a.iterations):
        start=cuda.Event();end=cuda.Event();start.record(stream);assert context.execute_async_v3(stream.handle);end.record(stream);end.synchronize();times.append(start.time_till(end))
    parity={}
    if a.onnx:
        ref=ort.InferenceSession(a.onnx,providers=['CPUExecutionProvider']).run(None,{'input':x})
        for name,expected in zip(outputs,ref):
            actual=np.empty(alloc.shapes[name],dtype=np.float32);cuda.memcpy_dtoh(actual,alloc.buffers[name]);parity[name]={'mae':float(np.mean(np.abs(actual-expected))),'max_abs':float(np.max(np.abs(actual-expected)))}
    result={'iterations':len(times),'mean_ms':float(np.mean(times)),'p50_ms':float(np.percentile(times,50)),'p95_ms':float(np.percentile(times,95)),'outputs':{n:list(alloc.shapes[n]) for n in outputs},'engine_device_memory_mb':engine.device_memory_size_v2/1024**2,'parity':parity}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
