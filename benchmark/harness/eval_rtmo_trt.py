#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2,numpy as np
import pycuda.autoinit  # noqa
import pycuda.driver as cuda
import tensorrt as trt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from benchmark.harness.benchmark_dynamic_trt import Allocator
from rtmlib.tools.object_detection.post_processings import multiclass_nms

class Runner:
    def __init__(self,path):
        log=trt.Logger(trt.Logger.ERROR);self.runtime=trt.Runtime(log);self.engine=self.runtime.deserialize_cuda_engine(Path(path).read_bytes());self.context=self.engine.create_execution_context();self.context.set_input_shape('input',(1,3,640,640));self.stream=cuda.Stream();self.alloc=Allocator()
        for name in ('dets','keypoints'):self.context.set_output_allocator(name,self.alloc)
        self.input=cuda.mem_alloc(1*3*640*640*4);self.context.set_tensor_address('input',int(self.input))
    def __call__(self,image):
        ratio=min(640/image.shape[0],640/image.shape[1]);res=cv2.resize(image,(int(image.shape[1]*ratio),int(image.shape[0]*ratio)));pad=np.full((640,640,3),114,np.uint8);pad[:res.shape[0],:res.shape[1]]=res;x=np.ascontiguousarray(pad.transpose(2,0,1)[None].astype(np.float32));cuda.memcpy_htod_async(self.input,x,self.stream);assert self.context.execute_async_v3(self.stream.handle);self.stream.synchronize();outputs=[]
        for name in ('dets','keypoints'):
            value=np.empty(self.alloc.shapes[name],np.float32);cuda.memcpy_dtoh(value,self.alloc.buffers[name]);outputs.append(value)
        return outputs,ratio

def main():
    p=argparse.ArgumentParser();p.add_argument('--engine',required=True);p.add_argument('--coco-root',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();ann=a.coco_root/'annotations/person_keypoints_val2017.json';c=COCO(str(ann));ids=sorted(c.getImgIds(catIds=[1]));runner=Runner(a.engine);results=[]
    for n,i in enumerate(ids):
        info=c.loadImgs(i)[0];(dets,kpts),ratio=runner(cv2.imread(str(a.coco_root/'val2017'/info['file_name'])));dets=dets[0];kpts=kpts[0]
        _,keep=multiclass_nms(dets[:,:4]/ratio,dets[:,4,None],nms_thr=.65,score_thr=.001)
        kpts=kpts[keep] if keep is not None else kpts[:0]
        for det,kp in zip(dets,kpts):
            flat=[]
            for x,y,s in kp:flat += [float(x/ratio),float(y/ratio),float(s)]
            results.append({'image_id':i,'category_id':1,'keypoints':flat,'score':float(np.mean(kp[:,2]))})
        if (n+1)%500==0:print(f'{n+1}/{len(ids)}',flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(results));dt=c.loadRes(str(a.out));e=COCOeval(c,dt,'keypoints');e.params.imgIds=ids;e.evaluate();e.accumulate();e.summarize();print('stats',e.stats.tolist())
if __name__=='__main__':main()
