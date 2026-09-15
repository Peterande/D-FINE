#!/usr/bin/env python3
from __future__ import annotations
import argparse,time
from pathlib import Path
import cv2,numpy as np
from rtmlib import RTMO

SKELETON=[(15,13),(13,11),(16,14),(14,12),(11,12),(5,11),(6,12),(5,6),(5,7),(7,9),(6,8),(8,10),(0,1),(0,2),(1,3),(2,4)]
def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--input',required=True);p.add_argument('--out',required=True);p.add_argument('--threshold',type=float,default=.2);a=p.parse_args()
    model=RTMO(a.model,model_input_size=(640,640),score_thr=a.threshold,nms_thr=.65,device='cpu');cap=cv2.VideoCapture(a.input);fps=cap.get(cv2.CAP_PROP_FPS) or 30;w=int(cap.get(3));h=int(cap.get(4));Path(a.out).parent.mkdir(parents=True,exist_ok=True);writer=cv2.VideoWriter(a.out,cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h));times=[]
    while True:
        ok,frame=cap.read()
        if not ok:break
        t=time.perf_counter();kpts,scores=model(frame);times.append((time.perf_counter()-t)*1000)
        for kp,sc in zip(kpts,scores):
            for x,y in kp[sc>=a.threshold]:cv2.circle(frame,(int(x),int(y)),3,(0,0,255),-1)
            for x,y in SKELETON:
                if sc[x]>=a.threshold and sc[y]>=a.threshold:cv2.line(frame,tuple(kp[x].astype(int)),tuple(kp[y].astype(int)),(255,0,0),2)
        writer.write(frame)
    cap.release();writer.release();print({'frames':len(times),'mean_ms':float(np.mean(times)),'p50_ms':float(np.percentile(times,50)),'p95_ms':float(np.percentile(times,95))})
if __name__=='__main__':main()
