#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from rfdetr import RFDETRLarge

def iou(a,b):
    if not len(a) or not len(b): return np.zeros((len(a),len(b)))
    x1=np.maximum(a[:,None,0],b[None,:,0]); y1=np.maximum(a[:,None,1],b[None,:,1])
    x2=np.minimum(a[:,None,2],b[None,:,2]); y2=np.minimum(a[:,None,3],b[None,:,3])
    inter=np.maximum(0,x2-x1)*np.maximum(0,y2-y1)
    aa=(a[:,2]-a[:,0])*(a[:,3]-a[:,1]); bb=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    return inter/np.maximum(aa[:,None]+bb[None,:]-inter,1e-9)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--ann',required=True); p.add_argument('--images',required=True); p.add_argument('--out',required=True,type=Path); p.add_argument('--threshold',type=float,default=.5); a=p.parse_args()
    coco=COCO(a.ann); model=RFDETRLarge(); matched=fp=total=0; values=[]
    for iid in sorted(coco.getImgIds()):
        info=coco.loadImgs(iid)[0]; result=model.predict(Image.open(Path(a.images)/info['file_name']).convert('RGB'),threshold=a.threshold)
        pred=np.asarray([b for b,l in zip(result.xyxy,result.class_id) if int(l)==1],dtype=np.float32).reshape(-1,4)
        gt=[]
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=iid,iscrowd=False)):
            x,y,w,h=ann['bbox']; gt.append([x,y,x+w,y+h])
        gt=np.asarray(gt,dtype=np.float32).reshape(-1,4); total+=len(gt); matrix=iou(pred,gt); used=set(); mp=set()
        for pi in (np.argsort(-matrix.max(1)) if matrix.size else []):
            gi=int(np.argmax(matrix[pi]))
            if matrix[pi,gi]>=.5 and gi not in used: used.add(gi);mp.add(int(pi));values.append(float(matrix[pi,gi]))
        matched+=len(used);fp+=len(pred)-len(mp)
    out={'images':len(coco.getImgIds()),'gt_persons':total,'found':matched,'recall':matched/max(1,total),'false_positives':fp,'fp_per_image':fp/max(1,len(coco.getImgIds())),'mean_iou_matched':float(np.mean(values))}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__': main()
