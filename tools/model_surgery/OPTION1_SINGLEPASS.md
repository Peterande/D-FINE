# Option 1: Single-pass model surgery (shared X backbone)

Denne workflowen fullfører **Option 1** med fokus på inference/kombo:
- Én forward pass per frame (shared backbone+encoder)
- Seg + det + pose i samme pass
- Ingen per-person pose-crop inference

## Viktig avklaring (misforståelse)

`DETRPoseTransformer` i dette repoet er **pose-only** (pred_logits + pred_keypoints), og returnerer ikke egne `pred_boxes`.
Derfor beholdes `det_decoder` i production-pipeline for stabile person-boxes.

Det betyr:
- `det_decoder` = person deteksjon/bokser
- `pose_decoder` = keypoints
- matching mellom disse skjer i samme frame (single pass)

## 1) Bygg merged checkpoint

```bash
python tools/model_surgery/build_pose_seg_singlepass_x.py \
  --det-config segmentation_sivert/base_dfine/dfine_hgnetv2_x_obj2coco.yml \
  --det-ckpt /home/berna/D-FINE-NEWBRINGER/outputs/standard/dfine_0.73.pth \
  --pose-config /home/berna/D-FINE-NEWBRINGER/pose_estimation_berna/base_dfine/dfine_hgnetv2_x_obj2coco_detrpose_paper.yml \
  --pose-ckpt /home/berna/D-FINE-NEWBRINGER/outputs/standard/detrpose_paper_x_v1/best.pth \
  --seg-ckpt /home/berna/D-FINE-NEWBRINGER/outputs/standard/dfine_0.73.pth \
  --seg-num-classes 7 \
  --seg-feature-dim 384 \
  --seg-dropout 0.1 \
  --image-size 640 \
  --min-keep-ratio 0.90 \
  --out /home/berna/D-FINE-NEWBRINGER/outputs/merged/pose_seg_singlepass_x.pth
```

Bygg-scriptet feiler eksplisitt hvis `--seg-ckpt` ikke inneholder `seg_head.*` keys.

## 2) Kjør single-pass inference

```bash
python tools/model_surgery/infer_video_singlepass_x.py \
  --det-config /home/berna/D-FINE-NEWBRINGER/segmentation_sivert/base_dfine/dfine_hgnetv2_x_obj2coco.yml \
  --pose-config /home/berna/D-FINE-NEWBRINGER/pose_estimation_berna/base_dfine/dfine_hgnetv2_x_obj2coco_detrpose_paper.yml \
  --merged-ckpt /home/berna/D-FINE-NEWBRINGER/outputs/merged/pose_seg_singlepass_x.pth \
  --input /home/berna/D-FINE-NEWBRINGER/datasets/SogO.mp4 \
  --out /home/berna/D-FINE-NEWBRINGER/outputs/merged/pose_seg_singlepass_x.mp4 \
  --device cuda \
  --image-size 640 \
  --score-thr 0.35 \
  --kpt-thr 0.35 \
  --match-min-iou 0.10 \
  --seg-feature-dim 384 \
  --profile
```

## Tuning

- Hvis pose matcher feil person i nærkontakt:
  - øk `--match-min-iou` til `0.15`
  - øk `--score-thr` til `0.40`
- Hvis for få keypoints tegnes:
  - senk `--kpt-thr` til `0.30`