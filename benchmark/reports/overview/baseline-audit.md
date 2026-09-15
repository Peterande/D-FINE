# Tagtwo Baseline-Audit — Fase 1

Verifisert 31. august 2026, branch `model-candidate-benchmark` (fra `pose_estimation_head` @ 540ac17).
Statisk verifikasjon mot kode og artefakter. **Ingen modell kjørt, ingen hash beregnet, ingenting committet.**

Webversjon: https://claude.ai/code/artifact/c307ba80-3448-409e-af11-2965d8ce79b3

> Monorepoet `/home/berna/tagtwo-monorepo` er kun **lest** under dette arbeidet (`ls`, `find`,
> `grep`, `sed -n`, `diff`, `wc -l`). Ingen filer der er endret.

---

## Hovedfunn

| Spørsmål | Svar |
|---|---|
| Matcher `pose_estimation_head` produksjon? | **Nei** — monorepoet er foran branchen |
| Finnes produksjonsvektene lokalt? | **Nei** — hentes fra S3 som `dfine_0.48.pth` |
| Hva er output-formatet? | **Rå logits** — all postprosessering i runtime |
| Er tersklene like? | **Avvik** — produksjon `0.50`, repo-verktøy `0.35` |

---

## 01 — Verifisert produksjonsarkitektur

| Komponent | Verifisert innhold | Kilde |
|---|---|---|
| Backbone | HGNetv2-B5, `hidden_dim 384`, `feat_channels [384,384,384]` = D-FINE-X | `models/configs/dfine_hgnetv2_x_obj2coco.yml` |
| Delt arkitektur | `SharedBackboneDualDecoder` — én backbone, det-dekoder + posedekoder + seghode | `model_surgery/shared_arch.py` |
| Segmenteringshode | FPN-ASPP, 7 klasser, `feature_dim 384`, dropout 0.1 | `shared_arch.py` SegmentationHead |
| Body-part-klasser | 0 background, 1 head, 2 torso, 3 upper_arms, 4 lower_arms, 5 upper_legs, 6 lower_legs | `utils/hit_detector.py:7` |
| Posehode | DETRPose-basert, 17 keypoints, valgfri i grafen | `shared_arch.py:229` |
| Preprocessing | `/255.0` → ImageNet mean `[.485,.456,.406]` / std `[.229,.224,.225]` → NCHW | `utils/dfine_segmentation.py:88-100` |
| Terskel | `confidence_threshold = 0.5`, `nms_iou = 0.6` | `utils/config.py:27-30` |
| Runtime | FP16 TensorRT, 640x640, engine bygges per GPU + TRT-versjon | `utils/model_manager.py:194` |

### Output-kontrakt

| Tensor | Form | Postprosessering i runtime |
|---|---|---|
| `det_pred_logits` | logits | sigmoid (focal-loss-stil) → argmax klasse |
| `det_pred_boxes` | rå | kanonisk xyxy, koordinatrom autodetekteres |
| `seg_logits` | N,C,H,W logits | `np.argmax(axis=0)` → uint8 klassemaske |
| `pose_pred_logits` | logits | keypoint-dekoding |

### VERIFISERT: hit-beslutningen kobler segmentering og pose

Kroppsdel bestemmes av `np.argmax` over maskeklasser i crosshair-regionen. Ved okklusjon faller
`_infer_occluded_body_part` tilbake på en torso-firkant fra keypoint 5, 6, 11, 12 (COCO skuldre/hofter).
Pose og segmentering er **ikke uavhengige** i beslutningen.

Konsekvens: en "komponentvis" pose-bytte endrer også hit-resultatet. Benchmarken må rapportere
hit-metrikker separat for maskestyrte og posestyrte beslutninger.

---

## 02 — Repo mot monorepo: branchen er en eldre forfar

| Fil | pose_estimation_head | monorepo | Forhold |
|---|---|---|---|
| `shared_arch.py` | 175 linjer | 294 linjer | Monorepo = repo + tillegg. Repo har **ingenting** unikt. |
| `export_singlepass_onnx.py` | 189 linjer | 619 linjer | Monorepo legger til hele Hailo-banen |

Monorepoet har som branchen mangler:
- `HailoLiteSegmentationHead`, `HailoLiteDetectionSegmentationHead`
- `shape_compatible_state` / `load_shape_compatible`
- Valgfri posedekoder
- Flagg: `--hailo-seg-only`, `--hailo-det-seg`, `--det-num-classes`, `--force-seg-feature-dim`

### IKKE I BRIEFEN: et helt Hailo-spor

Monorepoets `model_surgery/` er ~30 filer, nesten alle om **Hailo-8** edge-NPU: lærer-dump,
elevtrening, kalibrering, HEF-kompilering, APK-pakking, live-verifisering mot RPi.

Hvis produktet skal kjøre på Hailo-8 i tillegg til RTX, må kandidater også kunne destilleres til
en kompakt CNN-elev. D-FINEs transformerdekoder er nettopp det Hailo DFC 3.33 ikke parser.
**Må avklares før kandidatlisten låses.**

Strukturelt avvik: monorepoet importerer flatt (`from shared_arch import`, `REPO = parents[1]`,
configrot `utils/dfine_config`); branchen pakkebasert (`from tools.model_surgery.shared_arch import`,
`parents[2]`). Koden er vendorert og restrukturert, ikke delt. Adaptere må håndtere begge.

---

## 03 — Checkpoints og proveniens

`models/checkpoints/` og `models/original_weights/` er **tomme**.
`model_manager.py:36` → `DEFAULT_TRAINED_MODEL_S3_KEY = ...dfine_0.48.pth` (Digital Ocean Spaces).

Engine-navnemønster: `dfine_{task}_{size}_{precision}_miou{miou}_{gpu}_trt{version}.engine`

| Artefakt | Sted | Størrelse | Status |
|---|---|---|---|
| `dfine_segpose_x_fp16_miou48_..._trt1013035.engine` | monorepo | 153.9 MB | PRODUKSJON |
| `dfine_segpose_x_fp16_miou48_..._trt109034.engine` | monorepo | 154.5 MB | eldre TRT |
| `best_modelsurgery.pth` | `Traning/` | 738.8 MB | UVERIFISERT |
| `dfine_segmentation_640_fp16_map73_rtx5070ti_trt10.15.1.engine` | `Traning/` | 199.1 MB | ANNEN MODELL |

Lokalt engine heter `map73` (deteksjons-mAP), produksjon heter `miou48` (segmenterings-mIoU).
Ulik oppgave, ulik TRT-versjon (10.15.1 mot 10.13.0.35). Ingen grunn til å anta at
`best_modelsurgery.pth` er produksjonsvekten.

---

## 04 — Gjenbruk

`tools/model_surgery/infer_video_singlepass_x.py` (32 KB) er beste base for Fase 3:
- `overlay_seg` med alfa, `draw_pose` med skjelett
- IoU-matrise, grådig matching, senterbasert fallback
- `--profile`-flagg
- Riktige forvalg: `--seg-num-classes 7`, `--seg-feature-dim 384`, `--image-size 640`

Andre: `parity_onnx_trt.py`, `build_trt_engine.py`, `onnx_singlepass_inf.py`,
monorepoets `cli/native_tensorrt_video_compare.py`.

### Mangler

| Mangler | Hvorfor |
|---|---|
| Fast frame-sett | Verktøyet dekoder video fortløpende; benchmark krever identiske frames |
| Rå output-dump | Ingen JSON/NPZ; må kjøre modellene på nytt ved hver viz-endring |
| Crosshair + hit-beslutning | Kaller ikke `HitDetector` — selve forskningsspørsmålet |
| Adapterlag | Modellbygging hardkodet mot `SharedBackboneDualDecoder` |
| Latency-percentiler | `--profile` gir ikke p50/p95 splittet på pre/inf/post |
| Side-by-side + HTML-rapport | Finnes ikke |
| Metrikker | Ingen IoU, OKS/PCK, confusion matrix, hit-FP/FN |

---

## 05 — Foreslått katalogstruktur

```
benchmark/
├── datasets/
│   └── tagtwo_v1/
│       ├── manifest.json      # scene_id, frame_id, crosshair, forventet hit
│       └── frames/            # uforanderlige PNG, hash i manifest
├── adapters/
│   ├── base.py                # ModelAdapter-protokoll → felles schema
│   ├── baseline_singlepass.py # dagens delte modell
│   └── ...                    # én fil per kandidat
├── harness/
│   ├── run.py                 # adapter x frame-sett → rå output
│   ├── schema.py              # validering
│   ├── decide.py              # wrapper rundt monorepoets HitDetector
│   └── timing.py              # p50/p95 per fase
├── metrics/
├── viz/
│   ├── overlay.py             # løftet fra infer_video_singlepass_x.py
│   ├── compare.py             # Original | Baseline | Kandidat
│   └── report.py              # statisk HTML
└── runs/
    └── <run_id>/
        ├── raw/               # NPZ per frame
        ├── results.json
        └── report.html
```

Tillegg til schemaet i briefen:
- `decision.source` = `mask` | `pose_fallback` — uten dette kan hit-regresjon ikke tilskrives riktig hode
- `schema_version` — så eldre kjøringer kan leses etter endringer

---

## 06 — Risiko og ukjente

**UAVKLART: letterbox ser ut til å mangle.** Finner ingen aspect-bevarende padding, verken i
`dfine_segmentation.py` (`F.interpolate`) eller repo-verktøyet (`cv2.resize`, `INTER_LINEAR`).
Tyder på direkte resize til 640x640 med forvrengt aspect. Hver kandidat må få nøyaktig samme
forvrengning, ellers måler benchmarken preprocessing. Må bekreftes ved kjøring.

- **Python eller Rust i produksjon?** Både `run_engine.py` og `rust/src/main.rs` er entrypoints.
- **Hailo i scope?** Se seksjon 02.
- **Ingen evalueringsdata ennå.** Vet ikke hvor Tagtwo-videoen ligger eller om crosshair er logget synkront.
- **D-FINE-seg-lisens ikke sjekket** — community-fork, ikke offisiell utgivelse.
- **mIoU 48 er ikke kontekstualisert** — uten per-klasse-tall vet vi ikke om armer/bein trekker ned.

---

## 07 — Billigste vei til første overlay

```bash
# Checkpoint-proveniens (leser hovedcheckouten, endrer ingenting)
sha256sum /home/berna/D-FINE-NEWBRINGER/Traning/best_modelsurgery.pth

# Første baseline-overlay, 200 frames, med profilering
cd /home/berna/D-FINE-NEWBRINGER-benchmark
python tools/model_surgery/infer_video_singlepass_x.py \
  --det-config  /home/berna/D-FINE-NEWBRINGER/Traning/dfine_hgnetv2_x_obj2coco.yml \
  --pose-config /home/berna/D-FINE-NEWBRINGER/Traning/dfine_hgnetv2_x_obj2coco_detrpose_paper.yml \
  --merged-ckpt /home/berna/D-FINE-NEWBRINGER/Traning/best_modelsurgery.pth \
  --input  <tagtwo_video.mp4> \
  --out    runs/baseline_smoke.mp4 \
  --image-size 640 --seg-num-classes 7 --seg-feature-dim 384 \
  --score-thr 0.5 \
  --max-frames 200 --profile
```

Merk `--score-thr 0.5`, ikke verktøyets forvalg `0.35` — samkjører med produksjonens
`confidence_threshold`. Denne kjøringen svarer på tre ting: om `best_modelsurgery.pth` laster inn i
den delte arkitekturen, om segmenteringen gir plausible 7-klasses masker, og hva grunnlatency er.

---

## 08 — Anbefaling: mål før du optimaliserer

Hit-beslutningen har to veier: maskens argmax, og posebasert torso-fallback ved okklusjon.
**Vi vet ikke hvilken som avgjør flest faktiske hits.** Uten det tallet er valget mellom en
segmenterings- og en posekandidat gjetning. Første slice bør logge `decision.source` og telle
fordelingen. Noen timers arbeid, gjør resten retningsbestemt.

Når tallet foreligger:

1. **Segmentering, hvis masken dominerer.** mIoU 48 er svakeste ledd og ligger rett på
   beslutningsstien. Dyrest — kandidat må trenes til samme sju klasser — men høyest gevinst.
2. **Pose, hvis fallback dominerer.** Offisielle DETRPose-S/M kan sammenlignes uten omtrening.
   Treffer direkte på okklusjonshåndtering. Billigst.
3. **Deteksjon sist.** D-FINE-X med obj2coco er allerede største variant med sterkeste førtrening.
   Lite hodrom. DEIMv2/ECDet interessante, men ikke først.

**Advarsel om Fase 5:** EdgeCrafter og D-FINE-seg leverer *instance*-segmentering; produksjon
trenger *semantisk* body-part-segmentering i sju klasser. Ikke samme oppgave — direkte metrisk
sammenligning uten omtrening vil være misvisende.
