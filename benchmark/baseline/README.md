# Frozen production baseline

`production_manifest.json` is the machine-readable identity and contract for the ModelSurgery
baseline. Large artifacts stay at the absolute paths recorded in the manifest.

Validate local artifact identity and the model IO contract with:

```bash
python benchmark/baseline/audit_manifest.py
```

The DigitalOcean Spaces checkpoint was downloaded to a temporary directory on 3 September 2026.
Its size, SHA-256, and a byte-for-byte comparison prove that it is identical to
`Traning/best_modelsurgery.pth`. The temporary copy is not part of the baseline and may be deleted.

The 10.15 reference TensorRT engine was deserialized and its six bindings matched the ONNX
contract. The production 10.13 plan cannot be deserialized by the 10.15 runtime installed on this
host; that limitation is retained as an explicit unknown instead of inferring its bindings.
