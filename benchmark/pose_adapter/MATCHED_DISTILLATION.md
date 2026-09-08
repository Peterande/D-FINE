# Matched pose distillation result

Hungarian matching correctly recovered a synthetic teacher-query permutation with zero loss.
Using the same seed, data, optimizer, adapter architecture and 500 steps as the earlier query-index
run reduced the final distillation term from roughly 0.68 to 0.18.

This did not improve pose accuracy:

| Candidate | 200-image COCO pose AP |
|---|---:|
| Baseline | 47.1 |
| Query-index adapter distillation | 43.4 |
| Hungarian-matched adapter distillation | 43.6 |

Matched checkpoint SHA-256:
`6bd2e3ab1e51c8c9aeda4366ad5c4553a7a96262017dbc2f583224e25e35199c`.

Protected backbone, encoder, detection decoder and segmentation head hashes remained unchanged.
Because the candidate failed the predeclared screen, it was not promoted to expensive full COCO
and all-task evaluation. Conclusion: query permutation was a real loss-design defect, but not the
main limitation. A 1x1 residual adapter alone cannot bridge the encoder representation gap.
