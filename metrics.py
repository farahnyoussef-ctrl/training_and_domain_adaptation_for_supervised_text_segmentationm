"""Standard Pk metric for text segmentation (Beeferman et al., 1999)."""
import numpy as np


def pk(ref, hyp, k=None):
    """Compute Pk for one document.

    ref, hyp: lists of 0/1 per sentence, 1 = this sentence starts a new segment.
    k: window size; by default half the average reference segment length.
    Returns the fraction of sentence pairs (i, i+k) where the reference and the
    hypothesis disagree on whether both sentences are in the same segment
    (lower is better).
    """
    ref = np.asarray(ref, dtype=int)
    hyp = np.asarray(hyp, dtype=int)
    assert len(ref) == len(hyp), "ref and hyp must have the same length"
    n = len(ref)
    if k is None:
        n_segments = max(int(ref.sum()), 1)
        k = max(2, int(round(n / (2 * n_segments))))
    if n <= k:
        return float("nan")
    ref_ids = np.cumsum(ref)  # segment id of every sentence
    hyp_ids = np.cumsum(hyp)
    ref_same = ref_ids[:-k] == ref_ids[k:]
    hyp_same = hyp_ids[:-k] == hyp_ids[k:]
    return float(np.mean(ref_same != hyp_same))


if __name__ == "__main__":
    ref = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    print("perfect prediction:", pk(ref, ref))
    print("no boundaries predicted:", pk(ref, [1] + [0] * 11))
