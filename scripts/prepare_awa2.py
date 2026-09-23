"""One-time preparation of AwA2 for the MCBM-paper protocol (raw images).

Paper (arXiv:2506.04877, App. F.2): y = animal class (50), c = 20 of the 85
human-annotated attributes covering fundamental appearance and morphology
features, n_y = remaining 65 attributes, n_ybar = empty. Continuous
attributes are binarized with a threshold of 0.5 (Xian et al. 2017
preprocessing, original_att scale 0-100 -> /100 > 0.5).

Concepts: the paper does not enumerate the 20 attributes. We take the first
20 NON-DEGENERATE attributes within the appearance+morphology block
(predicates 0-32: colors, patterns, texture, size, body parts, teeth,
horns/claws/tusks). "Non-degenerate" = the binarized class-level column is
not constant across the 50 classes (e.g. 'red'/'yellow' are all-zero at the
0.5 threshold and are skipped, since a constant column is not a learnable or
intervenable concept). This choice is documented in the results report.

Split: fixed across seeds (as for CIFAR-10 and CUB). We reuse the verified
xlsa17-derived split saved in data/awa2/awa2_cibm_code (train+val = 25,379
images for training, test = 11,943), so results are directly comparable
with the CIBM-code protocol. Labels/order come from res101.mat; image paths
are remapped to the local JPEGImages tree.
"""
import json
import os

import numpy as np
from scipy.io import loadmat

ROOT = "/research/hal-afsharim/minimal_cbm/data/awa2"
OUT = os.path.join(ROOT, "mcbm_awa2")
N_CONCEPTS = 20
# Appearance+morphology block of AwA2 predicates (colors, patterns, texture,
# size, body parts, teeth, horns/claws/tusks). We select the first 20
# NON-degenerate columns from this block.
MORPH_BLOCK = list(range(0, 33))


def main():
    os.makedirs(OUT, exist_ok=True)
    att = loadmat(os.path.join(ROOT, "xlsa17/data/AWA2/att_splits.mat"))
    res = loadmat(os.path.join(ROOT, "xlsa17/data/AWA2/res101.mat"))

    labels = res["labels"].ravel().astype(np.int64) - 1          # 0..49
    image_files = [str(f[0][0]) for f in res["image_files"]]
    rel_paths = []
    for p in image_files:
        assert "JPEGImages/" in p, p
        rel_paths.append(p.split("JPEGImages/")[-1])

    # original_att: 85 x 50, 0-100 scale -> binarize at 0.5 after /100
    original_att = att["original_att"]
    assert original_att.shape == (85, 50), original_att.shape
    class_attr = (original_att.T / 100.0 > 0.5).astype(np.int64)  # 50 x 85

    with open(os.path.join(
            ROOT, "AwA2-data/Animals_with_Attributes2/predicates.txt")) as f:
        predicates = [l.split()[-1] for l in f.read().strip().split("\n")]
    assert len(predicates) == 85

    # Select the first N_CONCEPTS non-degenerate morphology/appearance columns
    # (class-level binarized column not constant across the 50 classes).
    col_pos = class_attr.sum(axis=0)  # positive count per attribute over classes
    concept_idx = [j for j in MORPH_BLOCK if 0 < col_pos[j] < 50][:N_CONCEPTS]
    assert len(concept_idx) == N_CONCEPTS, len(concept_idx)
    nuisance_idx = [j for j in range(85) if j not in concept_idx]
    # Reorder class_attr so the first N_CONCEPTS columns are the concepts and
    # the rest are nuisances (loader slices by position).
    order = concept_idx + nuisance_idx
    class_attr = class_attr[:, order]
    predicates = [predicates[j] for j in order]

    cibm = os.path.join(ROOT, "awa2_cibm_code")
    tr = np.load(os.path.join(cibm, "train_indices.npy"))
    va = np.load(os.path.join(cibm, "val_indices.npy"))
    te = np.load(os.path.join(cibm, "test_indices.npy"))
    allidx = np.concatenate([tr, va, te])
    assert len(allidx) == 37322 and len(np.unique(allidx)) == 37322
    train_idx = np.sort(np.concatenate([tr, va]))
    test_idx = np.sort(te)

    for name, idxs in [("train", train_idx), ("test", test_idx)]:
        with open(os.path.join(OUT, f"{name}_list.txt"), "w") as f:
            for i in idxs:
                f.write(f"{rel_paths[i]}\t{labels[i]}\n")
        cls = np.unique(labels[idxs])
        print(name, len(idxs), "images,", len(cls), "classes")

    np.save(os.path.join(OUT, "class_attr_bin.npy"), class_attr)
    manifest = {
        "n_concepts": N_CONCEPTS,
        "concept_attributes": predicates[:N_CONCEPTS],
        "nuisance_attributes": predicates[N_CONCEPTS:],
        "concept_original_indices": [int(j) for j in concept_idx],
        "nuisance_original_indices": [int(j) for j in nuisance_idx],
        "concept_selection": "first 20 non-degenerate morphology/appearance "
                             "predicates (indices 0-32, constant columns skipped)",
        "binarization": "original_att/100 > 0.5",
        "train_size": int(len(train_idx)),
        "test_size": int(len(test_idx)),
        "split_source": "awa2_cibm_code train+val / test indices (xlsa17 order)",
        "images_root": "AwA2-data/Animals_with_Attributes2/JPEGImages",
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("concepts:", predicates[:N_CONCEPTS])

    # sanity: every image file exists
    img_root = os.path.join(ROOT, "AwA2-data/Animals_with_Attributes2/JPEGImages")
    missing = [p for p in rel_paths[:100] if not os.path.exists(os.path.join(img_root, p))]
    assert not missing, missing[:5]
    print("OK ->", OUT)


if __name__ == "__main__":
    main()
