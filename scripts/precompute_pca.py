"""
Run this ONCE, locally, with your real MusicVAE checkpoint -- produces
weights/pca_basis.pt: the 20 PCA slider directions + their scale, fit over a
broad sample of POP909 chunks. Needed because the 6 committed chorus files
are single chunks each -- nowhere near the >=20 samples PCA needs, and not
representative of the dataset's overall spread anyway.

sklearn is only a dependency for THIS script -- the deployed app never needs
it (see inference.loader.SimplePCA, which just holds the two arrays this
script produces).

Usage:
    python scripts/precompute_pca.py \
        --checkpoint musicvae_trained.pt \
        --pop909-root /path/to/POP909-Dataset/POP909 \
        --output weights/pca_basis.pt \
        --num-songs 150
"""
import argparse
import glob
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.loader import encode_dataset, load_musicvae_checkpoint
from musicvae.tokenizer import split_midi_into_chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to trained MusicVAE weights")
    parser.add_argument("--pop909-root", required=True, help="path to POP909-Dataset/POP909")
    parser.add_argument("--output", default="weights/pca_basis.pt")
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument(
        "--num-songs", type=int, default=150,
        help="how many songs to sample chunks from -- more gives a more representative "
             "basis but takes longer to encode; 100-200 is plenty for stable components",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    model = load_musicvae_checkpoint(args.checkpoint, device=args.device)

    song_dirs = sorted(glob.glob(os.path.join(args.pop909_root, "*")))
    song_dirs = [d for d in song_dirs if os.path.isdir(d)]
    if len(song_dirs) > args.num_songs:
        song_dirs = random.sample(song_dirs, args.num_songs)
    print(f"Sampling chunks from {len(song_dirs)} songs...")

    all_chunks = []
    skipped = 0
    for song_dir in song_dirs:
        song_id = os.path.basename(song_dir)
        midi_path = os.path.join(song_dir, f"{song_id}.mid")
        if not os.path.exists(midi_path):
            skipped += 1
            continue
        try:
            chunks, _ = split_midi_into_chunks(midi_path)
            all_chunks.extend(chunks)
        except ValueError:
            skipped += 1  # no notes / no instrument tracks -- same as extract_choruses.py handling

    print(f"Collected {len(all_chunks)} chunks total ({skipped} songs skipped)")
    if len(all_chunks) < args.n_components:
        raise ValueError(
            f"Only got {len(all_chunks)} chunks, need >= {args.n_components}. "
            f"Increase --num-songs or lower --n-components."
        )

    mu_all = encode_dataset(model, all_chunks, device=args.device)
    print(f"Encoded -> mu_all shape {tuple(mu_all.shape)}")

    from sklearn.decomposition import PCA
    pca = PCA(n_components=args.n_components)
    pca.fit(mu_all.numpy())

    total_variance_explained = pca.explained_variance_ratio_.sum()
    print(f"Top {args.n_components} components explain {total_variance_explained:.1%} of variance")
    if total_variance_explained < 0.5:
        print(
            "WARNING: less than 50% of variance captured -- sliders may feel like they "
            "barely change anything. Consider more --num-songs, or accept that the z-space "
            "genuinely has high-dimensional structure PCA can't fully compress into 20 axes."
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(
        {
            "components": pca.components_.astype(np.float32),
            "explained_variance": pca.explained_variance_.astype(np.float32),
            "num_chunks_used": len(all_chunks),
            "num_songs_used": len(song_dirs) - skipped,
        },
        args.output,
    )
    print(f"\nSaved PCA basis to {args.output}")
    print("Commit this file to the repo -- no sklearn needed at runtime to use it.")


if __name__ == "__main__":
    main()
