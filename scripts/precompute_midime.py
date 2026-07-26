"""
Run this ONCE, locally, with your real MusicVAE checkpoint. It trains MidiMe
on all 6 chorus tracks together (one shared 4-D latent space covering all of
them) and saves the resulting weights to commit into the repo.

Why train on all 6 jointly instead of retraining per-track in the app:
  - No training code needs to run in the deployed Streamlit process at all.
  - Training has run-to-run variance (random init, Adam noise) -- baking it
    in once means the live demo always shows the exact same, rehearsed result.
  - Training all 6 together gives ONE shared 4-D coordinate system. If each
    track got its own separate training run instead, slider 1 could mean
    "brighter" for track A and something unrelated for track B, since each
    run finds its own arbitrary latent axes. One joint space keeps a given
    slider's meaning roughly consistent no matter which track you start from.

Usage:
    python scripts/precompute_midime.py \\
        --checkpoint weights/musicvae.pt \\
        --chorus_dir assets/chorus_midis \\
        --output weights/midime_offline.pt \\
        --epochs 3000
"""
import argparse
import glob
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.loader import load_musicvae_checkpoint
from musicvae import midi_to_token_sequence, tokens_to_one_hot, train_midime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to trained MusicVAE weights")
    parser.add_argument("--chorus_dir", default="assets/chorus_midis")
    parser.add_argument("--output", default="weights/midime_offline.pt")
    parser.add_argument("--latent_size", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3000, help="one-time precompute, so it's cheap to run more epochs than a live per-session run would afford")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--free_bits", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = load_musicvae_checkpoint(args.checkpoint, device=args.device)

    midi_paths = sorted(glob.glob(os.path.join(args.chorus_dir, "*.mid")))
    if not midi_paths:
        raise FileNotFoundError(f"No .mid files found in {args.chorus_dir}")
    print(f"Found {len(midi_paths)} chorus tracks:")
    for p in midi_paths:
        print(f"  {os.path.basename(p)}")

    tokens_list = [midi_to_token_sequence(p) for p in midi_paths]
    one_hot = torch.stack([torch.tensor(tokens_to_one_hot(t)) for t in tokens_list]).to(args.device)

    with torch.no_grad():
        user_mu, _ = model.encode(one_hot)
    print(f"\nEncoded {user_mu.shape[0]} tracks -> mu shape {tuple(user_mu.shape)}")

    print(f"\nTraining MidiMe jointly on all {user_mu.shape[0]} tracks for {args.epochs} epochs...")
    midime_model = train_midime(
        user_mu,
        latent_size=args.latent_size,
        hidden_size=args.hidden_size,
        epochs=args.epochs,
        lr=args.lr,
        free_bits=args.free_bits,
        device=args.device,
    )

    # sanity check: report final per-track positions in the shared space, and
    # confirm they're actually distinct (not collapsed to one point)
    with torch.no_grad():
        w_mu_real, _ = midime_model.encode(user_mu)
    print("\nFinal per-track positions in shared 4-D space:")
    for path, w in zip(midi_paths, w_mu_real):
        print(f"  {os.path.basename(path)}: {w.numpy().round(3)}")

    pairwise_dist = torch.cdist(w_mu_real, w_mu_real)
    off_diagonal = pairwise_dist[~torch.eye(len(w_mu_real), dtype=torch.bool)]
    print(f"\nMin pairwise distance between tracks: {off_diagonal.min().item():.4f}")
    if off_diagonal.min().item() < 0.1:
        print("WARNING: some tracks are very close together in the shared space -- "
              "sliders may not clearly distinguish them. Consider more epochs or a smaller free_bits.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # per-track bpm, read straight from each file, so the app can decode+render
    # generated samples at a tempo consistent with that track without needing
    # the original mid at hand
    import pretty_midi
    track_bpms = []
    for p in midi_paths:
        _, tempi = pretty_midi.PrettyMIDI(p).get_tempo_changes()
        track_bpms.append(float(tempi[0]) if len(tempi) else 120.0)

    bundle = {
        "midime_state_dict": midime_model.state_dict(),
        "latent_size": args.latent_size,
        "hidden_size": args.hidden_size,
        "input_size": user_mu.shape[-1],
        "track_names": [os.path.splitext(os.path.basename(p))[0] for p in midi_paths],
        "w_mu_real": w_mu_real,       # [6, latent_size] -- per-track slider starting point
        "w_center": w_mu_real.mean(dim=0),  # global average, if a mode wants a neutral start
        "bpm": track_bpms,             # per-track, same order as track_names
    }
    torch.save(bundle, args.output)
    print(f"\nSaved MidiMe bundle to {args.output}")
    print("Bundle contains: weights + per-track w positions + bpm -- runtime needs zero encoding.")
    print("Commit this file to the repo -- the app loads it directly, no training at runtime.")


if __name__ == "__main__":
    main()
