"""
Minimal end-to-end demo of ``deep_neural_networkF.DeepNeuralNetworkRunner``.

Run from this folder (``deep_neural_network/``):

    python python/demo.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deep_neural_networkF import DeepNeuralNetworkRunner  # noqa: E402


@dataclass
class Listing:
    """Same contract as pricer items: text summary + numeric price."""

    summary: str
    price: float


def build_synthetic_split():
    """Small fake listings so training finishes quickly on CPU."""
    train = [
        Listing("compact sedan low miles bluetooth", 18500.0),
        Listing("suv third row towing package awd", 32900.0),
        Listing("electric hatch fast charge nav", 27900.0),
        Listing("pickup crew cab diesel 4x4", 41900.0),
        Listing("coupe manual sport exhaust", 24500.0),
        Listing("minivan sliding doors backup cam", 21900.0),
        Listing("convertible premium audio leather", 35900.0),
        Listing("hatchback fuel efficient city car", 15900.0),
    ]
    val = [
        Listing("awd crossover turbo roof rails", 28900.0),
        Listing("sedan hybrid lane assist warranty", 26500.0),
    ]
    return train, val


def main():
    train, val = build_synthetic_split()
    runner = DeepNeuralNetworkRunner(
        train,
        val,
        n_features=256,
        num_layers=4,
        hidden_size=128,
        dropout=0.1,
        batch_size=4,
        seed=42,
    )
    runner.setup()
    runner.train(epochs=3)

    probe = Listing("awd crossover heated seats remote start", 0.0)
    pred = runner.inference(probe)
    print("\nDemo inference (fake listing, price not used for training this row):")
    print(f"  Summary: {probe.summary!r}")
    print(f"  Predicted price: ${pred:,.2f}")

    also = runner.inference_from_summary("compact hybrid sedan adaptive cruise")
    print(f"  String-only prediction: ${also:,.2f}")


if __name__ == "__main__":
    main()
