import os

import torch

import train_hybrid_aco_policies as trainmod

SEED = 23092008
CHECKPOINT_DIR = "."


def main():
    inst = trainmod.make_instance(SEED)
    n_wings = inst["n_wings"]

    path = os.path.join(CHECKPOINT_DIR, f"policy_wings{n_wings}.pt")
    actor = trainmod.Actor()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    actor.load_state_dict(checkpoint["actor"] if "actor" in checkpoint else checkpoint)
    actor.eval()

    SUPPLIES = inst["SUPPLIES"]
    MASS, VALUE = inst["MASS"], inst["VALUE"]
    CAP, BUDGET = inst["CAP"], inst["BUDGET"]
    dist = inst["dist"]
    SHAFT = inst["SHAFT"]
    dist_scale = inst["EXIT_LEG"]

    here = SHAFT
    load = 0
    spent = 0.0
    remaining = set(SUPPLIES)
    n_total = len(SUPPLIES)
    pheromone = {}

    options = list(SUPPLIES)

    with torch.no_grad():
        feats = torch.tensor(
            [
                trainmod.candidate_features(
                    here, c, load, CAP, spent, BUDGET, dist, MASS, VALUE,
                    pheromone, len(remaining), n_total, dist_scale,
                )
                for c in options
            ],
            dtype=torch.float32,
        )
        probs = torch.softmax(actor(feats), dim=0)

    rows = [
        (node, VALUE[node], MASS[node], round(float(p), 6))
        for node, p in zip(options, probs)
    ]
    rows.sort(key=lambda r: r[3], reverse=True)

    print(f"n_wings={n_wings}  CAP={CAP}  BUDGET={BUDGET}  EXIT_LEG={dist_scale}\n")
    print(f"{'node':<14} {'value':>6} {'mass':>6} {'P(pick)':>10}")
    for node, value, mass, p in rows:
        print(f"{str(node):<14} {value:>6} {mass:>6} {p:>10.6f}")

    print("\n_rows = [")
    for i in range(0, len(rows), 2):
        chunk = rows[i:i + 2]
        line = ", ".join(f"({n}, {v}, {m}, {p})" for n, v, m, p in chunk)
        print(f"    {line},")
    print("]")


if __name__ == "__main__":
    main()