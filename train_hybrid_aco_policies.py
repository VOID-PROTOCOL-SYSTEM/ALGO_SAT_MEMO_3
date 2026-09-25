import argparse
import itertools
import multiprocessing as mp
import random as pyrandom

import networkx as nx
import torch
import torch.nn as nn

WING_COLS, WING_ROWS = 10, 10
N_SUPPLIES = 30
CAPACITY = 5


def _neighbours(cols, rows, c, r):
    for dc, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        nc, nr = c + dc, r + dr
        if 0 <= nc < cols and 0 <= nr < rows:
            yield nc, nr


def _build_wing(cols, rows, rng):
    visited = [[False] * rows for _ in range(cols)]
    g = nx.Graph()
    for c in range(cols):
        for r in range(rows):
            g.add_node((c, r))

    def carve(c, r):
        visited[c][r] = True
        dirs = list(_neighbours(cols, rows, c, r))
        rng.shuffle(dirs)
        for nc, nr in dirs:
            if not visited[nc][nr]:
                g.add_edge((c, r), (nc, nr), weight=1)
                carve(nc, nr)

    carve(0, 0)
    return g


def get_facility(seed):
    s = int(seed)
    n_wings = 2 + (s % 3)
    wings = [_build_wing(WING_COLS, WING_ROWS, pyrandom.Random(s * 31 + w * 7919))
             for w in range(n_wings)]

    junctions = []
    for w in range(n_wings - 1):
        jr = pyrandom.Random(s * 17 + w * 5003)
        rows_avail = list(range(2, WING_ROWS - 2))
        jr.shuffle(rows_avail)
        r1, r2 = sorted(rows_avail[:2])
        junctions.append(((w, WING_COLS - 1, r1), (w + 1, 0, r1)))
        junctions.append(((w, WING_COLS - 1, r2), (w + 1, 0, r2)))

    shaft = (0, 0, 0)
    exit_a = (n_wings - 1, WING_COLS - 1, WING_ROWS - 1)
    exit_b = (n_wings - 1, WING_COLS - 1, 0)

    srng = pyrandom.Random(s * 13 + 42)
    reserved = {shaft, exit_a, exit_b}
    for a, b in junctions:
        reserved.add(a)
        reserved.add(b)

    tier1, tier2 = [], []
    for w, wg in enumerate(wings):
        t1 = [(w, c, r) for (c, r) in wg.nodes()
              if wg.degree((c, r)) == 1 and (w, c, r) not in reserved]
        t2 = [(w, c, r) for (c, r) in wg.nodes()
              if wg.degree((c, r)) == 2 and (w, c, r) not in reserved]
        srng.shuffle(t1)
        srng.shuffle(t2)
        tier1.append(t1)
        tier2.append(t2)

    empty_wing = srng.choice(range(1, n_wings)) if n_wings >= 3 else None
    supply_wings = [w for w in range(n_wings) if w != empty_wing]

    supplies = []
    for tier in (tier1, tier2):
        idx = {w: 0 for w in supply_wings}
        while len(supplies) < N_SUPPLIES:
            added = False
            for w in supply_wings:
                if len(supplies) >= N_SUPPLIES:
                    break
                while idx[w] < len(tier[w]):
                    n = tier[w][idx[w]]
                    idx[w] += 1
                    if n not in supplies:
                        supplies.append(n)
                        added = True
                        break
            if not added:
                break
        if len(supplies) >= N_SUPPLIES:
            break
    supplies = supplies[:N_SUPPLIES]

    for w, wg in enumerate(wings):
        if w == 1:
            for (c1, r1), (c2, r2) in list(wg.edges()):
                wg[c1, r1][c2, r2]['weight'] = 1 + max(c1, c2) // 3
        elif w >= 2:
            cr = pyrandom.Random(s * 41 + w * 3331)
            for (c1, r1), (c2, r2) in list(wg.edges()):
                wg[c1, r1][c2, r2]['weight'] = cr.randint(1, 5)

    G = nx.Graph()
    for w, wg in enumerate(wings):
        for (a, b, d) in wg.edges(data=True):
            G.add_edge((w,) + a, (w,) + b, weight=d['weight'])
    for a, b in junctions:
        G.add_edge(a, b, weight=1)

    prng = pyrandom.Random(s * 977 + 13)
    masses = [prng.choice([1, 2, 3]) for _ in supplies]
    values = [prng.choice([1, 2, 3, 4, 5]) for _ in supplies]

    return dict(G=G, n_wings=n_wings, shaft=shaft, exits=[exit_a, exit_b],
                supplies=supplies, masses=masses, values=values, capacity=CAPACITY)


def best_order(units, trip_cost):
    if len(units) <= 1:
        return list(units)
    return list(min(itertools.permutations(units), key=trip_cost))


def make_instance(seed):
    """Build one self-contained training/eval instance from a seed."""
    fac = get_facility(seed)
    key_nodes = [fac['shaft']] + fac['supplies'] + fac['exits']
    DIST = {n: nx.single_source_dijkstra_path_length(fac['G'], n, weight='weight')
            for n in key_nodes}

    def dist(u, v):
        return DIST[u][v]

    SHAFT, EXITS, SUPPLIES = fac['shaft'], fac['exits'], fac['supplies']
    MASS = {u: m for u, m in zip(fac['supplies'], fac['masses'])}
    VALUE = {u: v for u, v in zip(fac['supplies'], fac['values'])}
    CAP = fac['capacity']
    EXIT_LEG = min(dist(SHAFT, e) for e in EXITS)

    def trip_cost(trip):
        here, total, load = SHAFT, 0.0, 0
        for u in trip:
            total += (1 + load) * dist(here, u)
            load += MASS[u]
            here = u
        total += (1 + load) * dist(here, SHAFT)
        return total

    def _best_order(units):
        return best_order(units, trip_cost)

    def plan_cost(plan):
        return sum(trip_cost(t) for t in plan) + EXIT_LEG

    def plan_value(plan):
        return sum(VALUE[u] for t in plan for u in t)

    def exemplar_a_nearest_fill(pool, budget):
        remaining, plan, spent = list(pool), [], 0.0
        while remaining:
            trip, here, load = [], SHAFT, 0
            while True:
                fits = [u for u in remaining if u not in trip and load + MASS[u] <= CAP]
                if not fits:
                    break
                u = min(fits, key=lambda x: dist(here, x))
                trip.append(u)
                load += MASS[u]
                here = u
            if not trip:
                break
            trip = _best_order(trip)
            c = trip_cost(trip)
            if spent + c + EXIT_LEG > budget:
                break
            spent += c
            plan.append(trip)
            for u in trip:
                remaining.remove(u)
        return plan

    full_plan = exemplar_a_nearest_fill(SUPPLIES, budget=float('inf'))
    FULL_EXTRACTION_COST = plan_cost(full_plan)
    BUDGET = round(FULL_EXTRACTION_COST * 0.60)

    return dict(
        seed=seed, n_wings=fac['n_wings'], SHAFT=SHAFT, EXITS=EXITS, SUPPLIES=SUPPLIES,
        MASS=MASS, VALUE=VALUE, CAP=CAP, EXIT_LEG=EXIT_LEG, BUDGET=BUDGET,
        dist=dist, trip_cost=trip_cost, plan_cost=plan_cost, plan_value=plan_value,
        best_order=_best_order,
    )


def seed_for_wings(rng, n_wings):
    """Draw a random seed that lands on the requested n_wings bucket
    (n_wings = 2 + seed % 3)."""
    target_mod = n_wings - 2
    while True:
        s = rng.randint(1, 999_999_999)
        if s % 3 == target_mod:
            return s

class Actor(nn.Module):
    def __init__(self, in_dim=7, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats):
        return self.net(feats).squeeze(-1)


class Critic(nn.Module):
    def __init__(self, in_dim=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats):
        return self.net(feats).squeeze(-1)

def candidate_features(here, cand, load, cap, spent, budget, dist, mass_of, value_of,
                        pheromone, n_remaining, n_total, dist_scale):
    if cand == "STOP":
        return [0.0, 0.0, 0.0, load / cap, spent / max(budget, 1e-9), 1.0,
                n_remaining / max(n_total, 1)]
    d = max(dist(here, cand), 1e-9)
    vpm = value_of[cand] / mass_of[cand]
    return [
        vpm / 5.0,
        d / max(dist_scale, 1e-9),
        mass_of[cand] / cap,
        load / cap,
        spent / max(budget, 1e-9),
        pheromone.get((here, cand), 1.0),
        n_remaining / max(n_total, 1),
    ]


def state_features(spent, budget, load, cap, remaining, value_of, mass_of, n_total):
    mean_vpm = (sum(value_of[u] / mass_of[u] for u in remaining) / len(remaining)
                if remaining else 0.0)
    return [spent / max(budget, 1e-9), load / cap, len(remaining) / max(n_total, 1),
            mean_vpm / 5.0]

def construct_plan(inst, actor, critic, pheromone, dist_scale, mode="sample"):
    """mode: 'sample' (stochastic, records a raw trajectory for later replay),
             'greedy' (deterministic argmax, no trajectory recorded)."""
    SUPPLIES, CAP, BUDGET = inst['SUPPLIES'], inst['CAP'], inst['BUDGET']
    dist, trip_cost, best_order_fn = inst['dist'], inst['trip_cost'], inst['best_order']
    SHAFT, MASS, VALUE, EXIT_LEG = inst['SHAFT'], inst['MASS'], inst['VALUE'], inst['EXIT_LEG']

    remaining = set(SUPPLIES)
    n_total = len(SUPPLIES)
    plan, spent = [], 0.0
    trajectory = []  # list of (state_feat, [cand_feats...], chosen_idx)

    with torch.no_grad():
        while remaining:
            trip, here, load = [], SHAFT, 0
            while True:
                candidates = [u for u in remaining if u not in trip and load + MASS[u] <= CAP]
                options = list(candidates) + (["STOP"] if trip else [])
                if not options:
                    break

                feats = torch.tensor(
                    [candidate_features(here, c, load, CAP, spent, BUDGET, dist, MASS, VALUE,
                                         pheromone, len(remaining), n_total, dist_scale)
                     for c in options],
                    dtype=torch.float32)
                probs = torch.softmax(actor(feats), dim=0)

                if mode == "greedy":
                    idx = int(torch.argmax(probs).item())
                else:
                    idx = int(torch.multinomial(probs, 1).item())
                    s_feat = state_features(spent, BUDGET, load, CAP, remaining, VALUE, MASS, n_total)
                    trajectory.append((s_feat, feats.tolist(), idx))

                pick = options[idx]
                if pick == "STOP":
                    break
                trip.append(pick)
                load += MASS[pick]
                here = pick

            if not trip:
                break
            trip = best_order_fn(trip)
            c = trip_cost(trip)
            if spent + c + EXIT_LEG > BUDGET:
                break
            spent += c
            plan.append(trip)
            for u in trip:
                remaining.discard(u)

    return plan, trajectory


def update_pheromone(pheromone, iter_plans, shaft, evaporation, elite_plan, elite_value, elitist_weight):
    for k in pheromone:
        pheromone[k] *= (1 - evaporation)
    for plan, v in iter_plans:
        if v <= 0:
            continue
        for trip in plan:
            here = shaft
            for u in trip:
                pheromone[(here, u)] = pheromone.get((here, u), 1.0) + v / 100.0
                here = u
            pheromone[(here, "STOP")] = pheromone.get((here, "STOP"), 1.0) + v / 100.0
    if elite_plan:
        for trip in elite_plan:
            here = shaft
            for u in trip:
                pheromone[(here, u)] = pheromone.get((here, u), 1.0) + elitist_weight * (elite_value / 100.0)
                here = u
            pheromone[(here, "STOP")] = pheromone.get((here, "STOP"), 1.0) + elitist_weight * (elite_value / 100.0)

def rollout_worker(args):
    seed, actor_state, critic_state, n_ants, n_waves = args
    torch.set_num_threads(1)  # avoid thread oversubscription across processes

    actor = Actor()
    actor.load_state_dict(actor_state)
    actor.eval()
    critic = Critic()
    critic.load_state_dict(critic_state)
    critic.eval()

    inst = make_instance(seed)
    dist_scale = inst['EXIT_LEG']
    pheromone = {}

    trajectories = []       # list of (trajectory, normalized_value)
    all_plans = []          # list of (plan, raw_value), across ALL waves -- for the group-relative baseline
    total_value = max(sum(inst['VALUE'].values()), 1)

    ants_per_wave = max(n_ants // n_waves, 1)
    for wave in range(n_waves):
        wave_plans = []
        for _ in range(ants_per_wave):
            plan, traj = construct_plan(inst, actor, critic, pheromone, dist_scale, mode="sample")
            v = inst['plan_value'](plan)
            wave_plans.append((plan, v))
            all_plans.append((plan, v))
            if traj:
                trajectories.append((traj, v / total_value))

        if wave_plans:
            wave_best_plan, wave_best_value = max(wave_plans, key=lambda pv: pv[1])
            update_pheromone(pheromone, wave_plans, inst['SHAFT'], evaporation=0.35,
                              elite_plan=wave_best_plan, elite_value=wave_best_value, elitist_weight=2.0)

    best_plan, best_value = max(all_plans, key=lambda pv: pv[1]) if all_plans else ([], 0)

    group_mean = (sum(v for _, v in all_plans) / len(all_plans) / total_value) if all_plans else 0.0

    return trajectories, best_plan, best_value, seed, group_mean

def evaluate(actor, critic, val_seeds):
    """Greedy rollout on held-out seeds -- no gradient, just a sanity score."""
    scores = []
    for seed in val_seeds:
        inst = make_instance(seed)
        dist_scale = inst['EXIT_LEG']
        plan, _ = construct_plan(inst, actor, critic, {}, dist_scale, mode="greedy")
        v = inst['plan_value'](plan)
        total = max(sum(inst['VALUE'].values()), 1)
        scores.append(v / total)
    return sum(scores) / len(scores)


def train_policy_for_wing_count(n_wings, n_instances, n_ants, n_workers, lr, seed, out_path,
                                 n_waves=4, entropy_start=0.05, entropy_end=0.005,
                                 baseline_blend=0.5):
    print(f"\n=== Training policy for n_wings = {n_wings} ===")
    rng = pyrandom.Random(seed)
    torch.manual_seed(seed)

    actor = Actor()
    critic = Critic()
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=lr)

    val_seeds = [seed_for_wings(pyrandom.Random(seed + 999 + i), n_wings) for i in range(8)]

    batch_size = n_workers  # one instance per worker per training step
    n_steps = max(n_instances // batch_size, 1)

    pool = mp.Pool(processes=n_workers)
    try:
        for step in range(n_steps):
            seeds_batch = [seed_for_wings(rng, n_wings) for _ in range(batch_size)]
            actor_state = {k: v.clone() for k, v in actor.state_dict().items()}
            critic_state = {k: v.clone() for k, v in critic.state_dict().items()}
            args = [(s, actor_state, critic_state, n_ants, n_waves) for s in seeds_batch]

            results = pool.map(rollout_worker, args)
            frac = step / max(n_steps - 1, 1)
            entropy_coef = entropy_start + (entropy_end - entropy_start) * frac

            total_loss = torch.tensor(0.0)
            n_traj = 0
            for trajectories, _, _, _, group_mean in results:
                for traj, v_norm in trajectories:
                    n_traj += 1
                    state_feats = torch.tensor([t[0] for t in traj], dtype=torch.float32)
                    values = critic(state_feats)
                    returns = torch.full_like(values, float(v_norm))

                    critic_advantage = (returns - values).detach()
                    group_advantage = torch.full_like(values, float(v_norm - group_mean)).detach()
                    advantage = (baseline_blend * group_advantage
                                 + (1 - baseline_blend) * critic_advantage)

                    log_probs = []
                    entropies = []
                    for (_, cand_feats, chosen_idx) in traj:
                        cand_t = torch.tensor(cand_feats, dtype=torch.float32)
                        probs = torch.softmax(actor(cand_t), dim=0)
                        log_probs.append(torch.log(probs[chosen_idx] + 1e-9))
                        entropies.append(-(probs * torch.log(probs + 1e-9)).sum())
                    log_probs_t = torch.stack(log_probs)
                    entropy_t = torch.stack(entropies).mean()

                    actor_loss = -(log_probs_t * advantage).sum()

                    critic_loss = nn.functional.mse_loss(values, returns)
                    total_loss = total_loss + actor_loss + 0.5 * critic_loss - entropy_coef * entropy_t

            if n_traj > 0:
                opt.zero_grad()
                (total_loss / n_traj).backward()
                opt.step()

            if step % max(n_steps // 20, 1) == 0 or step == n_steps - 1:
                val_score = evaluate(actor, critic, val_seeds)
                print(f"  step {step:5d}/{n_steps}  "
                      f"train_instances_seen={min((step + 1) * batch_size, n_instances):5d}  "
                      f"held_out_value_fraction={val_score:.3f}")
    finally:
        pool.close()
        pool.join()

    torch.save({'actor': actor.state_dict(), 'critic': critic.state_dict(),
                'n_wings': n_wings}, out_path)
    print(f"  saved -> {out_path}")
    return actor, critic

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=int, default=1500,
                         help="training instances PER wing-count bucket")
    parser.add_argument("--ants", type=int, default=16,
                         help="rollouts per instance")
    parser.add_argument("--waves", type=int, default=4,
                         help="pheromone waves per instance (ants split evenly across these)")
    parser.add_argument("--workers", type=int, default=max(mp.cpu_count() - 2, 1))
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--baseline-blend", type=float, default=0.5,
                         help="0.0 = pure critic baseline, 1.0 = pure group-relative baseline")
    parser.add_argument("--entropy-start", type=float, default=0.05)
    parser.add_argument("--entropy-end", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"CPU count: {mp.cpu_count()}, using {args.workers} worker processes")

    for n_wings, out_path in [(2, "policy_wings2.pt"),
                               (3, "policy_wings3.pt"),
                               (4, "policy_wings4.pt")]:
        train_policy_for_wing_count(
            n_wings=n_wings,
            n_instances=args.instances,
            n_ants=args.ants,
            n_workers=args.workers,
            lr=args.lr,
            seed=args.seed + n_wings,   # different seed stream per bucket
            out_path=out_path,
            n_waves=args.waves,
            entropy_start=args.entropy_start,
            entropy_end=args.entropy_end,
            baseline_blend=args.baseline_blend,
        )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()