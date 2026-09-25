import itertools
import random
import time

import torch

import train_hybrid_aco_policies as trainmod

MAIN_SEED = 23092008
N_SEEDS = 10
CHECKPOINT_DIR = "."
ANTS = 32
ITERATIONS = 20
WING_BUCKETS = [2, 3, 4]

def aco_plan_value(plan, inst):
    return inst["plan_value"](plan)

def aco_plan_cost(plan, inst):
    return inst["plan_cost"](plan)

def copy_plan(plan):
    return [list(t) for t in plan]

def better_plan(a, b, inst):
    if b is None:
        return True
    av = aco_plan_value(a, inst)
    bv = aco_plan_value(b, inst)
    if av > bv:
        return True
    if av == bv:
        return aco_plan_cost(a, inst) < aco_plan_cost(b, inst) - 1e-9
    return False

def add_caching(inst):
    raw_best_order = inst["best_order"]
    raw_trip_cost = inst["trip_cost"]
    best_order_cache, trip_cost_cache = {}, {}

    def cached_best_order(trip):
        key = frozenset(trip)
        cached = best_order_cache.get(key)
        if cached is None:
            cached = tuple(raw_best_order(list(trip)))
            best_order_cache[key] = cached
        return list(cached)

    def cached_trip_cost(trip):
        key = tuple(trip)
        cached = trip_cost_cache.get(key)
        if cached is None:
            cached = raw_trip_cost(list(trip))
            trip_cost_cache[key] = cached
        return cached

    inst = dict(inst)
    inst["best_order"] = cached_best_order
    inst["trip_cost"] = cached_trip_cost
    return inst

def inst_validate(inst, plan):
    CAP, BUDGET, MASS = inst["CAP"], inst["BUDGET"], inst["MASS"]
    seen = set()
    for trip in plan:
        load = 0.0
        for u in trip:
            if u not in inst["SUPPLIES"] or u in seen:
                return False, "bad"
            seen.add(u)
            load += MASS[u]
        if load > CAP + 1e-9:
            return False, "cap"
    if aco_plan_cost(plan, inst) > BUDGET + 1e-9:
        return False, "budget"
    return True, "OK"

def initialise_pheromone(inst):
    pheromone = {}
    for u in inst["SUPPLIES"]:
        pheromone[inst["SHAFT"], u] = 1.0
        pheromone[u, "STOP"] = 1.0
    for u in inst["SUPPLIES"]:
        for v in inst["SUPPLIES"]:
            if u != v:
                pheromone[u, v] = 1.0
    return pheromone

def update_pheromone_elite(pheromone, ranked_plans, shaft, evaporation, elite_plan,
                            elite_value, elitist_weight, min_pheromone=0.2,
                            max_pheromone=12.0, inst=None):
    for key in pheromone:
        pheromone[key] *= 1.0 - evaporation
    ranked = sorted(ranked_plans, key=lambda x: (x[1], -aco_plan_cost(x[0], inst)), reverse=True)
    if ranked:
        elite_count = min(max(2, min(8, len(ranked) // 4)), len(ranked))
        for rank, (plan, value) in enumerate(ranked[:elite_count]):
            if value <= 0:
                continue
            deposit = 0.75 * ((elite_count - rank) / elite_count) * value / 100.0
            for trip in plan:
                here = shaft
                for u in trip:
                    pheromone[(here, u)] = pheromone.get((here, u), 1.0) + deposit
                    here = u
                pheromone[(here, "STOP")] = pheromone.get((here, "STOP"), 1.0) + deposit
    if elite_plan and elite_value > 0:
        deposit = elitist_weight * elite_value / 100.0
        for trip in elite_plan:
            here = shaft
            for u in trip:
                pheromone[(here, u)] = pheromone.get((here, u), 1.0) + deposit
                here = u
            pheromone[(here, "STOP")] = pheromone.get((here, "STOP"), 1.0) + deposit
    for key in pheromone:
        pheromone[key] = min(max(pheromone[key], min_pheromone), max_pheromone)

def or_opt_plan(plan, inst, max_segment=3, first_improvement=True):
    CAP, BUDGET, MASS = inst["CAP"], inst["BUDGET"], inst["MASS"]
    trip_cost = inst["trip_cost"]
    current = copy_plan(plan)
    improved = True
    while improved:
        improved = False
        current_cost = aco_plan_cost(current, inst)
        for src_i, src_trip in enumerate(current):
            n = len(src_trip)
            src_trip_cost = trip_cost(src_trip)
            for seg_len in range(1, min(max_segment, n) + 1):
                for start in range(n - seg_len + 1):
                    segment = src_trip[start:start + seg_len]
                    remainder = src_trip[:start] + src_trip[start + seg_len:]
                    seg_mass = sum(MASS[u] for u in segment)
                    remainder_cost = trip_cost(remainder) if remainder else 0.0
                    for dst_i, dst_trip in enumerate(current):
                        if dst_i == src_i:
                            dst_base, dst_base_cost = remainder, remainder_cost
                        else:
                            dst_base = dst_trip
                            dst_load = sum(MASS[u] for u in dst_base)
                            if dst_load + seg_mass > CAP + 1e-9:
                                continue
                            dst_base_cost = trip_cost(dst_base)
                        for seg_variant in (segment, list(reversed(segment))):
                            for pos in range(len(dst_base) + 1):
                                if dst_i == src_i and pos == start:
                                    continue
                                new_dst = dst_base[:pos] + seg_variant + dst_base[pos:]
                                new_dst_cost = trip_cost(new_dst)
                                if dst_i == src_i:
                                    cand_cost = current_cost - src_trip_cost + new_dst_cost
                                else:
                                    cand_cost = (current_cost - src_trip_cost - dst_base_cost
                                                 + remainder_cost + new_dst_cost)
                                if cand_cost > BUDGET + 1e-9:
                                    continue
                                if cand_cost < current_cost - 1e-9:
                                    candidate = copy_plan(current)
                                    if dst_i == src_i:
                                        candidate[src_i] = new_dst
                                    else:
                                        candidate[src_i] = remainder
                                        candidate[dst_i] = new_dst
                                    candidate = [t for t in candidate if t]
                                    current, current_cost, improved = candidate, cand_cost, True
                                    if first_improvement:
                                        break
                            if improved and first_improvement:
                                break
                        if improved and first_improvement:
                            break
                    if improved and first_improvement:
                        break
                if improved and first_improvement:
                    break
            if improved and first_improvement:
                break
    return current

def top_up_plan(plan, inst, max_insert_size=3):
    CAP, BUDGET, MASS = inst["CAP"], inst["BUDGET"], inst["MASS"]
    current = copy_plan(plan)
    while True:
        collected = {u for trip in current for u in trip}
        remaining = [u for u in inst["SUPPLIES"] if u not in collected]
        if not remaining:
            break
        current_value = aco_plan_value(current, inst)
        current_cost = aco_plan_cost(current, inst)
        best_candidate, best_score = None, None
        max_size = min(max_insert_size, len(remaining))
        for size in range(1, max_size + 1):
            for combo in itertools.combinations(remaining, size):
                combo_mass = sum(MASS[u] for u in combo)
                if combo_mass > CAP + 1e-9:
                    continue
                for i in range(len(current)):
                    old_load = sum(MASS[u] for u in current[i])
                    if old_load + combo_mass > CAP + 1e-9:
                        continue
                    candidate = copy_plan(current)
                    candidate[i] = inst["best_order"](candidate[i] + list(combo))
                    cc = aco_plan_cost(candidate, inst)
                    if cc > BUDGET + 1e-9:
                        continue
                    cv = aco_plan_value(candidate, inst)
                    score = (cv - current_value, -(cc - current_cost))
                    if best_score is None or score > best_score:
                        best_score, best_candidate = score, candidate
                new_trip = inst["best_order"](list(combo))
                candidate = copy_plan(current)
                candidate.append(new_trip)
                cc = aco_plan_cost(candidate, inst)
                if cc > BUDGET + 1e-9:
                    continue
                cv = aco_plan_value(candidate, inst)
                score = (cv - current_value, -(cc - current_cost))
                if best_score is None or score > best_score:
                    best_score, best_candidate = score, candidate
        if best_candidate is None or best_score[0] <= 0:
            break
        current = best_candidate
    return current

def swap_search(plan, inst, max_remove=2, max_add=2, pool_size=14):
    MASS, VALUE = inst["MASS"], inst["VALUE"]
    current = copy_plan(plan)
    while True:
        collected = {u for trip in current for u in trip}
        remaining = [u for u in inst["SUPPLIES"] if u not in collected]
        if not remaining:
            break
        base_value = aco_plan_value(current, inst)
        base_cost = aco_plan_cost(current, inst)
        slack = max(inst["BUDGET"] - base_cost, 0.0)
        tight = slack / max(inst["BUDGET"], 1e-9) < 0.1
        best_candidate, best_score = None, None
        current_supplies = sorted([u for trip in current for u in trip],
                                   key=lambda u: VALUE[u] / max(MASS[u], 1e-9))[:pool_size]
        remaining_pool = sorted(remaining, key=lambda u: VALUE[u] / max(MASS[u], 1e-9),
                                 reverse=True)[:pool_size]
        max_remove_now = min(max_remove, len(current_supplies))
        max_add_now = min(max_add, len(remaining_pool))
        for remove_n in range(1, max_remove_now + 1):
            for add_n in range(1, max_add_now + 1):
                for removed in itertools.combinations(current_supplies, remove_n):
                    for added in itertools.combinations(remaining_pool, add_n):
                        candidate = [[u for u in trip if u not in removed] for trip in current]
                        candidate = [t for t in candidate if t]
                        for assignment_mode in ("existing", "new"):
                            trial = copy_plan(candidate)
                            valid = True
                            if assignment_mode == "existing":
                                for u in added:
                                    possible = []
                                    for i, trip in enumerate(trial):
                                        load = sum(MASS[x] for x in trip)
                                        if load + MASS[u] <= inst["CAP"] + 1e-9:
                                            possible.append((load, i))
                                    if not possible:
                                        valid = False
                                        break
                                    _, idx = min(possible)
                                    trial[idx].append(u)
                            else:
                                combo_mass = sum(MASS[u] for u in added)
                                if combo_mass > inst["CAP"] + 1e-9:
                                    valid = False
                                else:
                                    trial.append(list(added))
                            if not valid:
                                continue
                            trial = [inst["best_order"](t) for t in trial if t]
                            candidate_cost = aco_plan_cost(trial, inst)
                            if candidate_cost > inst["BUDGET"] + 1e-9:
                                continue
                            candidate_value = aco_plan_value(trial, inst)
                            gain = candidate_value - base_value
                            cost_increase = candidate_cost - base_cost
                            if gain <= 0:
                                continue
                            score = ((gain / max(cost_increase, 1e-9), gain, -cost_increase)
                                     if tight else (gain, -cost_increase))
                            if best_score is None or score > best_score:
                                best_score, best_candidate = score, trial
        if best_candidate is None or best_score is None or best_score[0] <= 0:
            break
        current = best_candidate
    return current

def run_iterative_aco(inst, actor, critic, initial_best, n_ants=32, n_iterations=20,
                       evaporation=0.15, elitist_weight=1.5, dist_scale=None, mode="sample",
                       seed=0, use_or_opt=True, use_topup=True):
    if dist_scale is None:
        dist_scale = inst["EXIT_LEG"]
    pheromone = initialise_pheromone(inst)
    global_best = copy_plan(initial_best)
    gv = aco_plan_value(global_best, inst)
    gc = aco_plan_cost(global_best, inst)
    for _iteration in range(n_iterations):
        iter_plans = []
        for _ant in range(n_ants):
            raw_plan, _traj = trainmod.construct_plan(inst, actor, critic, pheromone, dist_scale, mode=mode)
            valid, _ = inst_validate(inst, raw_plan)
            p = raw_plan if valid else []
            if p:
                if use_or_opt:
                    p = or_opt_plan(p, inst, max_segment=3)
                if use_topup:
                    p = top_up_plan(p, inst, max_insert_size=2)
            v = aco_plan_value(p, inst)
            iter_plans.append((copy_plan(p), v))
            c = aco_plan_cost(p, inst)
            if v > gv or (v == gv and c < gc - 1e-9):
                global_best, gv, gc = copy_plan(p), v, c
        iter_plans.append((copy_plan(global_best), gv))
        update_pheromone_elite(pheromone, iter_plans, inst["SHAFT"], evaporation,
                                global_best, gv, elitist_weight, inst=inst)
    return global_best

def hybrid_solve(inst, actor, critic, seed, ants=32, iterations=20, evaporation=0.15,
                  elitist_weight=1.5, mode="sample"):
    master_rng = random.Random(seed)
    best = []
    configs = [
        {"ants": ants, "iterations": iterations, "evaporation": evaporation, "elitist_weight": elitist_weight},
        {"ants": max(ants, 36), "iterations": max(iterations, 24), "evaporation": 0.22, "elitist_weight": 1.15},
        {"ants": max(ants, 36), "iterations": max(iterations, 36), "evaporation": 0.12, "elitist_weight": 1.75},
    ]
    for cfg in configs:
        run_seed = master_rng.randrange(1_000_000_000)
        candidate = run_iterative_aco(inst, actor, critic, initial_best=best, n_ants=cfg["ants"],
                                       n_iterations=cfg["iterations"], evaporation=cfg["evaporation"],
                                       elitist_weight=cfg["elitist_weight"], mode=mode, seed=run_seed)
        if better_plan(candidate, best, inst):
            best = copy_plan(candidate)
    topped_up = top_up_plan(best, inst, max_insert_size=3)
    if better_plan(topped_up, best, inst):
        best = copy_plan(topped_up)
    swapped = swap_search(best, inst, max_remove=2, max_add=2)
    if better_plan(swapped, best, inst):
        best = copy_plan(swapped)
    topped_up_again = top_up_plan(best, inst, max_insert_size=3)
    if better_plan(topped_up_again, best, inst):
        best = copy_plan(topped_up_again)
    return best

def exact_solve_subset(pool, budget, trip_cost, MASS, VALUE, CAP, EXIT_LEG):
    n = len(pool)
    full = 1 << n
    INF = float("inf")

    trip_cost_of_mask = [INF] * full
    for size in range(1, min(CAP, n) + 1):
        for combo in itertools.combinations(range(n), size):
            mass = sum(MASS[pool[i]] for i in combo)
            if mass > CAP + 1e-9:
                continue
            units = [pool[i] for i in combo]
            best = (min(trip_cost(list(p)) for p in itertools.permutations(units))
                    if len(units) > 1 else trip_cost(units))
            mask = 0
            for i in combo:
                mask |= (1 << i)
            trip_cost_of_mask[mask] = best

    best_cost = [INF] * full
    best_cost[0] = 0.0
    for mask in range(1, full):
        sub = mask
        c = INF
        while sub > 0:
            tc = trip_cost_of_mask[sub]
            if tc < INF:
                rest = mask ^ sub
                bc = best_cost[rest]
                if bc < INF and tc + bc < c:
                    c = tc + bc
            sub = (sub - 1) & mask
        best_cost[mask] = c

    value_of_mask = [0] * full
    for mask in range(1, full):
        low = mask & (-mask)
        i = low.bit_length() - 1
        value_of_mask[mask] = value_of_mask[mask ^ low] + VALUE[pool[i]]

    cap_energy = budget - EXIT_LEG
    if cap_energy < -1e-9:
        return 0

    best_value = 0
    for mask in range(full):
        if best_cost[mask] <= cap_energy + 1e-9 and value_of_mask[mask] > best_value:
            best_value = value_of_mask[mask]
    return best_value

def draw_seeds_for_bucket(n_wings):
    rng = random.Random(f"{MAIN_SEED}-{n_wings}")
    main_inst_wings = trainmod.make_instance(MAIN_SEED)["n_wings"]
    seeds = [MAIN_SEED] if n_wings == main_inst_wings else []
    while len(seeds) < N_SEEDS:
        s = trainmod.seed_for_wings(rng, n_wings)
        if s not in seeds:
            seeds.append(s)
    return seeds

def load_actor_critic(n_wings):
    actor = trainmod.Actor()
    critic = trainmod.Critic()
    ck = torch.load(f"{CHECKPOINT_DIR}/policy_wings{n_wings}.pt", map_location="cpu", weights_only=True)
    actor.load_state_dict(ck["actor"])
    critic.load_state_dict(ck["critic"])
    actor.eval()
    critic.eval()
    return actor, critic

def exemplar_a_nearest_fill(inst, pool, budget):
    SHAFT, MASS, CAP = inst["SHAFT"], inst["MASS"], inst["CAP"]
    dist, trip_cost, best_order, EXIT_LEG = inst["dist"], inst["trip_cost"], inst["best_order"], inst["EXIT_LEG"]
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
        trip = best_order(trip)
        c = trip_cost(trip)
        if spent + c + EXIT_LEG > budget:
            break
        spent += c
        plan.append(trip)
        for u in trip:
            remaining.remove(u)
    return plan

def run_bucket(n_wings):
    seeds = draw_seeds_for_bucket(n_wings)
    actor, critic = load_actor_critic(n_wings)

    rows = []
    for seed in seeds:
        inst = add_caching(trainmod.make_instance(seed))

        t0 = time.time()
        full_plan = hybrid_solve(inst, actor, critic, seed=seed, ants=ANTS, iterations=ITERATIONS)
        elapsed = time.time() - t0
        full_value = aco_plan_value(full_plan, inst)
        full_cost = aco_plan_cost(full_plan, inst)

        pool15 = sorted(inst["SUPPLIES"], key=lambda u: inst["VALUE"][u] / inst["MASS"][u], reverse=True)[:15]
        full_plan_15 = exemplar_a_nearest_fill(inst, pool15, budget=float("inf"))
        full_cost_15 = inst["plan_cost"](full_plan_15)
        budget15 = round(full_cost_15 * 0.60)
        inst15 = dict(inst)
        inst15["SUPPLIES"] = pool15
        inst15["BUDGET"] = budget15

        hybrid_plan_15 = hybrid_solve(inst15, actor, critic, seed=seed, ants=ANTS, iterations=ITERATIONS)
        hybrid_value_15 = aco_plan_value(hybrid_plan_15, inst15)

        exact_value_15 = exact_solve_subset(
            pool15,
            budget15,
            inst["trip_cost"],
            inst["MASS"],
            inst["VALUE"],
            inst["CAP"],
            inst["EXIT_LEG"],
        )

        gap = (100.0 * (exact_value_15 - hybrid_value_15) / exact_value_15
               if exact_value_15 > 0 else 0.0)

        rows.append((seed, full_value, round(full_cost, 1), round(elapsed, 4),
                     exact_value_15, hybrid_value_15, round(gap, 1)))
        print(f"[wings={n_wings}] seed={seed}  full_value={full_value}  full_cost={full_cost:.1f}  "
              f"time={elapsed:.2f}s  k15_exact={exact_value_15}  "
              f"k15_hybrid={hybrid_value_15}  gap={gap:.1f}%")

    return rows

def main():
    results = {}
    for n_wings in WING_BUCKETS:
        print(f"\n=== n_wings={n_wings} (policy_wings{n_wings}.pt) ===")
        results[n_wings] = run_bucket(n_wings)

    for n_wings in WING_BUCKETS:
        print(f"\n_rows_wings{n_wings} = [")
        for r in results[n_wings]:
            print(f"    {r},")
        print("]")

if __name__ == "__main__":
    main()