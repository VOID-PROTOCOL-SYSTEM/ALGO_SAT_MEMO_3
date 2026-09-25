import argparse
import csv
import itertools
import multiprocessing as mp
import os
import random
import time

import networkx as nx
import torch

import train_hybrid_aco_policies as trainmod

def fac_generator_seed_based(seed):
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
        wing_names = ['Alpha', 'Beta', 'Gamma', 'Delta'][:n_wings]

        wings = [_build_wing(WING_COLS, WING_ROWS, random.Random(s * 31 + w * 7919))
                 for w in range(n_wings)]

        junctions = []
        for w in range(n_wings - 1):
            jr = random.Random(s * 17 + w * 5003)
            rows_avail = list(range(2, WING_ROWS - 2))
            jr.shuffle(rows_avail)
            r1, r2 = sorted(rows_avail[:2])
            junctions.append(((w, WING_COLS - 1, r1), (w + 1, 0, r1)))
            junctions.append(((w, WING_COLS - 1, r2), (w + 1, 0, r2)))

        shaft = (0, 0, 0)  # entry == extraction point
        exit_a = (n_wings - 1, WING_COLS - 1, WING_ROWS - 1)
        exit_b = (n_wings - 1, WING_COLS - 1, 0)

        srng = random.Random(s * 13 + 42)
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

        # Amendment A2 corridor cost models
        for w, wg in enumerate(wings):
            if w == 1:
                for (c1, r1), (c2, r2) in list(wg.edges()):
                    wg[c1, r1][c2, r2]['weight'] = 1 + max(c1, c2) // 3
            elif w >= 2:
                cr = random.Random(s * 41 + w * 3331)
                for (c1, r1), (c2, r2) in list(wg.edges()):
                    wg[c1, r1][c2, r2]['weight'] = cr.randint(1, 5)

        # flatten to one graph
        G = nx.Graph()
        for w, wg in enumerate(wings):
            for (a, b, d) in wg.edges(data=True):
                G.add_edge((w,) + a, (w,) + b, weight=d['weight'])
        for a, b in junctions:
            G.add_edge(a, b, weight=1)

        prng = random.Random(s * 977 + 13)
        masses = [prng.choice([1, 2, 3]) for _ in supplies]
        values = [prng.choice([1, 2, 3, 4, 5]) for _ in supplies]

        return dict(G=G, wings=wings, n_wings=n_wings, wing_names=wing_names,
                    junctions=junctions, shaft=shaft, exits=[exit_a, exit_b],
                    supplies=supplies, masses=masses, values=values,
                    capacity=CAPACITY, wing_cols=WING_COLS, wing_rows=WING_ROWS)

    fac = get_facility(seed)

    # ---- metric closure: cheapest corridor path between every pair of key nodes
    _key = [fac['shaft']] + fac['supplies'] + fac['exits']
    DIST = {n: nx.single_source_dijkstra_path_length(fac['G'], n, weight='weight')
            for n in _key}
    PATH = {n: nx.single_source_dijkstra_path(fac['G'], n, weight='weight')
            for n in _key}

    def dist(u, v):
        return DIST[u][v]

    def corridor_path(u, v):
        return PATH[u][v]

    SHAFT = fac['shaft']
    EXITS = fac['exits']
    SUPPLIES = fac['supplies']
    MASS = {u: m for u, m in zip(fac['supplies'], fac['masses'])}
    VALUE = {u: v for u, v in zip(fac['supplies'], fac['values'])}
    CAP = fac['capacity']
    EXIT_LEG = min(dist(SHAFT, e) for e in EXITS)

    return dict(fac=fac, shaft=SHAFT, mass=MASS, value=VALUE, exits=EXITS, supplies=SUPPLIES,
                cap=CAP, exit_leg=EXIT_LEG, dist=DIST, path=PATH)

def ext_aco_plan_value(plan, inst):
    return inst["plan_value"](plan)

def ext_aco_plan_cost(plan, inst):
    return inst["plan_cost"](plan)

def ext_copy_plan(plan):
    return [list(t) for t in plan]

def ext_better_plan(a, b, inst):
    if b is None:
        return True
    av = ext_aco_plan_value(a, inst)
    bv = ext_aco_plan_value(b, inst)
    if av > bv:
        return True
    if av == bv:
        return ext_aco_plan_cost(a, inst) < ext_aco_plan_cost(b, inst) - 1e-9
    return False

def ext_add_caching(inst):
    raw_best_order = inst["best_order"]
    raw_trip_cost = inst["trip_cost"]
    best_order_cache = {}
    trip_cost_cache = {}

    def ext_cached_best_order(trip):
        key = frozenset(trip)
        cached = best_order_cache.get(key)
        if cached is None:
            cached = tuple(raw_best_order(list(trip)))
            best_order_cache[key] = cached
        return list(cached)

    def ext_cached_trip_cost(trip):
        key = tuple(trip)
        cached = trip_cost_cache.get(key)
        if cached is None:
            cached = raw_trip_cost(list(trip))
            trip_cost_cache[key] = cached
        return cached

    inst = dict(inst)
    inst["best_order"] = ext_cached_best_order
    inst["trip_cost"] = ext_cached_trip_cost
    return inst

def ext_load_policy_for_instance(inst, checkpoint_dir):
    n_wings = inst["n_wings"]
    path = os.path.join(checkpoint_dir, f"policy_wings{n_wings}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find policy checkpoint:\n{path}")

    actor = trainmod.Actor(7)
    critic = trainmod.Critic(4)

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    if "actor" in checkpoint:
        actor.load_state_dict(checkpoint["actor"])
    else:
        actor.load_state_dict(checkpoint)

    if "critic" in checkpoint:
        critic.load_state_dict(checkpoint["critic"])

    actor.eval()
    critic.eval()
    return actor, critic, path

def ext_inst_validate(inst, plan):
    CAP = inst["CAP"]
    BUDGET = inst["BUDGET"]
    MASS = inst["MASS"]
    seen = set()

    for trip in plan:
        load = 0.0
        for u in trip:
            if u not in inst["SUPPLIES"]:
                return False, f"Unknown supply: {u}"
            if u in seen:
                return False, f"Duplicate supply: {u}"
            seen.add(u)
            load += MASS[u]
        if load > CAP + 1e-9:
            return False, f"Capacity exceeded: {load:.3f} > {CAP:.3f}"

    cost = ext_aco_plan_cost(plan, inst)
    if cost > BUDGET + 1e-9:
        return False, f"Budget exceeded: {cost:.3f} > {BUDGET:.3f}"

    return True, "OK"

def ext_initialise_pheromone(inst):
    pheromone = {}
    supplies = inst["SUPPLIES"]
    shaft = inst["SHAFT"]

    for u in supplies:
        pheromone[shaft, u] = 1.0
        pheromone[u, "STOP"] = 1.0

    for u in supplies:
        for v in supplies:
            if u != v:
                pheromone[u, v] = 1.0
    return pheromone

def ext_update_pheromone_elite(pheromone, ranked_plans, shaft, evaporation, elite_plan, elite_value,
                                elitist_weight, min_pheromone=0.2, max_pheromone=12.0, inst=None):
    for key in pheromone:
        pheromone[key] *= 1.0 - evaporation

    if inst is not None:
        ranked = sorted(ranked_plans, key=lambda x: (x[1], -ext_aco_plan_cost(x[0], inst)), reverse=True)
    else:
        ranked = sorted(ranked_plans, key=lambda x: x[1], reverse=True)

    if ranked:
        elite_count = max(2, min(8, len(ranked) // 4))
        elite_count = min(elite_count, len(ranked))
        elite_group = ranked[:elite_count]

        for rank, (plan, value) in enumerate(elite_group):
            if value <= 0:
                continue
            rank_weight = (elite_count - rank) / elite_count
            deposit = 0.75 * rank_weight * value / 100.0

            for trip in plan:
                here = shaft
                for u in trip:
                    key = (here, u)
                    if key not in pheromone:
                        pheromone[key] = 1.0
                    pheromone[key] += deposit
                    here = u
                key = (here, "STOP")
                if key not in pheromone:
                    pheromone[key] = 1.0
                pheromone[key] += deposit

    if elite_plan and elite_value > 0:
        elite_deposit = elitist_weight * elite_value / 100.0
        for trip in elite_plan:
            here = shaft
            for u in trip:
                key = (here, u)
                if key not in pheromone:
                    pheromone[key] = 1.0
                pheromone[key] += elite_deposit
                here = u
            key = (here, "STOP")
            if key not in pheromone:
                pheromone[key] = 1.0
            pheromone[key] += elite_deposit

    for key in pheromone:
        pheromone[key] = min(max(pheromone[key], min_pheromone), max_pheromone)

def ext_or_opt_plan(plan, inst, max_segment=3, first_improvement=True):
    CAP = inst["CAP"]
    BUDGET = inst["BUDGET"]
    MASS = inst["MASS"]
    trip_cost = inst["trip_cost"]
    current = ext_copy_plan(plan)

    improved = True
    while improved:
        improved = False
        current_cost = ext_aco_plan_cost(current, inst)

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
                            dst_base = remainder
                            dst_base_cost = remainder_cost
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
                                    candidate = ext_copy_plan(current)
                                    if dst_i == src_i:
                                        candidate[src_i] = new_dst
                                    else:
                                        candidate[src_i] = remainder
                                        candidate[dst_i] = new_dst
                                    candidate = [t for t in candidate if t]

                                    current = candidate
                                    current_cost = cand_cost
                                    improved = True
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

def ext_run_iterative_aco(inst, actor, critic, initial_best, n_ants=32, n_iterations=20,
                           evaporation=0.15, elitist_weight=1.5, dist_scale=None, mode="sample",
                           seed=0, use_or_opt=True, use_topup=True,
                           snapshot_cb=None, time_log=None, start_time=None):
    if dist_scale is None:
        dist_scale = inst["EXIT_LEG"]

    rng = random.Random(seed)
    pheromone = ext_initialise_pheromone(inst)

    global_best = ext_copy_plan(initial_best)
    global_best_value = ext_aco_plan_value(global_best, inst)
    global_best_cost = ext_aco_plan_cost(global_best, inst)

    for iteration in range(1, n_iterations + 1):
        iter_plans = []
        raw_values = []

        for ant in range(n_ants):
            raw_plan, trajectory = trainmod.construct_plan(inst, actor, critic, pheromone, dist_scale, mode=mode)

            valid, _ = ext_inst_validate(inst, raw_plan)
            plan_for_ant = raw_plan if valid else []

            if plan_for_ant:
                if use_or_opt:
                    plan_for_ant = ext_or_opt_plan(plan_for_ant, inst, max_segment=3)
                if use_topup:
                    plan_for_ant = ext_top_up_plan(plan_for_ant, inst, max_insert_size=2)

            value = ext_aco_plan_value(plan_for_ant, inst)
            raw_values.append(value)
            iter_plans.append((ext_copy_plan(plan_for_ant), value))

            cost = ext_aco_plan_cost(plan_for_ant, inst)
            if value > global_best_value or (value == global_best_value and cost < global_best_cost - 1e-9):
                global_best = ext_copy_plan(plan_for_ant)
                global_best_value = value
                global_best_cost = cost

        iter_plans.append((ext_copy_plan(global_best), global_best_value))
        ext_update_pheromone_elite(pheromone, iter_plans, inst["SHAFT"], evaporation, global_best,
                                    global_best_value, elitist_weight, inst=inst)

        if snapshot_cb is not None:
            snapshot_cb(pheromone, global_best_value)
        if time_log is not None and start_time is not None:
            time_log.append((round(time.time() - start_time, 3), global_best_value))

    return global_best


def ext_top_up_plan(plan, inst, max_insert_size=3, pool_size=None):
    CAP = inst["CAP"]
    BUDGET = inst["BUDGET"]
    MASS = inst["MASS"]
    VALUE = inst["VALUE"]
    current = ext_copy_plan(plan)

    while True:
        collected = {u for trip in current for u in trip}
        remaining = [u for u in inst["SUPPLIES"] if u not in collected]
        if not remaining:
            break

        if pool_size is not None:
            remaining = sorted(
                remaining,
                key=lambda u: VALUE[u] / max(MASS[u], 1e-9),
                reverse=True,
            )[:pool_size]

        current_value = ext_aco_plan_value(current, inst)
        current_cost = ext_aco_plan_cost(current, inst)
        best_candidate = None
        best_score = None
        max_size = min(max_insert_size, len(remaining))

        for size in range(1, max_size + 1):
            for combo in itertools.combinations(remaining, size):
                combo_mass = sum(MASS[u] for u in combo)
                if combo_mass > CAP + 1e-9:
                    continue

                for i in range(len(current)):
                    old_trip = current[i]
                    old_load = sum(MASS[u] for u in old_trip)
                    if old_load + combo_mass > CAP + 1e-9:
                        continue

                    candidate = ext_copy_plan(current)
                    candidate[i] = inst["best_order"](candidate[i] + list(combo))

                    candidate_cost = ext_aco_plan_cost(candidate, inst)
                    if candidate_cost > BUDGET + 1e-9:
                        continue

                    candidate_value = ext_aco_plan_value(candidate, inst)
                    gain = candidate_value - current_value
                    extra_cost = candidate_cost - current_cost
                    score = (gain, -extra_cost)

                    if best_score is None or score > best_score:
                        best_score = score
                        best_candidate = candidate

                new_trip = inst["best_order"](list(combo))
                candidate = ext_copy_plan(current)
                candidate.append(new_trip)

                candidate_cost = ext_aco_plan_cost(candidate, inst)
                if candidate_cost > BUDGET + 1e-9:
                    continue

                candidate_value = ext_aco_plan_value(candidate, inst)
                gain = candidate_value - current_value
                extra_cost = candidate_cost - current_cost
                score = (gain, -extra_cost)

                if best_score is None or score > best_score:
                    best_score = score
                    best_candidate = candidate

        if best_candidate is None:
            break
        if best_score[0] <= 0:
            break

        current = best_candidate
    return current


def ext_swap_search(plan, inst, max_remove=2, max_add=2, pool_size=14):
    MASS = inst["MASS"]
    VALUE = inst["VALUE"]
    current = ext_copy_plan(plan)

    while True:
        collected = {u for trip in current for u in trip}
        remaining = [u for u in inst["SUPPLIES"] if u not in collected]
        if not remaining:
            break

        base_value = ext_aco_plan_value(current, inst)
        base_cost = ext_aco_plan_cost(current, inst)

        slack = max(inst["BUDGET"] - base_cost, 0.0)
        slack_ratio = slack / max(inst["BUDGET"], 1e-9)
        tight = slack_ratio < 0.1

        best_candidate = None
        best_score = None

        current_supplies_all = [u for trip in current for u in trip]
        current_supplies = sorted(current_supplies_all, key=lambda u: VALUE[u] / max(MASS[u], 1e-9))[:pool_size]
        remaining_pool = sorted(remaining, key=lambda u: VALUE[u] / max(MASS[u], 1e-9), reverse=True)[:pool_size]
        max_remove_now = min(max_remove, len(current_supplies))
        max_add_now = min(max_add, len(remaining_pool))

        for remove_n in range(1, max_remove_now + 1):
            for add_n in range(1, max_add_now + 1):
                for removed in itertools.combinations(current_supplies, remove_n):
                    for added in itertools.combinations(remaining_pool, add_n):
                        candidate = [[u for u in trip if u not in removed] for trip in current]
                        candidate = [t for t in candidate if t]

                        for assignment_mode in ("existing", "new"):
                            trial = ext_copy_plan(candidate)
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

                            candidate_cost = ext_aco_plan_cost(trial, inst)
                            if candidate_cost > inst["BUDGET"] + 1e-9:
                                continue

                            candidate_value = ext_aco_plan_value(trial, inst)
                            gain = candidate_value - base_value
                            cost_increase = candidate_cost - base_cost

                            if gain <= 0:
                                continue

                            if tight:
                                score = (gain / max(cost_increase, 1e-9), gain, -cost_increase)
                            else:
                                score = (gain, -cost_increase)

                            if best_score is None or score > best_score:
                                best_score = score
                                best_candidate = trial

        if best_candidate is None or best_score is None or best_score[0] <= 0:
            break
        current = best_candidate
    return current


def ext_hybrid_solve(inst, actor, critic, seed, ants=32, iterations=20, evaporation=0.15,
                      elitist_weight=1.5, mode="sample", use_or_opt=True, use_topup=True,
                      use_final_topup=True, use_swap_search=True,
                      snapshot_cb=None, time_log=None):
    _start_time = time.time()
    torch.manual_seed(seed)
    master_rng = random.Random(seed)
    best = []
    best_source = "empty seed"

    configs = [
        {"ants": ants, "iterations": iterations, "evaporation": evaporation, "elitist_weight": elitist_weight},
        {"ants": max(ants, 36), "iterations": max(iterations, 24), "evaporation": 0.22, "elitist_weight": 1.15},
        {"ants": max(ants, 36), "iterations": max(iterations, 36), "evaporation": 0.12, "elitist_weight": 1.75},
    ]

    for restart, cfg in enumerate(configs):
        run_seed = master_rng.randrange(1000000000)
        candidate = ext_run_iterative_aco(inst, actor, critic, initial_best=best, n_ants=cfg["ants"],
            n_iterations=cfg["iterations"], evaporation=cfg["evaporation"], elitist_weight=cfg["elitist_weight"],
            mode=mode, seed=run_seed, use_or_opt=use_or_opt, use_topup=use_topup,
            snapshot_cb=snapshot_cb, time_log=time_log, start_time=_start_time)

        if ext_better_plan(candidate, best, inst):
            best = ext_copy_plan(candidate)
            best_source = f"learned ACO restart {restart + 1}"

    if use_final_topup:
        topped_up = ext_top_up_plan(best, inst, max_insert_size=3)
        if ext_better_plan(topped_up, best, inst):
            best = ext_copy_plan(topped_up)
            best_source = best_source + " + top-up"

    if use_swap_search:
        swapped = ext_swap_search(best, inst, max_remove=2, max_add=2)
        if ext_better_plan(swapped, best, inst):
            best = ext_copy_plan(swapped)
            best_source = best_source + " + swap search"

    if use_final_topup:
        topped_up_again = ext_top_up_plan(best, inst, max_insert_size=3)
        if ext_better_plan(topped_up_again, best, inst):
            best = ext_copy_plan(topped_up_again)
            best_source = best_source + " + top-up"

    return best, best_source


def make_trip_cost(inst):
    SHAFT, dist, MASS = inst["SHAFT"], inst["dist"], inst["MASS"]

    def _trip_cost(trip):
        here, total, load = SHAFT, 0.0, 0
        for u in trip:
            total += (1 + load) * dist(here, u)
            load += MASS[u]
            here = u
        total += (1 + load) * dist(here, SHAFT)
        return total
    return _trip_cost


def make_plan_cost(inst, trip_cost):
    EXIT_LEG = inst["EXIT_LEG"]

    def _plan_cost(plan):
        return sum(trip_cost(t) for t in plan) + EXIT_LEG
    return _plan_cost


def make_plan_value(inst):
    VALUE = inst["VALUE"]

    def _plan_value(plan):
        return sum(VALUE[u] for t in plan for u in t)
    return _plan_value


def make_best_order(trip_cost):
    def _best_order(units):
        if len(units) <= 1:
            return list(units)
        return list(min(itertools.permutations(units), key=trip_cost))
    return _best_order


def ext_exemplar_a_nearest_fill(pool, budget, inst):
    """A -- pack whatever is closest, ignore priority entirely."""
    SHAFT = inst["SHAFT"]
    MASS = inst["MASS"]
    CAP = inst["CAP"]
    dist = inst["dist"]
    best_order = inst["best_order"]
    trip_cost = inst["trip_cost"]
    EXIT_LEG = inst["EXIT_LEG"]

    remaining, plan, spent = list(pool), [], 0.0
    while remaining:
        trip, here, load = [], SHAFT, 0
        while True:
            fits = [u for u in remaining
                    if u not in trip and load + MASS[u] <= CAP]
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


def build_inst(fac_result, pool, budget):
    """fac_result = the dict returned by fac_generator_seed_based(seed)."""
    inst = dict(
        n_wings=fac_result["fac"]["n_wings"],
        SHAFT=fac_result["shaft"],
        EXITS=fac_result["exits"],
        SUPPLIES=pool,
        MASS=fac_result["mass"],
        VALUE=fac_result["value"],
        CAP=fac_result["cap"],
        BUDGET=budget,
        EXIT_LEG=fac_result["exit_leg"],
        dist=lambda u, v: fac_result["dist"][u][v]
    )
    trip_cost = make_trip_cost(inst)
    inst["trip_cost"] = trip_cost
    inst["plan_cost"] = make_plan_cost(inst, trip_cost)
    inst["plan_value"] = make_plan_value(inst)
    inst["best_order"] = make_best_order(trip_cost)
    return ext_add_caching(inst)

CSV_FIELDS = [
    "seed", "n_wings", "exemplar_cost", "budget", "cap", "n_supplies",
    "total_value", "plan_value", "plan_cost", "n_trips", "best_source",
    "solve_time_s", "status", "error",
]

_POLICY_CACHE = {}
_CHECKPOINT_DIR = None


def _init_worker(checkpoint_dir):
    global _POLICY_CACHE, _CHECKPOINT_DIR
    _POLICY_CACHE = {}
    _CHECKPOINT_DIR = checkpoint_dir
    torch.set_num_threads(1)


def _get_policy(inst):
    n_wings = inst["n_wings"]
    if n_wings not in _POLICY_CACHE:
        actor, critic, _ = ext_load_policy_for_instance(inst, _CHECKPOINT_DIR)
        _POLICY_CACHE[n_wings] = (actor, critic)
    return _POLICY_CACHE[n_wings]


def solve_one_seed(task):
    seed, budget_fraction, ants, iterations = task
    t0 = time.time()
    exemplar_cost = None
    budget = None
    try:
        fac_result = fac_generator_seed_based(seed)
        pool = fac_result["supplies"]

        scratch_inst = build_inst(fac_result, pool, float("inf"))
        exemplar_plan = ext_exemplar_a_nearest_fill(pool, float("inf"), scratch_inst)
        exemplar_cost = scratch_inst["plan_cost"](exemplar_plan)
        budget = budget_fraction * exemplar_cost

        inst = build_inst(fac_result, pool, budget)

        actor, critic = _get_policy(inst)

        best_plan, best_source = ext_hybrid_solve(
            inst, actor, critic, seed=seed, ants=ants, iterations=iterations,
        )

        plan_value = inst["plan_value"](best_plan)
        plan_cost = inst["plan_cost"](best_plan)
        total_value = sum(inst["VALUE"][u] for u in pool)

        return {
            "seed": seed,
            "n_wings": inst["n_wings"],
            "exemplar_cost": round(exemplar_cost, 4),
            "budget": round(budget, 4),
            "cap": inst["CAP"],
            "n_supplies": len(pool),
            "total_value": total_value,
            "plan_value": plan_value,
            "plan_cost": round(plan_cost, 4),
            "n_trips": len(best_plan),
            "best_source": best_source,
            "solve_time_s": round(time.time() - t0, 3),
            "status": "ok",
            "error": "",
        }
    except Exception as e:
        return {
            "seed": seed,
            "n_wings": "",
            "exemplar_cost": round(exemplar_cost, 4) if exemplar_cost is not None else "",
            "budget": round(budget, 4) if budget is not None else "",
            "cap": "",
            "n_supplies": "",
            "total_value": "",
            "plan_value": "",
            "plan_cost": "",
            "n_trips": "",
            "best_source": "",
            "solve_time_s": round(time.time() - t0, 3),
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
        }


def append_rows_to_csv(path, rows):
    file_exists = os.path.exists(path)
    write_header = (not file_exists) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def seed_task_stream(start_seed, num_seeds, budget_fraction, ants, iterations):
    if num_seeds is None:
        seeds = itertools.count(start_seed)
    else:
        seeds = range(start_seed, start_seed + num_seeds)
    for s in seeds:
        yield (s, budget_fraction, ants, iterations)


def main():
    parser = argparse.ArgumentParser(description="Mass fac testing runner (multiprocessing, 8 workers)")
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=5000,
                         help="Number of seeds to solve. Ignored if --continuous is set.")
    parser.add_argument("--continuous", action="store_true",
                         help="Run forever starting at --start-seed, until interrupted (Ctrl+C).")
    parser.add_argument("--budget-fraction", type=float, default=0.6,
                         help="Per-seed BUDGET is this fraction of the 'A -- nearest fill' exemplar "
                              "plan's cost (ext_exemplar_a_nearest_fill run with an unlimited budget).")
    parser.add_argument("--ants", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint-dir", type=str,
                         default=os.path.dirname(os.path.abspath(__file__)),
                         help="Directory containing policy_wings{2,3,4}.pt (defaults to this script's dir).")
    parser.add_argument("--out", type=str, default="fac_results.csv")
    parser.add_argument("--flush-every", type=int, default=16)
    args = parser.parse_args()

    num_seeds = None if args.continuous else args.num_seeds
    tasks = seed_task_stream(args.start_seed, num_seeds, args.budget_fraction, args.ants, args.iterations)

    buffer = []
    completed = 0
    start_time = time.time()
    total_label = "∞" if num_seeds is None else str(num_seeds)

    print(f"Starting: seeds from {args.start_seed}, count={total_label}, "
          f"workers={args.workers}, budget_fraction={args.budget_fraction}, out={args.out}", flush=True)

    with mp.Pool(processes=args.workers, initializer=_init_worker,
                 initargs=(args.checkpoint_dir,)) as pool:
        try:
            for result in pool.imap_unordered(solve_one_seed, tasks, chunksize=1):
                buffer.append(result)
                completed += 1

                if len(buffer) >= args.flush_every:
                    append_rows_to_csv(args.out, buffer)
                    buffer = []

                if completed % args.flush_every == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    print(f"[{completed}/{total_label}] seed={result['seed']} "
                          f"status={result['status']} value={result.get('plan_value', '')} "
                          f"rate={rate:.2f} seeds/s", flush=True)
        except KeyboardInterrupt:
            print("\nInterrupted — flushing remaining buffered rows...", flush=True)
        finally:
            if buffer:
                append_rows_to_csv(args.out, buffer)

    elapsed = time.time() - start_time
    print(f"Done. {completed} seeds processed in {elapsed:.1f}s "
          f"({completed / elapsed if elapsed else 0:.2f} seeds/s). Results in {args.out}")


if __name__ == "__main__":
    main()