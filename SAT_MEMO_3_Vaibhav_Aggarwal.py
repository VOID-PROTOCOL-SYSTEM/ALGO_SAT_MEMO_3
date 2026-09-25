import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="medium",
    app_title="Vaibhav Memo 3",
    css_file="cyberpunk-marimo.css",
)


@app.cell
def imports():
    import marimo as mo
    import random
    import math
    import copy
    import time
    import itertools
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    import matplotlib.animation as animation
    from matplotlib.colors import LinearSegmentedColormap
    import json
    import os
    import datetime

    SAVE_FILE_M02 = "Global_Files/responses_M02.json"   # Memo 02 responses (read-only reference)
    SAVE_FILE_M03 = "Global_Files/responses_M03.json"   # Memo 03 responses (written here)
    return (
        LinearSegmentedColormap,
        SAVE_FILE_M03,
        animation,
        datetime,
        itertools,
        json,
        mcolors,
        mo,
        nx,
        os,
        plt,
        random,
        time,
    )


@app.cell
def header(mo):
    mo.md("""
    # Operation Emberlight -- Memo 03 Workbook
    ## Load Sensitivity and Triage

    | Field | Value |
    |---|---|
    | **Name** | *Vaibhav Aggarwal* |
    | **Student number** | *24277364L* |
    | **Facility seed** | *23092008* |
    | **Teacher** | *Kodie Nielsen* |

    > **Field update -- three things have changed.**
    >
    > **1. Carried mass now costs energy.** A corridor of stability weight `w`
    > traversed while carrying total mass `L` costs `w x (1 + L)`.
    >
    > **2. CRUDY-1 cannot lift the whole payload.** It has a **mass capacity of
    > C = 5 units**. The entry shaft is now the **extraction point**: CRUDY-1
    > descends, collects up to C mass units, returns to the shaft, deposits its
    > load (carried mass resets to 0), and descends again. At the end it must
    > reach an exit under its own power.
    >
    > **3. The operation is energy-limited.** A total battery budget `B` covers
    > the whole operation, including the final run to an exit. `B` is about 60%
    > of what full extraction costs, so **you cannot take everything**. The
    > directive is to **maximise priority value delivered within `B`**.

    **This workbook is issued complete.** All six actions are below. There are
    two submission points; the whole problem is visible from the start.

    | Section | Criterion | Due |
    |---|---|---|
    | `[M3-0]` Orientation + exemplars | -- | Obs W8|
    | `[M3-1]` Improved data model & algorithm | **C8** | W11 |
    | `[M3-2]` Quality of improved solution | **C9** | W11 |
    | `[M3-3]` Time complexity of improved solution | **C5b** | W11|
    | `[M3-4]` Intractability & the case for a heuristic | C5b / C7 | W11 |
    | `[M3-5]` Comparing time complexities | **C7** | W11 |
    | `[M3-6]` Comparing coherence & fitness | **C10** | W11 |

    > **Sequencing advice.** Run `[M3-4]` early, even roughly. It tells you
    > whether an optimal answer is reachable at your facility size -- and the
    > answer may change the algorithm you submit at Observation A.
    """)
    return


@app.cell
def seed_cell(mo):
    seed_input = mo.ui.number(
        start=1, stop=999999999, step=1, value=23092008,
        label="Your facility seed (from your Memo 01 cover sheet)"
    )
    mo.vstack([
        mo.md("### Enter your seed, then press Tab to rebuild the facility."),
        seed_input
    ])
    return (seed_input,)


@app.cell
def generator(nx, random):
    """Multi-wing facility generator, extended for Memo 03.

    Identical wing/junction/weight construction to Memo 02, plus:
      - k = 30 supply units (Memo 02 used 5)
      - each unit carries a mass m in {1,2,3} and a priority value v in {1..5}
      - supply sites fall back from dead ends to degree-2 corridor nodes so that
        k = 30 is placeable on 2-wing seeds as well as 3- and 4-wing seeds.
    """
    WING_COLS, WING_ROWS = 10, 10
    N_SUPPLIES = 30
    CAPACITY = 5

    def _neighbours(cols, rows, c, r):
        for dc, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nc, nr = c + dc, r + dr
            if 0 <= nc < cols and 0 <= nr < rows:
                yield nc, nr

    def _build_wing(cols, rows, rng):
        """Single-wing maze as a spanning tree of the grid.

        This is EXACTLY the carve used in the Memo 01 and Memo 02 workbooks --
        recursive, shuffling all four neighbours. It must stay bit-for-bit
        identical, or a student's facility would change between memos and their
        Memo 02 analysis would no longer describe their own maze.
        """
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

        shaft = (0, 0, 0)                                    # entry == extraction point
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

    return (get_facility,)


@app.cell
def build_facility(get_facility, nx, seed_input):
    fac = get_facility(seed_input.value)

    # ---- metric closure: cheapest corridor path between every pair of key nodes
    _key = [fac['shaft']] + fac['supplies'] + fac['exits']
    DIST = {n: nx.single_source_dijkstra_path_length(fac['G'], n, weight='weight')
            for n in _key}
    PATH = {n: nx.single_source_dijkstra_path(fac['G'], n, weight='weight')
            for n in _key}

    def dist(u, v):
        """Cheapest corridor cost from u to v under the Memo 02 weights,
        carrying nothing. Multiply by (1 + L) to get the load-aware cost."""
        return DIST[u][v]

    def corridor_path(u, v):
        """The actual sector-by-sector route behind dist(u, v)."""
        return PATH[u][v]

    return corridor_path, dist, fac


@app.cell
def cost_helpers(dist, fac):
    """The load-aware cost model. Use these -- do not re-implement them."""

    SHAFT = fac['shaft']
    EXITS = fac['exits']
    SUPPLIES = fac['supplies']
    MASS = {u: m for u, m in zip(fac['supplies'], fac['masses'])}
    VALUE = {u: v for u, v in zip(fac['supplies'], fac['values'])}
    CAP = fac['capacity']
    EXIT_LEG = min(dist(SHAFT, e) for e in EXITS)

    def trip_cost(trip):
        """Energy for one shuttle: shaft -> units in the given order -> shaft.

        Mass accumulates as units are picked up, so each leg is charged at the
        mass carried *along that leg*.
        """
        here, total, load = SHAFT, 0.0, 0
        for u in trip:
            total += (1 + load) * dist(here, u)
            load += MASS[u]
            here = u
        total += (1 + load) * dist(here, SHAFT)
        return total

    def trip_mass(trip):
        return sum(MASS[u] for u in trip)

    def trip_value(trip):
        return sum(VALUE[u] for u in trip)

    def plan_cost(plan):
        """Total energy for a list of trips, including the final run to an exit."""
        return sum(trip_cost(t) for t in plan) + EXIT_LEG

    def plan_value(plan):
        return sum(trip_value(t) for t in plan)

    def validate_plan(plan, budget=None):
        """Return (ok, list_of_problems). Always run this before quoting a result."""
        problems = []
        seen = []
        for i, t in enumerate(plan, 1):
            if trip_mass(t) > CAP:
                problems.append(
                    f"Trip {i} carries mass {trip_mass(t)}, over capacity {CAP}.")
            for u in t:
                if u not in MASS:
                    problems.append(f"Trip {i} contains {u}, which is not a supply unit.")
                if u in seen:
                    problems.append(f"Unit {u} appears in more than one trip.")
                seen.append(u)
        if budget is not None:
            c = plan_cost(plan)
            if c > budget:
                problems.append(f"Plan costs {c:.0f}, over budget {budget:.0f}.")
        return (len(problems) == 0), problems

    return (
        CAP,
        EXITS,
        EXIT_LEG,
        MASS,
        SHAFT,
        SUPPLIES,
        VALUE,
        plan_cost,
        plan_value,
        trip_cost,
        trip_value,
        validate_plan,
    )


@app.cell
def exemplar_heuristics(
    CAP,
    EXIT_LEG,
    MASS,
    SHAFT,
    VALUE,
    dist,
    itertools,
    trip_cost,
):
    """The three reference heuristics from Appendix M3-A.

    All three respect capacity and budget. They differ only in how they choose
    what goes into each trip.
    """

    def best_order(units):
        """Cheapest collection order within a single trip (brute force; trips are small)."""
        if len(units) <= 1:
            return list(units)
        return list(min(itertools.permutations(units), key=trip_cost))

    def exemplar_a_nearest_fill(pool, budget):
        """A -- pack whatever is closest, ignore priority entirely."""
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

    def exemplar_b_high_value(pool, budget):
        """B -- sort everything by priority, fill trips from the top of the list."""
        remaining = sorted(pool, key=lambda u: (-VALUE[u], MASS[u]))
        plan, spent = [], 0.0
        while remaining:
            trip, load = [], 0
            for u in remaining:
                if load + MASS[u] <= CAP:
                    trip.append(u)
                    load += MASS[u]
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

    def exemplar_c_value_per_mass(pool, budget):
        """C -- sort by priority per unit of mass, then fill trips in that order.

        Reasons about the capacity constraint (mass) but ignores geography
        entirely, so a trip can be scattered across the whole facility.
        """
        remaining = sorted(pool, key=lambda u: -(VALUE[u] / MASS[u]))
        plan, spent = [], 0.0
        while remaining:
            trip, load = [], 0
            for u in remaining:
                if load + MASS[u] <= CAP:
                    trip.append(u)
                    load += MASS[u]
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

    return (
        best_order,
        exemplar_a_nearest_fill,
        exemplar_b_high_value,
        exemplar_c_value_per_mass,
    )


@app.cell
def budget_calc(SUPPLIES, exemplar_a_nearest_fill, plan_cost):
    """Battery budgets, auto-calibrated to this student's facility.

    BUDGET          -- the mission budget: 60% of what full extraction costs.
    BUDGET_RESERVE  -- a contingency scenario at 35%, used in [M3-4]. Which
                       approach performs best is NOT stable across the two, so
                       an algorithm must be evaluated under both.
    """
    _full_plan = exemplar_a_nearest_fill(SUPPLIES, budget=float('inf'))
    FULL_EXTRACTION_COST = plan_cost(_full_plan)
    BUDGET = round(FULL_EXTRACTION_COST * 0.60)
    BUDGET_RESERVE = round(FULL_EXTRACTION_COST * 0.35)
    return BUDGET, BUDGET_RESERVE, FULL_EXTRACTION_COST


@app.cell
def facility_summary(
    BUDGET,
    BUDGET_RESERVE,
    CAP,
    EXIT_LEG,
    FULL_EXTRACTION_COST,
    SUPPLIES,
    fac,
    mo,
    seed_input,
):
    _V = fac['G'].number_of_nodes()
    _E = fac['G'].number_of_edges()

    _mid = (len(fac['supplies']) + 1) // 2

    _left_rows = "\n".join(
        f"| S{i+1:02d} | {u[0]} ({fac['wing_names'][u[0]]}) | ({u[1]}, {u[2]}) | {m} | {v} |"
        for i, (u, m, v) in enumerate(
            zip(
                fac['supplies'][:_mid],
                fac['masses'][:_mid],
                fac['values'][:_mid]
            )
        )
    )

    _right_rows = "\n".join(
        f"| S{i+1:02d} | {u[0]} ({fac['wing_names'][u[0]]}) | ({u[1]}, {u[2]}) | {m} | {v} |"
        for i, (u, m, v) in enumerate(
            zip(
                fac['supplies'][_mid:],
                fac['masses'][_mid:],
                fac['values'][_mid:]
            ),
            start=_mid
        )
    )

    mo.vstack([
        mo.md(f"""
    ## Your Facility -- Seed {int(seed_input.value)}

    | Quantity | Value |
    |---|---|
    | Sectors `V` | **{_V}** |
    | Corridors `E` | **{_E}** |
    | Wings | {fac['n_wings']} ({', '.join(fac['wing_names'])}) |
    | Supply units `k` | **{len(SUPPLIES)}** |
    | Total mass | {sum(fac['masses'])} |
    | Total priority value available | **{sum(fac['values'])}** |
    | CRUDY-1 mass capacity `C` | **{CAP}** |
    | Cost to extract everything | {FULL_EXTRACTION_COST:,.0f} |
    | **Battery budget `B`** | **{BUDGET:,}** |
    | Contingency budget `B_reserve` (used in [M3-4]) | {BUDGET_RESERVE:,} |
    | Final shaft-to-exit run | {EXIT_LEG:,.0f} (paid out of `B`) |

    > `B` is 60% of the cost of extracting everything. **Roughly a third of the
    > payload must be abandoned** -- deciding which third is part of the problem.
    >
    > `B_reserve` is a harsher 35% scenario. You design against `B`, but [M3-4]
    > asks you to evaluate under both. **Which approach performs best is not the
    > same at the two budgets** -- that is the point of testing twice.

    ### Supply Manifest
    """),
        mo.hstack([
            mo.md(
                f"""\
    | Unit | Wing | Position | Mass | Priority |
    |---|---|---|---|---|
    {_left_rows}
    """
            ),
            mo.md(
                f"""\
    | Unit | Wing | Position | Mass | Priority |
    |---|---|---|---|---|
    {_right_rows}
    """
            )
        ])
    ])
    return


@app.cell
def draw_setup(
    LinearSegmentedColormap,
    MASS,
    VALUE,
    corridor_path,
    fac,
    mcolors,
    plt,
):
    """Facility rendering -- same visual language as the Memo 01 / Memo 02 workbooks.

    Corridors are drawn as thick lines coloured by their stability weight, walls
    are drawn wherever no corridor exists, and each wing sits in its own bounded
    grid. Memo 03 adds: supply markers sized by mass, and colour-coded shuttle
    trips drawn along the actual corridor route.
    """
    COL_BG       = '#F5F7FA'
    COL_GRID     = '#C8D0DC'
    COL_WALL     = '#44546A'
    COL_ENTRY    = '#0B6E6B'
    COL_EXIT     = '#7A1E2C'
    COL_SUPPLY   = '#4AA8A0'
    COL_JUNCTION = '#7A1E2C'
    COL_DROPPED  = '#AEB6C2'
    _GAP = 3

    # Qualitative palette, deliberately kept clear of the teal-amber-maroon
    # weight ramp so trips never read as corridor costs.
    TRIP_COLOURS = ['#6D28D9', '#1E40AF', '#DB2777', '#059669', '#EA580C',
                    '#0891B2', '#9333EA', '#65A30D', '#E11D48', '#2563EB',
                    '#C026D3', '#0D9488', '#F59E0B', '#4F46E5', '#BE123C']

    _WEIGHT_CMAP = LinearSegmentedColormap.from_list(
        'emberweight', ['#B8E0DE', '#F4C97A', '#7A1E2C'], N=256
    )

    def _cost_color(weight, min_w=1, max_w=5):
        norm = (weight - min_w) / max(max_w - min_w, 1)
        return _WEIGHT_CMAP(norm)

    def _mass_size(m):
        """Star size encodes how expensive a unit is to carry."""
        return {1: 9, 2: 12, 3: 15}.get(m, 11)

    def draw_facility(plan=None, abandoned=None, show_labels=True,
                      only_trips=None, title="Weighted Multi-Wing Facility"):
        """Render the facility.

        plan        -- list of trips; each is drawn in its own colour along the
                       real corridor route. When a plan is shown the weighted
                       corridors are muted so the routes stay readable.
        only_trips  -- optional list of trip indices to draw (1-based). Use this
                       when a plan has many trips and the map gets crowded.
        """
        wc, wr = fac['wing_cols'], fac['wing_rows']
        nw = fac['n_wings']
        total_w = nw * wc + (nw - 1) * _GAP
        showing_plan = bool(plan)
        # corridors recede when routes are on top of them
        _corr_alpha = 0.30 if showing_plan else 1.0
        _corr_lw = 3.0 if showing_plan else 4.5

        fig_w = max(12, total_w * 0.62)
        fig_h = max(6, wr * 0.62 + 2.0)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.set_facecolor(COL_BG)
        fig.patch.set_facecolor(COL_BG)

        def xoff(w):
            return w * (wc + _GAP)

        # ---- wings: grid, corridors coloured by weight, walls, boundary ----
        for w, wing in enumerate(fac['wings']):
            ox = xoff(w)
            for c in range(wc + 1):
                ax.plot([ox + c, ox + c], [0, wr], color=COL_GRID, lw=0.3, zorder=1)
            for r in range(wr + 1):
                ax.plot([ox, ox + wc], [r, r], color=COL_GRID, lw=0.3, zorder=1)

            for (c1, r1), (c2, r2), data in wing.edges(data=True):
                ax.plot([ox + c1 + 0.5, ox + c2 + 0.5], [r1 + 0.5, r2 + 0.5],
                        color=_cost_color(data.get('weight', 1)), lw=_corr_lw,
                        alpha=_corr_alpha, solid_capstyle='round', zorder=2)

            for c in range(wc):
                for r in range(wr):
                    if c + 1 < wc and not wing.has_edge((c, r), (c + 1, r)):
                        ax.plot([ox + c + 1, ox + c + 1], [r, r + 1],
                                color=COL_WALL, lw=1.4, zorder=3)
                    if r + 1 < wr and not wing.has_edge((c, r), (c, r + 1)):
                        ax.plot([ox + c, ox + c + 1], [r + 1, r + 1],
                                color=COL_WALL, lw=1.4, zorder=3)

            ax.add_patch(plt.Rectangle((ox, 0), wc, wr, fill=False,
                                       edgecolor=COL_WALL, lw=2.2, zorder=4))
            model_names = ['Uniform', 'Depth-based', 'Randomised', 'Randomised']
            model_lbl = model_names[w] if w < len(model_names) else 'Randomised'
            ax.text(ox + wc / 2, wr + 0.55, f"Wing {fac['wing_names'][w]}",
                    ha='center', va='bottom', fontsize=9, fontweight='bold',
                    color='#0B1F3B', zorder=8)
            ax.text(ox + wc / 2, wr + 0.15, f"({model_lbl})", ha='center',
                    va='bottom', fontsize=7, color='#44546A', zorder=8)

        # ---- inter-wing junctions ----
        for (w1, c1, r1), (w2, c2, r2) in fac['junctions']:
            x1, y1 = xoff(w1) + c1 + 0.5, r1 + 0.5
            x2, y2 = xoff(w2) + c2 + 0.5, r2 + 0.5
            ax.plot([x1, x2], [y1, y2], color=COL_JUNCTION, lw=2.0,
                    linestyle='--', alpha=0.8, zorder=5)
            ax.plot(x1, y1, 'o', ms=8, color=COL_JUNCTION, zorder=6)
            ax.plot(x2, y2, 'o', ms=8, color=COL_JUNCTION, zorder=6)

        # ---- shuttle trips, drawn along the real corridor route ----
        trip_of = {}
        drawn_trips = 0
        if plan:
            _wanted = set(only_trips) if only_trips else None
            for i, trip in enumerate(plan):
                if _wanted is not None and (i + 1) not in _wanted:
                    continue
                drawn_trips += 1
                col = TRIP_COLOURS[i % len(TRIP_COLOURS)]
                for u in trip:
                    trip_of[u] = col
                stops = [fac['shaft']] + list(trip) + [fac['shaft']]
                for a, b in zip(stops, stops[1:]):
                    seg = corridor_path(a, b)
                    xs = [xoff(n[0]) + n[1] + 0.5 for n in seg]
                    ys = [n[2] + 0.5 for n in seg]
                    # white underlay keeps overlapping routes legible
                    ax.plot(xs, ys, color='white', lw=6.4, alpha=0.85, zorder=6,
                            solid_capstyle='round')
                    ax.plot(xs, ys, color=col, lw=3.6, alpha=0.95, zorder=7,
                            solid_capstyle='round')
                # number the trip at its first collection point
                if trip:
                    _f = trip[0]
                    ax.text(xoff(_f[0]) + _f[1] + 0.5, _f[2] + 0.5, str(i + 1),
                            ha='center', va='center', fontsize=6.5,
                            fontweight='bold', color='white', zorder=13,
                            bbox=dict(boxstyle='circle,pad=0.16', fc=col,
                                      ec='white', lw=0.7))

        # ---- supply units: star sized by mass ----
        abandoned = set(abandoned or [])
        for i, u in enumerate(fac['supplies']):
            ws, cs, rs = u
            ox = xoff(ws)
            x, y = ox + cs + 0.5, rs + 0.5
            if u in abandoned:
                ax.plot(x, y, marker='x', ms=7, color=COL_DROPPED,
                        markeredgewidth=1.8, zorder=9)
            else:
                ax.plot(x, y, marker='*', markersize=_mass_size(MASS[u]),
                        color=trip_of.get(u, COL_SUPPLY),
                        markeredgecolor='white' if u in trip_of else COL_ENTRY,
                        markeredgewidth=0.8, zorder=9)
            if show_labels:
                ax.text(x + 0.30, y + 0.22,
                        f"S{i+1}", fontsize=5.2, color=COL_WALL, zorder=10)
                ax.text(x + 0.30, y - 0.42,
                        f"m{MASS[u]}/p{VALUE[u]}", fontsize=4.6,
                        color='#6B7480', zorder=10)

        # ---- shaft (the Memo 01 entry, now the extraction point) and exits ----
        we, ce, re = fac['shaft']
        ax.add_patch(plt.Circle((xoff(we) + ce + 0.5, re + 0.5), 0.34,
                                color=COL_ENTRY, zorder=11))
        ax.text(xoff(we) + ce + 0.5, re + 0.5, 'S', ha='center', va='center',
                fontsize=7, color='white', fontweight='bold', zorder=12)

        for lbl, (wx, cx, rx) in zip(['A', 'B'], fac['exits']):
            ax.add_patch(plt.Circle((xoff(wx) + cx + 0.5, rx + 0.5), 0.3,
                                    color=COL_EXIT, zorder=11))
            ax.text(xoff(wx) + cx + 0.5, rx + 0.5, lbl, ha='center', va='center',
                    fontsize=6, color='white', fontweight='bold', zorder=12)

        # ---- corridor-cost colourbar (unchanged from Memo 02) ----
        sm = plt.cm.ScalarMappable(cmap=_WEIGHT_CMAP,
                                   norm=mcolors.Normalize(vmin=1, vmax=5))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.018, pad=0.02)
        cbar.set_label('Corridor cost  w(e)' + ('  (muted)' if showing_plan else ''),
                       fontsize=8, color='#0B1F3B')
        cbar.set_ticks([1, 2, 3, 4, 5])
        cbar.ax.tick_params(labelsize=7)

        # ---- legend ----
        handles = [
            plt.Line2D([], [], marker='o', ls='', ms=7, color=COL_ENTRY,
                       label='S  extraction shaft (Memo 01 entry)'),
            plt.Line2D([], [], marker='o', ls='', ms=6, color=COL_EXIT,
                       label='A / B  exits'),
            plt.Line2D([], [], marker='*', ls='', ms=9, color=COL_SUPPLY,
                       markeredgecolor=COL_ENTRY, label='supply unit  (size = mass)'),
            plt.Line2D([], [], marker='x', ls='', ms=7, color=COL_DROPPED,
                       label='abandoned'),
            plt.Line2D([], [], ls='--', lw=2, color=COL_JUNCTION,
                       label='inter-wing junction'),
        ]
        if plan:
            _lbl = (f'shuttle trips ({drawn_trips} of {len(plan)} shown)'
                    if only_trips else
                    f'shuttle trips ({len(plan)}, numbered at first pickup)')
            handles.append(plt.Line2D([], [], lw=4, color=TRIP_COLOURS[0],
                                      label=_lbl))
        ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.02),
                  ncol=3, fontsize=7, framealpha=0.9, borderpad=0.6)

        ax.set_xlim(-0.5, total_w + 0.5)
        ax.set_ylim(-1.0, wr + 1.4)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=11, fontweight='bold',
                     color='#0B1F3B', pad=10)
        plt.tight_layout()
        return fig

    return (draw_facility,)


@app.cell
def facility_plot(draw_facility, mo, seed_input):
    mo.vstack([
        mo.md("""
    ### Your facility schematic

    Same rendering as your Memo 01 and Memo 02 workbooks -- corridors coloured by
    stability weight `w(e)`, walls in slate, wings bounded and labelled with their
    cost model, junctions dashed. **New in Memo 03:** each supply unit is marked
    with its mass and priority (`m2/p4` = mass 2, priority 4), and the star size
    shows the mass. The entry is now labelled **S** for extraction shaft.
    """),
        draw_facility(title=f"Emberlight Complex -- Seed {int(seed_input.value)}")
    ])
    return


@app.cell
def m30_header(mo):
    mo.md("""
    ---
    # [M3-0] Orientation

    *Not separately assessed -- but the gateway to [M3-1].*

    Below, the three reference heuristics from **Appendix M3-A** run on **your**
    facility. They all obey the same capacity and budget. They differ only in how
    they decide **what goes into each trip**.

    Study the spread between them before you design anything.
    """)
    return


@app.cell
def m30_exemplar_run(
    BUDGET,
    SUPPLIES,
    VALUE,
    exemplar_a_nearest_fill,
    exemplar_b_high_value,
    exemplar_c_value_per_mass,
    mo,
    plan_cost,
    plan_value,
    time,
):
    _rows = []
    exemplar_plans = {}
    for _name, _fn in [("A -- nearest-fill", exemplar_a_nearest_fill),
                       ("B -- highest-value-first", exemplar_b_high_value),
                       ("C -- priority-per-mass", exemplar_c_value_per_mass)]:
        _t0 = time.time()
        _plan = _fn(SUPPLIES, BUDGET)
        _ms = (time.time() - _t0) * 1000
        exemplar_plans[_name] = _plan
        _rows.append(
            f"| {_name} | {plan_value(_plan)} | "
            f"{sum(len(t) for t in _plan)} | {len(_plan)} | "
            f"{plan_cost(_plan):,.0f} | {_ms:,.1f} ms |"
        )

    _best = max(plan_value(p) for p in exemplar_plans.values())
    _worst = min(plan_value(p) for p in exemplar_plans.values())
    _spread = 100 * (_best / max(_worst, 1) - 1)

    mo.md(f"""
    ### The three exemplars on your facility

    | Heuristic | Priority value | Units | Trips | Energy used | Time |
    |---|---|---|---|---|---|
    {chr(10).join(_rows)}

    Budget `B` = **{BUDGET:,}**.
    Total priority value available = **{sum(VALUE.values())}**.

    > **Spread between best and worst: {_spread:.1f}%.** Same facility, same
    > budget, same hardware. The entire difference is the bundling rule.
    >
    > Note which two are closest together, and what they have in common. If two
    > opposite rules land near each other, neither rule is the thing that matters.
    """)
    return


@app.cell
def m30_input(SAVE_FILE_M03, json, mo, os):
    _saved = ""
    if os.path.exists(SAVE_FILE_M03):
        try:
            with open(SAVE_FILE_M03, "r") as _f:
                _d = json.load(_f)
            if _d:
                _saved = _d[-1].get("M30_orientation", "")
        except Exception:
            pass

    resp_m30 = mo.ui.text_area(
        label="**[M3-0] Which Memo 01/02 assumptions does this revision invalidate?**",
        value=_saved, rows=10, full_width=True,
        placeholder=(
            "List three to five, each naming something specific in your model.\n\n"
            "1. My Graph ADT stored only ... on each vertex, so it cannot represent ...\n"
            "2. My cost function had signature cost(edge) -- it takes no load argument, so ...\n"
            "3. My algorithm returned a single path. Under the shuttle protocol the answer is ...\n"
            "4. I assumed every supply unit would be collected. Under the battery budget\n"
            "   that is ...\n\n"
            "Then: which exemplar (A, B or C) does my Memo 01/02 algorithm most resemble,\n"
            "and what one decision does mine make differently?"
        )
    )
    resp_m30
    return (resp_m30,)


@app.cell
def _(mo, resp_m30):
    mo.callout(mo.md(resp_m30.value), kind="success")
    return


@app.cell
def m31_header(mo):
    mo.md("""
    ---
    # [M3-1] Improved Data Model & Algorithm
    ### Criterion 8 -- 300-500 words

    Design an improved **data model and algorithm combination** that correctly
    handles load-sensitive, capacity-constrained, **energy-limited** extraction.

    **The revised problem contains four decisions. Name the technique you apply
    to each, and say why it suits that decision.**

    | Decision | Question |
    |---|---|
    | **Selection** | which units are worth extracting at all, given `B`? |
    | **Grouping** | which units travel together in one trip, subject to `C`? |
    | **Ordering** | in what sequence within a trip, given mass accumulates? |
    | **Routing** | which corridors between consecutive collection points? |

    These interact. A unit worth taking on its own may not be worth taking once
    you account for the trip it forces you to fly.

    **Available helpers** -- use these rather than re-implementing the cost model:

    ```python
    SUPPLIES            # list of all supply unit nodes
    MASS[u], VALUE[u]   # mass and priority of unit u
    CAP                 # mass capacity (5)
    SHAFT, EXITS        # extraction shaft, exit nodes
    dist(u, v)          # cheapest corridor cost u -> v, carrying nothing
    corridor_path(u, v) # the actual sector list behind dist(u, v)
    trip_cost(trip)     # load-aware energy for one shuttle (list of units)
    plan_cost(plan)     # total energy for a list of trips + final exit run
    plan_value(plan)    # total priority delivered
    validate_plan(plan, budget)  # (ok, problems) -- ALWAYS run before quoting
    best_order(units)   # cheapest order within one trip (brute force)
    ```

    A **plan** is a list of trips; a **trip** is a list of supply unit nodes, in
    collection order. For example: `[[s1, s7], [s3], [s2, s9, s4]]`.
    """)
    return


@app.cell
def m31_input(SAVE_FILE_M03, json, mo, os):
    _saved = ""
    if os.path.exists(SAVE_FILE_M03):
        try:
            with open(SAVE_FILE_M03, "r") as _f:
                _d = json.load(_f)
            if _d:
                _saved = _d[-1].get("M31_design", "")
        except Exception:
            pass

    resp_m31 = mo.ui.text_area(
        label="**[M3-1] Improved data model and algorithm (300-500 words)**",
        value=_saved, rows=20, full_width=True,
        placeholder=(
            "PART 1 -- Limitations of my current data model\n"
            "  Which ADT definitions or assumptions are no longer valid, and why.\n\n"
            "PART 2 -- My revised data model\n"
            "  What state I now store (unit mass, unit priority, carried mass, trip\n"
            "  membership ...), which ADT operations I added or changed, and the\n"
            "  rationale for each change.\n\n"
            "PART 3 -- My improved algorithm\n"
            "  Grouping:  I use ... because ...\n"
            "  Ordering:  I use ... because ...\n"
            "  Routing:   I use ... because ...\n"
            "  Why this combination suits the revised problem better than my\n"
            "  Memo 01/02 algorithm:\n\n"
            "PART 4 -- Why trip order does not matter\n"
            "  (State the reason in your own words -- it follows from the load reset.)"
        )
    )
    resp_m31
    return (resp_m31,)


@app.cell
def _(mo, resp_m31):
    mo.callout(mo.md(resp_m31.value), kind="success")
    return


@app.cell
def m31_pseudocode(SAVE_FILE_M03, json, mo, os):
    _saved = ""
    if os.path.exists(SAVE_FILE_M03):
        try:
            with open(SAVE_FILE_M03, "r") as _f:
                _d = json.load(_f)
            if _d:
                _saved = _d[-1].get("M31_pseudocode", "")
        except Exception:
            pass

    resp_m31_pseudo = mo.ui.code_editor(
        label="**[M3-1] Pseudocode (Algorithmics metalanguage)**",
        value=_saved,
        placeholder=(
            "PlanExtraction(units: Set, C: Integer, B: Real) -> Plan:\n"
            "  1  ...\n"
            "  2  ...\n"
            "  3  While ... Do\n"
            "  4      ...\n"
            "  5  Return plan\n\n"
            "Annotate each line with its cost when you reach [M3-3]."
        ), language = 'text'
    )
    resp_m31_pseudo
    return (resp_m31_pseudo,)


@app.cell
def _(mo, resp_m31_pseudo):
    mo.callout(mo.md(resp_m31_pseudo.text), kind="success")
    return


@app.cell
def _(json, os):
    CACHE_VERSION = "v1"

    def load_or_compute(cache_path, version, key, compute_fn):
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            if cached.get("version") == version and cached.get("key") == key:
                return cached["data"]
        data = compute_fn()
        with open(cache_path, "w") as f:
            json.dump({"version": version, "key": key, "data": data}, f)
        return data

    def plan_to_jsonable(plan):
        return [[list(u) for u in trip] for trip in plan]

    def plan_from_jsonable(data):
        return [[tuple(u) for u in trip] for trip in data]

    return CACHE_VERSION, load_or_compute, plan_from_jsonable, plan_to_jsonable


@app.cell
def _(json, os):
    def load_or_render_gif(gif_path, version, key, render_fn):
        meta_path = gif_path + ".json"
        if os.path.exists(gif_path) and os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("version") == version and meta.get("key") == key:
                return gif_path

        os.makedirs(os.path.dirname(gif_path) or ".", exist_ok=True)
        render_fn(gif_path)

        with open(meta_path, "w") as f:
            json.dump({"version": version, "key": key}, f)
        return gif_path

    return (load_or_render_gif,)


@app.cell
def m31_code_header(mo):
    mo.md("""
    ### [M3-1 RUN] Implement your algorithm

    Replace the body of `my_algorithm` below with your own design. It must
    accept a pool of units and a budget, and return a **plan** (list of trips).

    The default implementation is a deliberately naive placeholder: it takes
    units in manifest order, one per trip. It is valid but poor -- beat it.
    """)
    return


@app.cell
def m31_student_algorithm(
    BUDGET,
    CACHE_VERSION,
    CAP,
    EXITS,
    EXIT_LEG,
    MASS,
    SHAFT,
    SUPPLIES,
    VALUE,
    dist,
    fac,
    itertools,
    load_or_compute,
    mo,
    os,
    plan_cost,
    plan_from_jsonable,
    plan_to_jsonable,
    plan_value,
    random,
    seed_input,
    trip_cost,
    trip_value,
    validate_plan,
):
    """
    Design:
        trained actor-critic policy with pheromone guided ant colony optimiser that runs an or-opt for every ant in every ACO restart + elite pheromone learning + 
        final supply swap search iterations and forced supply insertion checks
    """

    import torch
    import train_hybrid_aco_policies as trainmod

    """ 
    train_hybrid_aco_policies is the python file that contains the actor-critic model using torch - utilises 3 separate checkpoint files (1 for every fac size - 2/3/4)
    and each trained on 15000 unique instances
    """
    # Same helper calculator functions from earlier in marimo file, but copied directly here
    def _trip_cost(trip):
        here, total, load = SHAFT, 0.0, 0
        for u in trip:
            total += (1 + load) * dist(here, u)
            load += MASS[u]
            here = u
        total += (1 + load) * dist(here, SHAFT)
        return total

    def _plan_cost(plan):
        return sum(trip_cost(t) for t in plan) + EXIT_LEG

    def _plan_value(plan):
        return sum(trip_value(t) for t in plan)

    def _best_order(units):
        if len(units) <= 1:
            return list(units)
        return list(min(itertools.permutations(units), key=trip_cost))

    # Calculator functions that are instance specific to the instance fed into script at initialisation/runtime
    # They still are based off of above
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

    # Memoize per-trip calculations
    def add_caching(inst):
        # Wrap inst's best_order/trip_cost with memoization so that repeated calls for the same supply set are just a dict lookup, instead of full calculation.
        raw_best_order = inst["best_order"]
        raw_trip_cost = inst["trip_cost"]
        best_order_cache = {}
        trip_cost_cache = {}

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

    # Actor-Critic Policy Loader
    def load_policy_for_instance(inst, checkpoint_dir):
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

    # Validator
    def inst_validate(inst, plan):
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

        cost = aco_plan_cost(plan, inst)
        if cost > BUDGET + 1e-9:
            return False, f"Budget exceeded: {cost:.3f} > {BUDGET:.3f}"

        return True, "OK"

    # Start Pheromone at all points as 1.0
    def initialise_pheromone(inst):
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

    # Pheromone Updater
    def update_pheromone_elite(pheromone, ranked_plans, shaft, evaporation, elite_plan, elite_value, elitist_weight, min_pheromone=0.2, max_pheromone=12.0, inst=None):
        for key in pheromone:
            pheromone[key] *= 1.0 - evaporation

        if inst is not None:
            ranked = sorted(ranked_plans, key=lambda x: (x[1], -aco_plan_cost(x[0], inst)), reverse=True)
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

    def or_opt_plan(plan, inst, max_segment=3, first_improvement=True):
        CAP = inst["CAP"]
        BUDGET = inst["BUDGET"]
        MASS = inst["MASS"]
        trip_cost = inst["trip_cost"]
        EXIT_LEG = inst.get("EXIT_LEG", 0.0)
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
                                        candidate = copy_plan(current)
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

    # Iteration Based ACO
    def run_iterative_aco(inst, actor, critic, initial_best, n_ants=32, n_iterations=20, evaporation=0.15, elitist_weight=1.5, dist_scale=None, mode="sample", seed=0):
        if dist_scale is None:
            dist_scale = inst["EXIT_LEG"]

        rng = random.Random(seed)
        pheromone = initialise_pheromone(inst)

        global_best = copy_plan(initial_best)
        global_best_value = aco_plan_value(global_best, inst)
        global_best_cost = aco_plan_cost(global_best, inst)

        for iteration in range(1, n_iterations + 1):
            iter_plans = []
            raw_values = []

            for ant in range(n_ants):
                raw_plan, trajectory = trainmod.construct_plan(inst, actor, critic, pheromone, dist_scale, mode=mode)

                valid, _ = inst_validate(inst, raw_plan)
                plan_for_ant = raw_plan if valid else []

                if plan_for_ant:
                    plan_for_ant = or_opt_plan(plan_for_ant, inst, max_segment=3) 
                    plan_for_ant = top_up_plan(plan_for_ant, inst, max_insert_size=2) 

                value = aco_plan_value(plan_for_ant, inst)
                raw_values.append(value)
                iter_plans.append((copy_plan(plan_for_ant), value))

                cost = aco_plan_cost(plan_for_ant, inst)
                if value > global_best_value or (value == global_best_value and cost < global_best_cost - 1e-9):
                    global_best = copy_plan(plan_for_ant)
                    global_best_value = value
                    global_best_cost = cost

            iter_plans.append((copy_plan(global_best), global_best_value))
            update_pheromone_elite(pheromone, iter_plans, inst["SHAFT"], evaporation, global_best, global_best_value, elitist_weight, inst=inst)
        return global_best

    # Attempt to add uncollected supplies through a new trip, or insertion to existing trip when and if feasible
    def top_up_plan(plan, inst, max_insert_size=3):
        CAP = inst["CAP"]
        BUDGET = inst["BUDGET"]
        MASS = inst["MASS"]
        current = copy_plan(plan)

        while True:
            collected = {u for trip in current for u in trip}
            remaining = [u for u in inst["SUPPLIES"] if u not in collected]
            if not remaining:
                break

            current_value = aco_plan_value(current, inst)
            current_cost = aco_plan_cost(current, inst)
            best_candidate = None
            best_score = None
            max_size = min(max_insert_size, len(remaining))

            for size in range(1, max_size + 1):
                for combo in itertools.combinations(remaining, size):
                    combo_mass = sum(MASS[u] for u in combo)
                    if combo_mass > CAP + 1e-9:
                        continue

                    # Existing Trip Supply Insertion
                    for i in range(len(current)):
                        old_trip = current[i]
                        old_load = sum(MASS[u] for u in old_trip)
                        if old_load + combo_mass > CAP + 1e-9:
                            continue

                        candidate = copy_plan(current)
                        candidate[i] = inst["best_order"](candidate[i] + list(combo))

                        candidate_cost = aco_plan_cost(candidate, inst)
                        if candidate_cost > BUDGET + 1e-9:
                            continue

                        candidate_value = aco_plan_value(candidate, inst)
                        gain = candidate_value - current_value
                        extra_cost = candidate_cost - current_cost
                        score = (gain, -extra_cost)

                        if best_score is None or score > best_score:
                            best_score = score
                            best_candidate = candidate

                    # New Trip Creation
                    new_trip = inst["best_order"](list(combo))
                    candidate = copy_plan(current)
                    candidate.append(new_trip)

                    candidate_cost = aco_plan_cost(candidate, inst)
                    if candidate_cost > BUDGET + 1e-9:
                        continue

                    candidate_value = aco_plan_value(candidate, inst)
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

    # Swap Search - escape bad supply set by swapping expensive or low-value supplies with uncollected supplies
    def swap_search(plan, inst, max_remove=2, max_add=2, pool_size=14):
        MASS = inst["MASS"]
        VALUE = inst["VALUE"]
        current = copy_plan(plan)

        while True:
            collected = {u for trip in current for u in trip}
            remaining = [u for u in inst["SUPPLIES"] if u not in collected]
            if not remaining:
                break

            base_value = aco_plan_value(current, inst)
            base_cost = aco_plan_cost(current, inst)

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

    # Multiple Start Hybrid ACO Based Search
    def hybrid_solve(inst, actor, critic, seed, ants=32, iterations=20, evaporation=0.15, elitist_weight=1.5, mode="sample"):
        """
        Run multiple varying ACO configurations so that if one config enters local minima, next configs can attempt to escape it
        """
        master_rng = random.Random(seed)
        best = []
        best_source = "empty seed"

        configs = [
            {
                "ants": ants,
                "iterations": iterations,
                "evaporation": evaporation,
                "elitist_weight": elitist_weight,
            },
            {
                "ants": max(ants, 36),
                "iterations": max(iterations, 24),
                "evaporation": 0.22,
                "elitist_weight": 1.15,
            },
            {
                "ants": max(ants, 36),
                "iterations": max(iterations, 36),
                "evaporation": 0.12,
                "elitist_weight": 1.75,
            },
        ]

        for restart, cfg in enumerate(configs):
            run_seed = master_rng.randrange(1000000000)
            candidate = run_iterative_aco(inst, actor, critic, initial_best=best, n_ants=cfg["ants"], n_iterations=cfg["iterations"],
                evaporation=cfg["evaporation"], elitist_weight=cfg["elitist_weight"], mode=mode, seed=run_seed)

            if better_plan(candidate, best, inst):
                best = copy_plan(candidate)
                best_source = f"learned ACO restart {restart + 1}"

        # Check to see if any supply can be added that ACO missed
        topped_up = top_up_plan(best, inst, max_insert_size=3)
        if better_plan(topped_up, best, inst):
            best = copy_plan(topped_up)
            best_source = best_source + " + top-up"

        # Swap collected supplies in trips with uncollected supplies to check if any improvement possible
        swapped = swap_search(best, inst, max_remove=2, max_add=2)
        if better_plan(swapped, best, inst):
            best = copy_plan(swapped)
            best_source = best_source + " + swap search"

        # Check if any supply can now be added due to any freed up budget from swap search
        topped_up_again = top_up_plan(best, inst, max_insert_size=3)
        if better_plan(topped_up_again, best, inst):
            best = copy_plan(topped_up_again)
            best_source = best_source + " + top-up"

        return best, best_source

    # ACO CONFIG SETTINGS EDIT
    aco_checkpoint_dir = "MEMO_3/"
    aco_ants = 32
    aco_iterations = 20
    aco_evaporation = 0.15
    aco_elitist_weight = 1.5
    aco_sample_mode = True # Preventing greedy based ant movement
    aco_rng_seed = seed_input.value

    def my_algorithm(pool, budget):
        random.seed(aco_rng_seed)
        torch.manual_seed(aco_rng_seed)

        aco_inst = dict(n_wings=fac["n_wings"], SHAFT=SHAFT, EXITS=EXITS, SUPPLIES=pool, MASS=MASS, VALUE=VALUE, CAP=CAP, BUDGET=budget, EXIT_LEG=EXIT_LEG,
                dist=dist, trip_cost=_trip_cost, plan_cost=_plan_cost, plan_value=_plan_value, best_order=_best_order)

        aco_inst = add_caching(aco_inst)
        aco_actor, aco_critic, aco_policy_path = load_policy_for_instance(aco_inst, aco_checkpoint_dir)

        plan, aco_best_source = hybrid_solve(aco_inst, aco_actor, aco_critic, seed=aco_rng_seed, ants=aco_ants, iterations=aco_iterations, evaporation=aco_evaporation,
            elitist_weight=aco_elitist_weight, mode=aco_sample_mode)
        return plan

    def _compute_m31():
        plan = my_algorithm(SUPPLIES, BUDGET)
        return plan_to_jsonable(plan)

    _key = {"seed": aco_rng_seed, "budget": BUDGET}
    my_plan = plan_from_jsonable(load_or_compute("MEMO_3/cache_m31.json", CACHE_VERSION, _key, _compute_m31))

    _ok, _problems = validate_plan(my_plan, BUDGET)

    _msg = (f"**Valid plan.** Priority delivered **{plan_value(my_plan)}** "
            f"of {sum(VALUE.values())}, using **{plan_cost(my_plan):,.0f}** "
            f"of budget {BUDGET:,} across {len(my_plan)} trips."
            if _ok else
            "**Plan is invalid:**\n\n" + "\n".join(f"- {p}" for p in _problems))

    mo.show_code(mo.callout(mo.md(_msg), kind="success" if _ok else "danger"))
    return aco_rng_seed, my_algorithm, my_plan, torch, trainmod


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### At a Glance

    Budget spent, priority value captured, and supplies collected, each as
    a gauge against its own ceiling.
    """)
    return


@app.cell
def _(BUDGET, SUPPLIES, VALUE, my_plan, plan_cost, plan_value, plt):
    _cost = plan_cost(my_plan)
    _value = plan_value(my_plan)
    _max_value = sum(VALUE.values())
    _collected = {u for trip in my_plan for u in trip}
    _n_collected = len(_collected)
    _n_total = len(SUPPLIES)

    def _gauge(ax, frac, label, sub, color):
        frac = max(0.0, min(1.0, frac))
        ax.pie([frac, 1 - frac], startangle=90, counterclock=False,
               colors=[color, '#e5e9ee'],
               wedgeprops=dict(width=0.32, edgecolor='white'))
        ax.text(0, 0.08, f"{frac * 100:.0f}%", ha='center', va='center',
                fontsize=18, fontweight='bold', color=color)
        ax.text(0, -0.22, sub, ha='center', va='center', fontsize=10, color='white')
        ax.set_title(label, fontsize=12, color='white')
        ax.set_aspect('equal')

    _fig, _axes = plt.subplots(1, 3, figsize=(11, 3.6))
    _fig.patch.set_alpha(0.0)
    for _ax in _axes:
        _ax.patch.set_visible(False)

    _gauge(_axes[0], _cost / BUDGET, "Budget spent",
           f"{_cost:,.0f} / {BUDGET:,}", '#7A1E2C')
    _gauge(_axes[1], _value / _max_value, "Priority value captured",
           f"{_value} / {_max_value}", '#0B6E6B')
    _gauge(_axes[2], _n_collected / _n_total, "Supplies collected",
           f"{_n_collected} / {_n_total}", '#F59E0B')

    plt.tight_layout()
    _fig
    return


@app.cell
def m32_header(mo):
    mo.md("""
    ---
    # [M3-2] Quality of the Improved Solution
    ### Criterion 9

    **Replace the code or values below**

    Evaluate your improved solution by the **advantages it offers over your
    initial (Memo 01/02) solution**, on three dimensions: **efficiency**,
    **coherence**, and **fitness for purpose**.

    > **Score your old algorithm honestly.** The comparison cell below re-costs
    > your Memo 01/02 approach under the *revised* model `w(e) x (1 + L)`. Quoting
    > its old Memo 02 figure -- measured under a different cost model -- is the
    > single most common error in this action.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### I Did Not implement my algorithm below since it can't be applied, it runs factorially and would never complete in a normal timespan, and to make it work would require algorithm rewrites to account for budget constraints, etc, so I have added it below as a reference code cell, but it isn't runnable or connected.
    """)
    return


@app.cell
def m32_comparison(
    BUDGET,
    CAP,
    MASS,
    SHAFT,
    SUPPLIES,
    VALUE,
    dist,
    mo,
    my_plan,
    plan_cost,
    plan_value,
    trip_cost,
):
    def memo02_style_plan(pool, budget):
        """Your Memo 01/02 approach, re-costed under the REVISED model.

        Memo 02 planned a single continuous traversal collecting nearest-first
        and ignored both capacity and payload mass. Here we keep that decision
        rule but force it to obey the shuttle protocol, so the two plans are
        comparable. Replace with your own Memo 01/02 rule if it differed.
        """
        remaining, plan, spent = list(pool), [], 0.0
        here = SHAFT
        while remaining:
            trip, load = [], 0
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
            c = trip_cost(trip)          # note: NOT reordered -- greedy order kept
            if spent + c > budget:
                break
            spent += c
            plan.append(trip)
            for u in trip:
                remaining.remove(u)
            here = SHAFT
        return plan

    old_plan = memo02_style_plan(SUPPLIES, BUDGET)

    _maxv = sum(VALUE.values())
    _rows = [
        ("Priority value delivered",
         f"{plan_value(old_plan)} of {_maxv}", f"{plan_value(my_plan)} of {_maxv}"),
        ("Units recovered",
         f"{sum(len(t) for t in old_plan)} of {len(SUPPLIES)}",
         f"{sum(len(t) for t in my_plan)} of {len(SUPPLIES)}"),
        ("Trips flown", f"{len(old_plan)}", f"{len(my_plan)}"),
        ("Energy used", f"{plan_cost(old_plan):,.0f}", f"{plan_cost(my_plan):,.0f}"),
        ("Energy left unspent",
         f"{BUDGET - plan_cost(old_plan):,.0f}",
         f"{BUDGET - plan_cost(my_plan):,.0f}"),
    ]
    _body = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in _rows)
    _delta = plan_value(my_plan) - plan_value(old_plan)

    mo.md(f"""
    ### Comparative output -- both algorithms, same facility, revised cost model

    | | Memo 01/02 approach | My improved algorithm |
    |---|---|---|
    {_body}

    **Difference in priority delivered: {_delta:+d}.**

    *Both plans are costed under `w(e) x (1 + L)` with capacity {CAP} and budget
    {BUDGET:,}.*
    """)
    return (memo02_style_plan,)


@app.cell
def m32_input(SAVE_FILE_M03, json, mo, os):
    _saved = ""
    if os.path.exists(SAVE_FILE_M03):
        try:
            with open(SAVE_FILE_M03, "r") as _f:
                _d = json.load(_f)
            if _d:
                _saved = _d[-1].get("M32_quality", "")
        except Exception:
            pass

    resp_m32 = mo.ui.text_area(
        label="**[M3-2] Quality of your improved solution**",
        value=_saved, rows=18, full_width=True,
        placeholder=(
            "EFFICIENCY\n"
            "  Complexity of improved vs original; which cases were costly or\n"
            "  infeasible before and are handled better now; any correctness or\n"
            "  optimality guarantees gained or lost.\n\n"
            "COHERENCE\n"
            "  Does the improved data model integrate cleanly with the improved\n"
            "  algorithm? Any state stored but unused, or recomputed because the\n"
            "  model does not hold it?\n\n"
            "FITNESS FOR PURPOSE\n"
            "  Does it satisfy the mission directive better, and under what\n"
            "  conditions? Where does it still fall short?\n\n"
            "MY COUNTEREXAMPLE (see Side Memo S3-B)\n"
            "  Two units P and Q, fully costed both ways, showing the mechanism."
        )
    )
    resp_m32
    return (resp_m32,)


@app.cell
def _(mo, resp_m32):
    mo.callout(mo.md(resp_m32.value), kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## S3-B Side Memo
    """)
    return


@app.cell
def _(mo):
    _resp = "Consider 3 units along a single corridor within the first wing, such that edge weights are uniform 1. The 3 units are P, R, and Q such that they are of respective distance to the shaft of 2, 5 and 8 and have mass of 3, 1 and 1. As total mass = 5, they can all be collected in 1 trip. Distances between them are PR = 3, PQ = 6 and RQ = 3. \n \n Six possible combinations of collection orders exist as shown below, with load accumulating as each unit is picked up: \n \n   "

    _data = [
        {"Order": "Q, R, P", "Cost": 35, "Notes": "Best"},
        {"Order": "R, Q, P", "Cost": 41, "Notes": ""},
        {"Order": "Q, P, R", "Cost": 65, "Notes": ""},
        {"Order": "P, Q, R", "Cost": 71, "Notes": ""},
        {"Order": "P, R, Q", "Cost": 77, "Notes": "Nearest-first greedy"},
        {"Order": "R, P, Q", "Cost": 89, "Notes": "Worst"},
    ]

    _table = mo.ui.table(
        data=_data,
        selection=None,
        text_justify_columns={
            "Order": "center",
            "Cost": "center",
            "Notes": "center",
        }
    )

    _resp_2 = "Nearest First greedy visits P, then R, then Q, each decision being the locally distance-wise cheapest. This creates the second worst of all six orderings - 77. Optimal collection order is when unit P (the heaviest) is collected last, and forcing Q, the furthest distance unit, to be collected first so that the longest traversals are kept unloaded for as long as possible. This achieves a collection cost of only 35, which is less than half of nearest-first. \n \n In the event that the mission budget for the trip was only 50, then nearest first's trip of cost 77 would be discarded, and the full value wouldn't be collected, even though the optimal route of 35 fits within the budget, and so shows that optimal full collection is possible. It shows that greedy's blindness to load doesn't just cost extra, but can turn a full collection trip into a trip with lost value."

    mo.callout(
        mo.vstack([
            _resp, 
            mo.center(_table),
            _resp_2
        ]), kind = 'success'
    )
    return


@app.cell
def obs_b_divider(mo):
    mo.md("""
    ---
    # Part B
    ### [M3-3] complexity - [M3-4] intractability - [M3-5] comparison - [M3-6] coherence

    Nothing new is added to the problem here. Observation A asked you to *build*
    something; Observation B asks you to **account** for it -- how expensive it
    is, whether an optimal answer was ever reachable, and how it compares with
    what you brought in from Memo 01/02.

    > You can start any of these as soon as you have a running algorithm. Do not
    > wait until Week 9.
    """)
    return


@app.cell
def m33_header(mo):
    mo.md("""
    ---
    # [M3-3] Time Complexity of the Improved Solution
    ### Criterion 5b --- 150-250 words

    Determine and justify the **tight upper bound** on your improved algorithm.

    Two things to keep separate:

    - the **corridor pathfinding** layer -- all pairwise distances, computed once
      up front in `O(k(V+E) log V)`;
    - the **decision** layer above it -- grouping, ordering and selection.

    These are usually in different complexity classes. Only one is the
    bottleneck. Say which, and why.

    The benchmarking cell below runs your algorithm at increasing `k` so you can
    compare **measured** growth against your **predicted** bound.
    """)
    return


@app.cell
def m33_benchmark(
    BUDGET,
    CACHE_VERSION,
    SUPPLIES,
    aco_rng_seed,
    load_or_compute,
    mo,
    my_algorithm,
    plan_value,
    plt,
    time,
):
    def _compute_m33():
        _ks, _times, _vals = [], [], []
        for _k in range(4, len(SUPPLIES) + 1, 2):
            _pool = SUPPLIES[:_k]
            _t0 = time.time()
            _p = my_algorithm(_pool, BUDGET)
            _el = (time.time() - _t0) * 1000
            _ks.append(_k)
            _times.append(_el)
            _vals.append(plan_value(_p))
        return {"ks": _ks, "times": _times, "vals": _vals}

    _key = {"seed": aco_rng_seed, "budget": BUDGET}
    _result = load_or_compute("MEMO_3/cache_m33.json", CACHE_VERSION, _key, _compute_m33)
    ks, times, vals = _result["ks"], _result["times"], _result["vals"]

    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.6))
    _axes[0].plot(ks, times, marker='o', color='#0B6E6B')
    _axes[0].set_xlabel("supply units k")
    _axes[0].set_ylabel("run time (ms)")
    _axes[0].set_title("Measured running time of your algorithm", fontsize=10)
    _axes[0].grid(alpha=0.3)
    _axes[1].plot(ks, vals, marker='s', color='#7A1E2C')
    _axes[1].set_xlabel("supply units k")
    _axes[1].set_ylabel("priority value delivered")
    _axes[1].set_title("Value delivered within budget B", fontsize=10)
    _axes[1].grid(alpha=0.3)
    plt.tight_layout()

    _tbl = "\n".join(f"| {a} | {b:,.2f} | {c} |" for a, b, c in zip(ks, times, vals))
    mo.vstack([
        _fig,
        mo.md(f"""
    | k | time (ms) | priority delivered |
    |---|---|---|
    {_tbl}
    """)
    ])
    return ks, times


@app.cell
def _(ks, plt, times):
    import numpy as np

    _k_arr = np.array(ks)
    _t_arr = np.array(times)
    _log_k = np.log(_k_arr)
    _log_t = np.log(_t_arr)
    _b_fit, _log_a_fit = np.polyfit(_log_k, _log_t, 1)
    _a_fit = np.exp(_log_a_fit)

    _k_smooth = np.linspace(_k_arr.min(), _k_arr.max(), 200)
    _t_smooth = _a_fit * np.power(_k_smooth, _b_fit)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(_k_arr, _t_arr, 'o', color='teal', label='Measured runtime')
    ax.plot(_k_smooth, _t_smooth, '-', color='orange',
            label=f'Fitted: $t = {_a_fit:.1f} \\cdot k^{{{_b_fit:.2f}}}$')

    ax.set_xlabel('supply units k')
    ax.set_ylabel('run time (ms)')
    ax.set_title('Measured running time vs. fitted power law')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig
    return (np,)


@app.cell
def m33_input(SAVE_FILE_M03, json, mo, os):
    _saved = ""
    if os.path.exists(SAVE_FILE_M03):
        try:
            with open(SAVE_FILE_M03, "r") as _f:
                _d = json.load(_f)
            if _d:
                _saved = _d[-1].get("M33_complexity", "")
        except Exception:
            pass

    resp_m33 = mo.ui.text_area(
        label="**[M3-3] Time complexity of your improved solution (150-250 words)**",
        value=_saved, rows=14, full_width=True,
        placeholder=(
            "PART 1 -- Line-by-line annotation of my pseudocode\n"
            "  Line 1: ... O(...)\n"
            "  Line 4: loop runs ... times, body costs ... -> multiplied\n\n"
            "PART 2 -- Pathfinding layer vs decision layer\n"
            "  All-pairs distances cost O(k(V+E) log V), computed once.\n"
            "  My decision layer costs ... . The bottleneck is ... because ...\n\n"
            "PART 3 -- Combining to a tight upper bound\n"
            "  O(______), driven by the parameter ...\n\n"
            "PART 4 -- Measured vs predicted\n"
            "  At k = ..., predicted ... , benchmark measured ... . The discrepancy is\n"
            "  explained by ..."
        )
    )
    resp_m33
    return (resp_m33,)


@app.cell
def _(mo, resp_m33):
    mo.callout(mo.md(resp_m33.value), kind="success")
    return


@app.cell
def m34_header(mo):
    mo.md("""
    ---
    # [M3-4] Intractability and the Case for a Heuristic
    ### Feeds the upper bands of C5b and C7 -- 200-300 words

    Below is a **calibration wing**: a reduced facility of only `k = 14` supply
    units, together with an **exact solver** that is guaranteed to return the
    optimal plan on it.

    This is the only place in the whole task where you can state your solution
    quality **with certainty**. Use it.

    > **Test under both budgets.** The solver runs at your mission budget `B` and
    > at the harsher contingency budget `B_reserve`. Which approach comes out best
    > is **not stable** across the two -- so a single measurement at a single
    > budget is not evidence that an approach is good.
    """)
    return


@app.cell
def exact_solver(CAP, EXIT_LEG, MASS, VALUE, best_order, itertools, trip_cost):
    """Exact solver: maximise delivered priority value within budget.

    Enumerates every capacity-feasible bundle, then runs a subset DP over which
    units are delivered. Correct, and exponential in k -- which is the point.
    """

    def solve_exact(pool, budget, max_bundle=5):
        k = len(pool)
        idx = {u: i for i, u in enumerate(pool)}

        # 1. every capacity-feasible bundle and its optimal cost
        bundles = {}
        for size in range(1, max_bundle + 1):
            for combo in itertools.combinations(pool, size):
                if sum(MASS[u] for u in combo) > CAP:
                    continue
                m = 0
                for u in combo:
                    m |= 1 << idx[u]
                bundles[m] = trip_cost(best_order(combo))

        by_low = {}
        for b, c in bundles.items():
            by_low.setdefault(b & -b, []).append((b, c))

        # 2. best[mask] = min energy to deliver exactly `mask`
        INF = float('inf')
        best = [INF] * (1 << k)
        best[0] = 0.0
        for mask in range(1, 1 << k):
            low = mask & -mask
            m = INF
            for b, c in by_low.get(low, ()):
                if b & mask == b:
                    prev = best[mask ^ b]
                    if prev < INF and prev + c < m:
                        m = prev + c
            best[mask] = m

        # 3. pick the highest-value affordable mask
        cap_energy = budget - EXIT_LEG
        bv, bm = 0, 0
        for mask in range(1 << k):
            if best[mask] <= cap_energy:
                v = 0
                mm = mask
                while mm:
                    v += VALUE[pool[(mm & -mm).bit_length() - 1]]
                    mm &= mm - 1
                if v > bv or (v == bv and best[mask] < best[bm]):
                    bv, bm = v, mask

        # 4. reconstruct the bundles chosen
        chosen, mask = [], bm
        while mask:
            low = mask & -mask
            for b, c in by_low.get(low, ()):
                if b & mask == b and abs(best[mask] - (best[mask ^ b] + c)) < 1e-9:
                    chosen.append(best_order([pool[i] for i in range(k) if b >> i & 1]))
                    mask ^= b
                    break
            else:
                break
        return bv, chosen, best[bm] + EXIT_LEG, len(bundles)

    return (solve_exact,)


@app.cell
def m34_calibration(
    CACHE_VERSION,
    SUPPLIES,
    aco_rng_seed,
    exemplar_a_nearest_fill,
    exemplar_b_high_value,
    exemplar_c_value_per_mass,
    load_or_compute,
    mo,
    my_algorithm,
    plan_cost,
    plan_value,
    solve_exact,
):
    def _compute_m34_calibration():
        _algos = [("A", exemplar_a_nearest_fill),
                  ("B", exemplar_b_high_value),
                  ("C", exemplar_c_value_per_mass),
                  ("Yours", my_algorithm)]

        _rows = []
        _last_summary = {}

        for _k in range(5, 21):
            _pool = SUPPLIES[:_k]
            _full = plan_cost(exemplar_a_nearest_fill(_pool, float('inf')))
            _mission_bud = round(_full * 0.60)
            _reserve_bud = round(_full * 0.35)

            _row = {"k": _k}
            for _label, _bud in [("M", _mission_bud), ("C", _reserve_bud)]:
                _optv, _optplan, _opte, _nbundles = solve_exact(_pool, _bud)
                _row[f"{_label}-Optimum"] = _optv

                _best, _bestname = -1, ""
                for _name, _fn in _algos:
                    _p = _fn(_pool, _bud)
                    _v = plan_value(_p)
                    _gap = 100 * (1 - _v / _optv) if _optv else 0.0
                    _row[f"{_label}-{_name}-val"] = _v
                    _row[f"{_label}-{_name}-gap%"] = round(_gap, 1)
                    if _name in ("A", "B", "C") and _v > _best:
                        _best, _bestname = _v, _name
                _last_summary[_label] = _bestname
            _rows.append(_row)

        return {"rows": _rows, "last_k": 15, "last_summary": _last_summary}


    _key = {"seed": aco_rng_seed, "k_range": [5, 15], "budget_props": [0.60, 0.35]}
    _calib = load_or_compute("MEMO_3/cache_m34_calibration.json", CACHE_VERSION, _key,
                              _compute_m34_calibration)

    def _split_rows(rows, label, budget_name):
        """Pull just one budget's columns out of the wide per-k row dicts,
        combining val + gap% into a single compact string per algorithm."""
        _algo_names = ["A", "B", "C", "Yours"]
        _out = []
        for _r in rows:
            _sub = {"k": _r["k"], "Optimum": _r[f"{label}-Optimum"]}
            for _name in _algo_names:
                _val = _r[f"{label}-{_name}-val"]
                _gap = _r[f"{label}-{_name}-gap%"]
                _sub[_name] = f"{_val} (-{_gap:.1f}%)" if _gap > 0 else f"{_val} (optimal)"
            _out.append(_sub)
        return _out


    _mission_rows = _split_rows(_calib["rows"], "M", "Mission")
    _reserve_rows = _split_rows(_calib["rows"], "C", "Contingency")

    _mission_table = mo.ui.table(
        data=_mission_rows,
        selection=None,
        text_justify_columns={col: "center" for col in _mission_rows[0].keys()},
        page_size=25
    )
    _reserve_table = mo.ui.table(
        data=_reserve_rows,
        selection=None,
        text_justify_columns={col: "center" for col in _reserve_rows[0].keys()},
        page_size=25
    )

    _flip = _calib["last_summary"]["M"] != _calib["last_summary"]["C"]

    mo.vstack([
        mo.md("""
    ### Calibration sweep -- k = 5 to 20

    For each facility size, the exact optimum and what each exemplar (and your
    algorithm) actually achieved, with the resulting gap below optimum (%).
    """),
        mo.md("**Mission budget (60% of full run)**"),
        mo.center(_mission_table),
        mo.md("**Contingency budget (35% of full run)**"),
        mo.center(_reserve_table)
    ])
    return


@app.cell
def m34_wall_header(mo):
    mo.md("""
    ### The tractability wall

    Run the exact solver at increasing `k` and watch the cost of certainty.

    **Start with `max k = 16`.** Each step of +2 multiplies the work by roughly
    five. Do not set this above 20 unless you are prepared to wait.
    """)
    return


@app.cell
def m34_wall_control(mo):
    wall_k = mo.ui.slider(start=8, stop=20, step=2, value=16,
                          label="Largest k to solve exactly", show_value=True)
    wall_k
    return (wall_k,)


@app.cell
def m34_wall_run(
    CACHE_VERSION,
    SUPPLIES,
    aco_rng_seed,
    exemplar_a_nearest_fill,
    load_or_compute,
    mo,
    plan_cost,
    plt,
    solve_exact,
    time,
    wall_k,
):
    def _compute_m34_wall(max_k):
        _ks, _secs = [], []
        for _k in range(8, max_k + 1, 2):
            _pool = SUPPLIES[:_k]
            _b = round(plan_cost(exemplar_a_nearest_fill(_pool, float('inf'))) * 0.60)
            _t0 = time.time()
            solve_exact(_pool, _b)
            _secs.append(time.time() - _t0)
            _ks.append(_k)
        return {"ks": _ks, "secs": _secs}

    # Cached per slider value -- moving the slider back to a value you've
    # already run loads instantly instead of recomputing.
    _key = {"seed": aco_rng_seed, "wall_k": wall_k.value}
    _wall = load_or_compute(f"MEMO_3/cache_m34_wall_{wall_k.value}.json", CACHE_VERSION,
                             _key, lambda: _compute_m34_wall(wall_k.value))
    _ks, _secs = _wall["ks"], _wall["secs"]

    _fig, _ax = plt.subplots(figsize=(6.4, 3.6))
    _ax.semilogy(_ks, [max(s, 1e-4) for s in _secs], marker='o', color='#7A1E2C')
    _ax.set_xlabel("supply units k")
    _ax.set_ylabel("exact solver time (s, log scale)")
    _ax.set_title("Cost of a guaranteed optimal answer", fontsize=10)
    _ax.grid(alpha=0.3, which='both')
    plt.tight_layout()

    # empirical growth factor per +2 units, from the last two measured points
    if len(_secs) >= 3 and _secs[-3] > 0:
        _g = (_secs[-1] / _secs[-3]) ** 0.5
    else:
        _g = float('nan')

    _rows = "\n".join(
        f"| {a} | {2**a:,} | {b:,.3f} s | {2**a * 8 / 1e9:,.3f} GB |"
        for a, b in zip(_ks, _secs)
    )
    _proj = ""
    if _g == _g and _g > 1:
        _t = _secs[-1]
        _lines = []
        for _kk in range(wall_k.value + 2, len(SUPPLIES) + 1, 2):
            _t *= _g ** 2
            _txt = (f"{_t:,.0f} s" if _t < 3600 else
                    f"{_t/3600:,.1f} hours" if _t < 86400 else f"{_t/86400:,.1f} days")
            _lines.append(
                f"| {_kk} | {2**_kk:,} | *{_txt}* | *{2**_kk * 8 / 1e9:,.2f} GB* |")
        _proj = "\n".join(_lines)

    mo.vstack([
        _fig,
        mo.md(f"""
    | k | subsets 2^k | time | minimum memory |
    |---|---|---|---|
    {_rows}
    {_proj}

    Measured growth factor: **{_g:.2f}x per two additional supply units**
    (italic rows are extrapolated).

    > Your facility has **k = {len(SUPPLIES)}**. Note that the **memory** column
    > reaches the wall before the time column does -- on a school laptop the
    > exact solver at k = {len(SUPPLIES)} does not run slowly, it fails to start.
    """)
    ])
    return


@app.cell
def m34_input(SAVE_FILE_M03, json, mo, os):
    _saved = ""
    if os.path.exists(SAVE_FILE_M03):
        try:
            with open(SAVE_FILE_M03, "r") as _f:
                _d = json.load(_f)
            if _d:
                _saved = _d[-1].get("M34_intractability", "")
        except Exception:
            pass

    resp_m34 = mo.ui.text_area(
        label="**[M3-4] Intractability and the case for a heuristic (200-300 words)**",
        value=_saved, rows=16, full_width=True,
        placeholder=(
            "1. EXACT OPTIMISATION DOES NOT SCALE\n"
            "   Measured on my machine: k=... took ...s, k=... took ...s.\n"
            "   Growth factor ~...x per two units. At my k = 30 that implies ... and\n"
            "   ... GB of memory.\n\n"
            "2. WHY THE PROBLEM IS INTRACTABLE, NOT JUST MY CODE SLOW\n"
            "   Number of candidate solutions: ... . No rearrangement of the search\n"
            "   avoids examining exponentially many because ...\n"
            "   Structures my problem contains: ... (knapsack-shaped / bin-packing-\n"
            "   shaped / tour-shaped -- name them).\n\n"
            "3. MY HEURISTIC MEASURED AGAINST GROUND TRUTH, AT BOTH BUDGETS\n"
            "   Mission budget:     mine ... vs optimum ...  -> gap ...%\n"
            "   Contingency budget: mine ... vs optimum ...  -> gap ...%\n"
            "   Did the ranking of the approaches change between the two? What that\n"
            "   tells me about claiming an approach is 'better': ...\n"
            "   What this does and does not tell me about the full facility: ...\n\n"
            "4. WHAT I GAVE UP AND WHY IT IS THE RIGHT TRADE HERE"
        )
    )
    resp_m34
    return (resp_m34,)


@app.cell
def _(mo, resp_m34):
    mo.callout(mo.md(resp_m34.value), kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SIDE MEMO S3-C
    """)
    return


@app.cell
def _(
    CACHE_VERSION,
    EXIT_LEG,
    MASS,
    SHAFT,
    SUPPLIES,
    VALUE,
    aco_rng_seed,
    dist,
    exemplar_a_nearest_fill,
    itertools,
    load_or_compute,
    mo,
    my_algorithm,
    pd,
    plan_cost,
    plan_value,
):
    def held_karp_order(units, shaft, dist_fn, mass_dict):
        n = len(units)
        if n <= 1:
            return list(units)

        idx = list(units)
        full = (1 << n) - 1

        load_of = {
            mask: sum(mass_dict[idx[i]] for i in range(n) if mask & (1 << i))
            for mask in range(1, full + 1)
        }

        dp = {}
        parent = {}

        for i in range(n):
            m = 1 << i
            dp[(m, i)] = dist_fn(shaft, idx[i])
            parent[(m, i)] = None

        for mask in range(1, full + 1):
            for i in range(n):
                if not (mask & (1 << i)) or (mask, i) not in dp:
                    continue

                load_after_mask = load_of[mask]

                for j in range(n):
                    if mask & (1 << j):
                        continue

                    new_mask = mask | (1 << j)
                    cost = dp[(mask, i)] + (1 + load_after_mask) * dist_fn(idx[i], idx[j])
                    key = (new_mask, j)

                    if key not in dp or cost < dp[key]:
                        dp[key] = cost
                        parent[key] = (mask, i)

        best_end, best_cost = None, float("inf")
        for i in range(n):
            key = (full, i)
            if key not in dp:
                continue
            total = dp[key] + (1 + load_of[full]) * dist_fn(idx[i], shaft)
            if total < best_cost:
                best_cost, best_end = total, i

        order, key = [], (full, best_end)
        while key is not None:
            mask, i = key
            order.append(idx[i])
            key = parent[key]
        order.reverse()
        return order

    def solve_exact_relaxed(pool, budget, relaxed_cap, max_bundle, MASS, VALUE, EXIT_LEG, SHAFT, dist):
        k = len(pool)
        idx = {u: i for i, u in enumerate(pool)}

        def bundle_cost(combo):
            order = held_karp_order(list(combo), SHAFT, dist, MASS)
            here, total, load = SHAFT, 0.0, 0
            for u in order:
                total += (1 + load) * dist(here, u)
                load += MASS[u]
                here = u
            total += (1 + load) * dist(here, SHAFT)
            return total, order

        bundles, bundle_orders = {}, {}
        for size in range(1, max_bundle + 1):
            for combo in itertools.combinations(pool, size):
                if sum(MASS[u] for u in combo) > relaxed_cap:
                    continue
                m = 0
                for u in combo:
                    m |= 1 << idx[u]
                cost, order = bundle_cost(combo)
                bundles[m] = cost
                bundle_orders[m] = order

        by_low = {}
        for b, c in bundles.items():
            by_low.setdefault(b & -b, []).append((b, c))

        INF = float("inf")
        best = [INF] * (1 << k)
        best[0] = 0.0
        for mask in range(1, 1 << k):
            low = mask & -mask
            m = INF
            for b, c in by_low.get(low, ()):
                if b & mask == b:
                    prev = best[mask ^ b]
                    if prev < INF and prev + c < m:
                        m = prev + c
            best[mask] = m

        cap_energy = budget - EXIT_LEG
        bv, bm = 0, 0
        for mask in range(1 << k):
            if best[mask] <= cap_energy:
                v = 0
                mm = mask
                while mm:
                    v += VALUE[pool[(mm & -mm).bit_length() - 1]]
                    mm &= mm - 1
                if v > bv or (v == bv and best[mask] < best[bm]):
                    bv, bm = v, mask

        chosen, mask = [], bm
        while mask:
            low = mask & -mask
            for b, c in by_low.get(low, ()):
                if b & mask == b and abs(best[mask] - (best[mask ^ b] + c)) < 1e-9:
                    chosen.append(bundle_orders[b])
                    mask ^= b
                    break
            else:
                break

        return bv, chosen, best[bm] + EXIT_LEG, len(bundles)

    _K_TESTS = list(range(10, 16))
    _BUDGET_PROPS = [("Mission", 0.60), ("Contingency", 0.35)]
    _MAX_BUNDLE = 15  
    _RELAXED_CAP = 200.0 

    def _compute_ext_relaxed_bound():
        _results = []

        for _k_test in _K_TESTS:
            _pool = SUPPLIES[:_k_test]
            _full_cost = plan_cost(exemplar_a_nearest_fill(_pool, float("inf")))

            for _budget_label, _budget_prop in _BUDGET_PROPS:
                _budget = round(_full_cost * _budget_prop)

                _heuristic_plan = my_algorithm(_pool, _budget)
                _heuristic_value = plan_value(_heuristic_plan)

                _bound_value, _chosen_bundles, _bound_cost, _n_bundles = solve_exact_relaxed(
                    _pool, _budget, relaxed_cap=_RELAXED_CAP, max_bundle=_MAX_BUNDLE,
                    MASS=MASS, VALUE=VALUE, EXIT_LEG=EXIT_LEG, SHAFT=SHAFT, dist=dist,
                )

                assert _bound_cost <= _budget + 1e-6, (
                    f"Relaxed exact solution exceeds budget at "
                    f"k={_k_test}, {_budget_label}: {_bound_cost} > {_budget}"
                )
                assert _bound_value >= _heuristic_value, (
                    f"Bound fell below the heuristic at "
                    f"k={_k_test}, {_budget_label}: bound={_bound_value} "
                    f"< heuristic={_heuristic_value} -- relaxation is invalid, stop and check"
                )

                _bundle_sizes = [len(b) for b in _chosen_bundles]
                _saturated = bool(_bundle_sizes) and max(_bundle_sizes) == _MAX_BUNDLE
                _gap_pct = (
                    100.0 * (_bound_value - _heuristic_value) / _bound_value
                    if _bound_value > 0 else 0.0
                )

                _results.append({
                    "budget_label": _budget_label,
                    "k": _k_test,
                    "budget": _budget,
                    "heuristic_value": _heuristic_value,
                    "bound_value": _bound_value,
                    "bound_vs_heuristic": _bound_value - _heuristic_value,
                    "gap_pct": round(_gap_pct, 2),
                    "bound_cost": round(_bound_cost, 1),
                    "n_bundles": _n_bundles,
                    "bundle_sizes": _bundle_sizes,
                    "saturated": _saturated,
                })

        return {"rows": _results}

    _key = {
        "seed": aco_rng_seed,
        "k_range": [_K_TESTS[0], _K_TESTS[-1]],
        "budget_props": [p for _, p in _BUDGET_PROPS],
        "max_bundle": _MAX_BUNDLE,
        "relaxed_cap": _RELAXED_CAP,
        "kind": "ext_relaxed_capacity_bound",
    }
    _cached = load_or_compute(
        "MEMO_3/cache_ext_relaxed_bound.json", CACHE_VERSION, _key,
        _compute_ext_relaxed_bound,
    )

    ext_k_df = pd.DataFrame(_cached["rows"])
    _mission_df = ext_k_df[ext_k_df["budget_label"] == "Mission"]
    _contingency_df = ext_k_df[ext_k_df["budget_label"] == "Contingency"]

    mo.vstack([
        mo.md(f"""
    ### Relaxed-capacity exact bound across k=10-15, both budgets

    The same facility instance and budget convention as the M3-4 calibration
    sweep is evaluated at each pool size, under both the mission (60%) and
    contingency (35%) budgets. The hybrid solution is compared against the
    exact capacity-relaxed bound for that same `k` and budget.

    **Mission:** mean gap {_mission_df['gap_pct'].mean():.2f}%, worst case {_mission_df['gap_pct'].max():.2f}%
    **Contingency:** mean gap {_contingency_df['gap_pct'].mean():.2f}%, worst case {_contingency_df['gap_pct'].max():.2f}%
    """),

        mo.ui.table(
            ext_k_df[[
                "budget_label", "k", "budget", "heuristic_value", "bound_value",
                "bound_vs_heuristic", "gap_pct", "bound_cost", "n_bundles",
                "bundle_sizes", "saturated",
            ]],
            selection=None,
            page_size=20
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Response:

    On the full k = 30 facility, the optimum is never known. This means that any quality claim regarding my solution must be made through comparisons rather than a single number. I used all three techniques described by the side memo, with each targeting a differing weakness of the others.

    #### Ground truth on a smaller instance:

    The exact DP solver stays tractable up to k = 16 to 20 depending on which benchmark I run (0.378s at k=16 and so extrapolation shows k = 30 to be around 6.9 hours). As this isn't practical, I scaled the facility down and then made comparisons. Across k = 5 to 20, under both the mission and contingency budgets, my algorithm matches the exact optimum in all 32/32 tested cases — a 0% gap at every tested k, on both budgets (calibration sweep). That's stronger than expected but still isn't a proof since a perfect record at k = 20 doesn't guarantee that the same will occur at k = 30, as the search space grows exponentially, while the heuristic's own structure (fixed feature vector, bounded local search) doesn't scale. So I treat near certain optimality at k = 30 as an assumption instead of a conclusion the data at k <= 20 proves.

    #### Bounding from the other side:

    Separate to earlier tests, I constructed an upper bound by relaxing the constraint that makes trip grouping hard - removing the capacity limit entirely means any subset of supplies affordable under the budget can be collected however is preferable without the added restriction that a single trip can't exceed the shuttle's real capacity. Because dropping a constraint can only let the solver do as well or better than the true capacity-limited problem, the resulting value is provably >= the true optimum. Across k = 10 to 15 (as shown in the data table above), under both the mission and contingency budgets, this relaxed bound matched my heuristic's delivered value exactly in 6 of 12 tested cases (50%), with the remaining gaps ranging from 2.94% to 7.69% (an overall average of 2.29%). The contingency budget is consistently looser than mission (average of 2.98% vs 1.60%). This is consistent with the pattern already visible: that a tighter budget leaves less slack between the relaxed and real-capacity solutions.

    #### Agreement between independent methods:

    Finally, the stage ablation results show three structurally different components — ACO/RL search, Or-opt local search, TopUp/SwapSearch combinatorial insertion — converging on the same total value, with TopUp and SwapSearch adding nothing further once Or-opt is in the loop (these results are shown below in Extension C). That agreement between methods built on different principles (learned policy vs. deterministic local search vs. exhaustive small-neighbourhood search) is weak evidence on its own, but it points in a different direction than either of the first two techniques: it doesn't depend on scaling down the problem or relaxing a constraint, it depends on independently-derived answers not contradicting each other.

    #### What doesn't count as a measurement:

    Saying that the algorithm delivered a good result, on its own says nothing, as it doesn't clarify whether more could be reached within budget by another method. Every number above is a comparison, whether it be against a smaller exact solution, against a relaxed bound, or against an alternative. Each on their own isn't enough since the calibration sweep spans 16 different k values under two budgets, and the relaxed bound spans six k values under two budgets, making one convenient measurement unable to justify the whole picture.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SIDE MEMO S3-D
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Count the candidate solutions:

    At the full facility, k = 30, every subset of the 30 supply units is a candidate extraction set, and this is well before deciding how to group them into trips and before order them within a trip: 2^30 = 1.07 billion subsets. That count comes before any consideration of trip grouping or route ordering (permutations/combinations) and so is the floored value of search space size an exact method has, but isn't the full space.

    #### The count, not just the runtime, is the argument:

    My laptop measurements show k = 8 solving in about 0.001 seconds and k = 20 in 15.785 seconds. This represents a slowdown of roughly 5.0x per two additional units, and is steeper than pure 2^k as bundle formation adds its own k-dependent overhead on top. Extrapolating that growth rate (log-linear fit across all five measured points) to my full k = 30 instance gives roughly 17,898 days (almost 49 years) to solve which is clearly infeasible for anything approaching real world use. While the extrapolation is based off measured values up to k = 20, and so can't be taken with complete certainty, what remains true is the subset count itself — 2^30 candidate states is a fact about the problem, not an artifact of machine speed, and that's the part of the argument that survives "just buy a faster computer". A 100x faster machine only buys about 6–7 more supply units before it hits the same bottleneck, since the thing growing is the number of states that must be examined, and this isn't an implementation inefficiency.

    #### Name the structure:

    This is knapsack shaped. Choosing a subset of supplies under a budget ceiling, while maximising collected value, is 0/1 knapsack, for which the only known exact algorithms are exponential in item count (supplies in this case). On top of this, grouping chosen units into capacity limited trips is bin packing shaped, and choosing the order within each trip to minimise cost is a small tour construction problem, best done thorough permutations on this smaller set. Since this problem is NP-hard in every major aspect, this makes this an intractability argument rather than a performance complaint.

    #### Say what I gave up, and why that trade is justified here:

    Giving up the exact solver's guarantee of optimality is the direct price of a heuristic implementation approach for the benefit of termination in seconds rather than years/decades. However, it is only justifiable if the trade still results in good enough answers such that the algorithm can be trusted for good answers. This is done through the two independent checks above: a 32/32 exact match record against ground truth across k = 5 to 20, and a relaxed-capacity upper bound that the heuristic matched in half of the twelve tested cases, trailing by at most 5.41% under the mission budget and 7.69% under the tighter contingency (battery) budget. This trade of optimality for time complexity is especially valuable in the contingency plan scenario, where a draining battery in a drone requires a fast, good enough, answer instead of a confirmed best case answer that might arrive hours later or at a time where the problem no longer matters (years/decades later).
    """)
    return


@app.cell
def m35_header(mo):
    mo.md("""
    ---
    # [M3-5] Comparing the Time Complexities
    ### Criterion 7  -- 400-600 words

    Compare the time complexity of your **initial** (Memo 01/02) solution with
    your **improved** (Memo 03) solution.

    **Replace the values with those you have calculated.**

    Note that the two algorithms **do not solve the same problem**. Say so, and
    say why that makes a naive comparison misleading.
    """)
    return


@app.cell
def m35_side_by_side(
    BUDGET,
    CACHE_VERSION,
    SUPPLIES,
    VALUE,
    aco_rng_seed,
    load_or_compute,
    memo02_style_plan,
    mo,
    my_algorithm,
    plan_cost,
    plan_value,
    time,
):
    def _compute_m35():
        _rows = []
        for _name, _fn in [("Memo 01/02 approach", memo02_style_plan),
                           ("Improved (Memo 03)", my_algorithm)]:
            _t0 = time.time()
            _p = _fn(SUPPLIES, BUDGET)
            _ms = (time.time() - _t0) * 1000
            _rows.append({"name": _name, "value": plan_value(_p),
                           "units": sum(len(t) for t in _p), "trips": len(_p),
                           "cost": plan_cost(_p), "ms": _ms})
        return {"rows": _rows}

    _key = {"seed": aco_rng_seed, "budget": BUDGET}
    _m35 = load_or_compute("MEMO_3/cache_m35.json", CACHE_VERSION, _key, _compute_m35)

    _table_rows = "\n".join(
        f"| {r['name']} | {r['value']} of {sum(VALUE.values())} | "
        f"{r['units']} | {r['trips']} | {r['cost']:,.0f} | {r['ms']:,.2f} ms |"
        for r in _m35["rows"])

    mo.md(f"""
    ### Side-by-side analysis

    | Algorithm | Priority delivered | Units | Trips | Energy | Wall-clock |
    |---|---|---|---|---|---|
    {_table_rows}

    Add your **complexity class** for each in the response below, and identify
    the facility size at which exact optimisation ceased to be usable -- that is
    the single most important number in your comparison.
    """)
    return


@app.cell
def m35_input(SAVE_FILE_M03, json, mo, os):
    _saved = ""
    if os.path.exists(SAVE_FILE_M03):
        try:
            with open(SAVE_FILE_M03, "r") as _f:
                _d = json.load(_f)
            if _d:
                _saved = _d[-1].get("M35_comparison", "")
        except Exception:
            pass

    resp_m35 = mo.ui.text_area(
        label="**[M3-5] Comparing time complexities (400-600 words)**",
        value=_saved, rows=18, full_width=True,
        placeholder=(
            "BOTH BOUNDS\n"
            "  Initial: O(...). Improved: O(...). The parameter responsible is ...\n"
            "  The two do not solve the same problem, because ...\n\n"
            "DIRECTION OF THE CHANGE\n"
            "  My improved solution is slower. What the extra cost buys is ...\n\n"
            "MY MEASUREMENTS\n"
            "  Operation counts and timings from my own benchmarking cells ...\n"
            "  Measured vs predicted growth ...\n\n"
            "THE TRACTABILITY BOUNDARY\n"
            "  Exact optimisation was usable up to k = ... and became unusable at\n"
            "  k = ... because ...\n"
        )
    )
    resp_m35
    return (resp_m35,)


@app.cell
def _(mo, resp_m35):
    mo.callout(mo.md(resp_m35.value), kind="success")
    return


@app.cell
def m36_header(mo):
    mo.md("""
    ---
    # [M3-6] Comparing Coherence and Fitness for Purpose
    ### Criterion 10 -- 300-400 words
    """)
    return


@app.cell
def m36_trip_selector(mo, my_plan):
    trip_pick = mo.ui.multiselect(
        options=[str(i + 1) for i in range(len(my_plan))],
        value=[],
        label=("Show only these trips (leave empty to show all "
               f"{len(my_plan)}) -- useful when routes overlap")
    )
    trip_pick
    return (trip_pick,)


@app.cell
def m36_final_viz(
    BUDGET,
    SUPPLIES,
    VALUE,
    draw_facility,
    mo,
    my_plan,
    plan_cost,
    plan_value,
    seed_input,
    trip_pick,
):
    _taken = {u for t in my_plan for u in t}
    _abandoned = [u for u in SUPPLIES if u not in _taken]
    _lost = sum(VALUE[u] for u in _abandoned)
    _only = [int(x) for x in trip_pick.value] or None

    _fig = draw_facility(
        plan=my_plan,
        abandoned=_abandoned,
        show_labels=False,
        only_trips=_only,
        title=(f"Final extraction plan -- Seed {int(seed_input.value)} -- "
               f"{len(my_plan)} trips, {plan_value(my_plan)} priority delivered")
    )

    mo.vstack([
        _fig,
        mo.md(f"""
    | | |
    |---|---|
    | Trips flown | {len(my_plan)} |
    | Units recovered | {len(_taken)} of {len(SUPPLIES)} |
    | **Priority delivered** | **{plan_value(my_plan)}** of {sum(VALUE.values())} |
    | Priority abandoned | {_lost} |
    | Energy used | {plan_cost(my_plan):,.0f} of {BUDGET:,} |

    *Each trip is drawn in its own colour. Grey crosses are abandoned units.*
    """)
    ])
    return


@app.cell
def m36_input(SAVE_FILE_M03, json, mo, os):
    _saved = ""
    if os.path.exists(SAVE_FILE_M03):
        try:
            with open(SAVE_FILE_M03, "r") as _f:
                _d = json.load(_f)
            if _d:
                _saved = _d[-1].get("M36_coherence", "")
        except Exception:
            pass

    resp_m36 = mo.ui.text_area(
        label="**[M3-6] Comparing coherence and fitness for purpose (300-400 words)**",
        value=_saved, rows=16, full_width=True,
        placeholder=(
            "COHERENCE\n"
            "  Initial solution: how well did model and algorithm fit together?\n"
            "  Improved solution: where does the model store state the algorithm\n"
            "  genuinely needs? Any state added and never used, or recomputed\n"
            "  because the model does not hold it? Cite your own code.\n\n"
            "FITNESS FOR PURPOSE\n"
            "  Each solution against the directive as it stood at the time, then\n"
            "  against the final directive. What that reveals about my original\n"
            "  specification, and what I would model differently starting again.\n\n"
            "REAL-WORLD CONSEQUENCE\n"
            "  Flying the initial solution under the final conditions: which\n"
            "  supplies are lost, is the battery exhausted before extraction,\n"
            "  does CRUDY-1 make it out?"
        )
    )
    resp_m36
    return (resp_m36,)


@app.cell
def _(mo, resp_m36):
    mo.callout(mo.md(resp_m36.value), kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Further Analysis


    The sections below follow on from findings made while constructing [M3-1] through to [M3-6]. It includes a specific complexity bottleneck identified in [M3-3], a checkpoint comparison run outside this notebook, and open questions about how much of the final delivered value actually comes from the learned actor-critic ACO policy versus the deterministic search stages that follow it.

    Every section below reconstructs the pieces of the pipeline it needs locally, in its own helper cell, rather than modifying `my_algorithm` directly. Nothing above this point is used.

    Further, the below sections are precomputed and read saved json cache from the project directory in order to save runtime which collectively is in excess of 2 hours.
    """)
    return


@app.cell
def _(nx, random):
    # Your fac geneator scripts all put under a single overhead function to be used below for fac generations that aren't my seed

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

            shaft = (0, 0, 0)                                    # entry == extraction point
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
            """Cheapest corridor cost from u to v under the Memo 02 weights,
            carrying nothing. Multiply by (1 + L) to get the load-aware cost."""
            return DIST[u][v]

        def corridor_path(u, v):
            """The actual sector-by-sector route behind dist(u, v)."""
            return PATH[u][v]

        SHAFT = fac['shaft']
        EXITS = fac['exits']
        SUPPLIES = fac['supplies']
        MASS = {u: m for u, m in zip(fac['supplies'], fac['masses'])}
        VALUE = {u: v for u, v in zip(fac['supplies'], fac['values'])}
        CAP = fac['capacity']
        EXIT_LEG = min(dist(SHAFT, e) for e in EXITS)

        return dict(fac=fac, shaft=SHAFT, mass=MASS, value=VALUE, exits=EXITS, supplies=SUPPLIES, cap=CAP, exit_leg=EXIT_LEG, dist=DIST, path=PATH)

    return (fac_generator_seed_based,)


@app.cell
def _(itertools, os, random, time, torch, trainmod):
    # Calculator functions that are instance specific to the instance fed into script at initialisation/runtime
    # Allows the script to run on any input fac based from generation above
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

    # Memoize per-trip calculations
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

    # Actor-Critic Policy Loader
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

    # Validator
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

    # Start Pheromone at all points as 1.0
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

    # Pheromone Updater
    def ext_update_pheromone_elite(pheromone, ranked_plans, shaft, evaporation, elite_plan, elite_value, elitist_weight, min_pheromone=0.2, max_pheromone=12.0, inst=None):
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

    # Or-opt run for every ant during ACO
    def ext_or_opt_plan(plan, inst, max_segment=3, first_improvement=True):
        CAP = inst["CAP"]
        BUDGET = inst["BUDGET"]
        MASS = inst["MASS"]
        trip_cost = inst["trip_cost"]
        EXIT_LEG = inst.get("EXIT_LEG", 0.0)
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

    # Iteration Based ACO
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


    # Attempt to add uncollected supplies through a new trip, or insertion to existing trip when and if feasible
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

    # Swap Search - escape bad supply set by swapping expensive or low-value supplies with uncollected supplies
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

    # Multiple Start Hybrid ACO Based Search
    def ext_hybrid_solve(inst, actor, critic, seed, ants=32, iterations=20, evaporation=0.15,
                          elitist_weight=1.5, mode="sample", use_or_opt=True, use_topup=True,
                          use_final_topup=True, use_swap_search=True,
                          snapshot_cb=None, time_log=None):
        _start_time = time.time()
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

    return (
        build_inst,
        ext_aco_plan_cost,
        ext_aco_plan_value,
        ext_hybrid_solve,
        ext_load_policy_for_instance,
        ext_run_iterative_aco,
        ext_top_up_plan,
    )


@app.function
# As used everywhere else in the project, the below heuristic is used to find what it would need for complete collection and then takes 60% or 35% to define fac budget
# So is used to define budget for any new fac generated

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


@app.cell
def _():
    EXT_CACHE_VERSION = "v5"
    return (EXT_CACHE_VERSION,)


@app.cell
def _(
    build_inst,
    ext_aco_plan_cost,
    ext_load_policy_for_instance,
    fac_generator_seed_based,
):
    fac_seedd = 23092008
    fac_resultt = fac_generator_seed_based(fac_seedd)

    instt = build_inst(fac_resultt, fac_resultt["supplies"], budget=float('inf'))
    actorr, criticc, _ = ext_load_policy_for_instance(inst=instt, checkpoint_dir="MEMO_3/")

    full_plann = ext_exemplar_a_nearest_fill(fac_resultt["supplies"], budget=float('inf'), inst=instt)
    FULL_EXTRACTION_COSTT = ext_aco_plan_cost(full_plann, instt)
    BUDGETT = round(FULL_EXTRACTION_COSTT * 0.60)
    return fac_resultt, fac_seedd


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Extension A -- TopUp's O(k^6) Bound: Isolating the Cause, Testing the Fix

    [M3-3] traced TopUp's O(k^6) bottleneck to one specific design decision -- `remaining` is never capped to a constant pool in the same way `SwapSearch` intentionally does. Two things drive TopUp's cost independently:
    1. How much work is already committed to the plan when it starts (empty vs a genuinely adversarial "fragmented" start, seeded with real trips to scan against), and,
    2. Whether `remaining` itself is capped before combinations are generated.

    Earlier passes tested each variable one at a time while this section runs the full combination of both, first on the real facility up to its max true size (k=30), then it pushes past that limit using a fabricated instance to see how far the uncapped blow-up actually goes.
    """)
    return


@app.cell
def _(
    EXT_CACHE_VERSION,
    build_inst,
    ext_aco_plan_cost,
    ext_aco_plan_value,
    ext_top_up_plan,
    fac_resultt,
    fac_seedd,
    load_or_compute,
    time,
):
    def _fragmented_start(pool, inst):
        MASS = inst["MASS"]
        sorted_pool = sorted(pool, key=lambda u: MASS[u])
        seed_units = sorted_pool[0::2]
        return [[u] for u in seed_units]

    def _compute_ext_a_real():
        ks = [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
        rows = []
        for k in ks:
            row = {"k": k}
            pool_k = fac_resultt["supplies"][:k]

            inst_k = build_inst(fac_resultt, pool_k, budget=float('inf'))
            full_plan_k = ext_exemplar_a_nearest_fill(pool_k, budget=float('inf'), inst=inst_k)
            full_cost_k = ext_aco_plan_cost(full_plan_k, inst_k)
            budget_k = round(full_cost_k * 2)

            for start_name, start_fn in [
                ("empty", lambda p, inst: []),
                ("fragmented", _fragmented_start),
            ]:
                for cap_name, pool_size in [("uncapped", None), ("capped", 14)]:
                    inst = dict(inst_k)
                    inst["BUDGET"] = budget_k

                    start_plan = start_fn(pool_k, inst)
                    t0 = time.time()
                    result = ext_top_up_plan(start_plan, inst, max_insert_size=3, pool_size=pool_size)
                    elapsed_ms = round((time.time() - t0) * 1000, 2)
                    value = ext_aco_plan_value(result, inst)
                    row[f"{start_name}_{cap_name}_ms"] = elapsed_ms
                    row[f"{start_name}_{cap_name}_value"] = value

            rows.append(row)
        return {"rows": rows}

    _key = {"seed": fac_seedd, "ks": [6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
            "kind": "ext_a_real_2x2"}
    ext_a_real_result = load_or_compute("MEMO_3/cache_ext_a_real.json", EXT_CACHE_VERSION, _key, _compute_ext_a_real)
    return (ext_a_real_result,)


@app.cell
def _(ext_a_real_result, mo, np, plt):
    rows = ext_a_real_result["rows"]
    _ks = [r["k"] for r in rows]

    series = [
        ("empty_uncapped_ms", "Empty start, uncapped", '#7A1E2C', 'o-'),
        ("empty_capped_ms", "Empty start, capped", '#B85C6B', 'o--'),
        ("fragmented_uncapped_ms", "Fragmented start, uncapped", '#0B6E6B', 's-'),
        ("fragmented_capped_ms", "Fragmented start, capped", '#5FA8A0', 's--'),
    ]

    def _fit(ks, ts):
        ks_arr = np.array(ks, dtype=float)
        ts_arr = np.array([max(t, 1e-3) for t in ts])
        b, loga = np.polyfit(np.log(ks_arr), np.log(ts_arr), 1)
        return b

    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    exponents = {}
    for key, label, color, style in series:
        ts = [r[key] for r in rows]
        b = _fit(_ks, ts)
        exponents[label] = b
        _ax.plot(_ks, ts, style, color=color, label=f"{label} (fit k^{b:.2f})")

    _ax.set_yscale('log')
    _ax.set_xlabel('pool size k')
    _ax.set_ylabel('TopUp time (ms, log scale)')
    _ax.set_title('TopUp: start condition x cap, on the real facility', fontsize=10)
    _ax.legend(fontsize=8)
    _ax.grid(alpha=0.3, which='both')
    plt.tight_layout()

    tbl_rows = "\n".join(
        f"| {r['k']} | {r['empty_uncapped_ms']:,.2f} | {r['empty_capped_ms']:,.2f} | "
        f"{r['fragmented_uncapped_ms']:,.2f} | {r['fragmented_capped_ms']:,.2f} | "
        f"{r['fragmented_uncapped_value']} |"
        for r in rows
    )

    mo.vstack([
        _fig,
        mo.md(f"""
    All four lines run the exact same `ext_top_up_plan` on the exact same facility and pool at each k -- only the starting plan and the cap differ. Log-scale y-axis, so a genuinely steeper-than-polynomial line bends visibly upward rather than staying straight.

    | k | empty, uncapped (ms) | empty, capped (ms) | fragmented, uncapped (ms) | fragmented, capped (ms) | value (fragmented) |
    |:-:|:-:|:-:|:-:|:-:|:-:|
    {tbl_rows}

    Fitted exponents: {", ".join(f"**{label}** ~k^{b:.2f}" for label, b in exponents.items())}.

    Finding 1: capping controls the blow-up. Uncapped fits at k^3.45–4.06; capped fits at k^2.86–3.86, and the gap widens with k (46ms vs. 12ms at k=30 for empty-start). The uncapped lines visibly bend upward on the log-scale plot — the signature of the high-order growth [M3-3] flagged. This confirms capping is a real fix, not a marginal tweak.

    Finding 2: starting condition still matters, even capped. Fragmented-start isn't reliably worse uncapped (it's cheaper than empty-start at low-mid k, only overtakes near k=30). But capped, fragmented-start stays consistently ~3x costlier than empty-start throughout (35ms vs. 12ms at k=30) — some cost from a pre-committed plan survives the cap.

    Significance: Capping should be treated as required, not optional — uncapped growth is dangerous past k=30 and keeps climbing (confirmed separately out to k=200). But capping isn't a complete fix: callers starting TopUp from a fragmented plan should budget for a real latency premium even in the "safe," capped regime.
    """)
    ])
    return


@app.cell
def _(
    CAP,
    EXT_CACHE_VERSION,
    aco_rng_seed,
    ext_top_up_plan,
    itertools,
    load_or_compute,
    random,
    time,
):
    def _build_synthetic_inst(k, seed, cap):
        rng = random.Random(seed)
        shaft = "SHAFT"
        pool = [f"u{i}" for i in range(k)]
        pos = {shaft: (0.0, 0.0)}
        for u in pool:
            pos[u] = (rng.uniform(-20, 20), rng.uniform(-20, 20))
        mass = {u: rng.choice([1, 2, 3]) for u in pool}
        value = {u: rng.choice([1, 2, 3, 4, 5]) for u in pool}

        def dist(a, b):
            ax, ay = pos[a]
            bx, by = pos[b]
            return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

        def trip_cost(trip):
            here, total, load = shaft, 0.0, 0
            for u in trip:
                total += (1 + load) * dist(here, u)
                load += mass[u]
                here = u
            total += (1 + load) * dist(here, shaft)
            return total

        def plan_cost(plan):
            return sum(trip_cost(t) for t in plan)

        def plan_value(plan):
            return sum(value[u] for t in plan for u in t)

        best_order_cache = {}
        def best_order(units):
            key = frozenset(units)
            if key not in best_order_cache:
                if len(units) <= 1:
                    best_order_cache[key] = tuple(units)
                else:
                    best_order_cache[key] = tuple(min(itertools.permutations(units), key=trip_cost))
            return list(best_order_cache[key])

        return {
            "SHAFT": shaft, "SUPPLIES": pool, "MASS": mass, "VALUE": value,
            "CAP": cap, "BUDGET": float("inf"), "dist": dist,
            "trip_cost": trip_cost, "plan_cost": plan_cost,
            "plan_value": plan_value, "best_order": best_order,
        }

    def _fragmented_start(inst):
        sorted_pool = sorted(inst["SUPPLIES"], key=lambda u: inst["MASS"][u])
        return [[u] for u in sorted_pool[0::2]]

    def _compute_ext_a_synthetic():
        ks = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
        rows = []
        for k in ks:
            row = {"k": k}

            inst_empty = _build_synthetic_inst(k, seed=aco_rng_seed, cap=CAP)
            t0 = time.time()
            ext_top_up_plan([], inst_empty, max_insert_size=3, pool_size=None)
            row["empty_uncapped_ms"] = round((time.time() - t0) * 1000, 2)

            inst_frag_unc = _build_synthetic_inst(k, seed=aco_rng_seed, cap=CAP)
            frag_plan = _fragmented_start(inst_frag_unc)
            t0 = time.time()
            ext_top_up_plan(frag_plan, inst_frag_unc, max_insert_size=3, pool_size=None)
            row["fragmented_uncapped_ms"] = round((time.time() - t0) * 1000, 2)

            inst_frag_cap = _build_synthetic_inst(k, seed=aco_rng_seed, cap=CAP)
            frag_plan_cap = _fragmented_start(inst_frag_cap)
            t0 = time.time()
            ext_top_up_plan(frag_plan_cap, inst_frag_cap, max_insert_size=3, pool_size=14)
            row["fragmented_capped_ms"] = round((time.time() - t0) * 1000, 2)

            rows.append(row)
        return {"rows": rows}

    _key = {"seed": aco_rng_seed,
            "ks": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
            "kind": "ext_a_synthetic_v2"}
    ext_a_synthetic_result = load_or_compute("MEMO_3/cache_ext_a_synthetic.json", EXT_CACHE_VERSION, _key, _compute_ext_a_synthetic)
    return (ext_a_synthetic_result,)


@app.cell
def _(ext_a_synthetic_result, mo, np, plt):
    _rows = ext_a_synthetic_result["rows"]
    _ks = [r["k"] for r in _rows]
    empty_unc = [r["empty_uncapped_ms"] for r in _rows]
    frag_unc = [r["fragmented_uncapped_ms"] for r in _rows]
    frag_cap = [r["fragmented_capped_ms"] for r in _rows]

    def _fit(ks, ts):
        ks_arr = np.array(ks, dtype=float)
        ts_arr = np.array([max(t, 1e-3) for t in ts])
        b, loga = np.polyfit(np.log(ks_arr), np.log(ts_arr), 1)
        return b

    b_empty = _fit(_ks, empty_unc)
    b_frag = _fit(_ks, frag_unc)
    b_cap = _fit(_ks, frag_cap)

    _fig, _ax = plt.subplots(figsize=(7, 4.5))
    _ax.plot(_ks, empty_unc, 'o--', color='#7A1E2C', label=f'Empty, uncapped (fit k^{b_empty:.2f})')
    _ax.plot(_ks, frag_unc, 'o-', color='#0B6E6B', label=f'Fragmented, uncapped (fit k^{b_frag:.2f})')
    _ax.plot(_ks, frag_cap, 's-', color='#5FA8A0', label=f'Fragmented, capped (fit k^{b_cap:.2f})')
    _ax.set_yscale('log')
    _ax.set_xlabel('pool size k (synthetic)')
    _ax.set_ylabel('TopUp time (ms, log scale)')
    _ax.set_title('Pushing past k=30 with a fabricated instance', fontsize=10)
    _ax.legend(fontsize=8)
    _ax.grid(alpha=0.3, which='both')
    plt.tight_layout()

    k100 = next(r for r in _rows if r["k"] == 100)
    k200 = next(r for r in _rows if r["k"] == 200)
    ratio = k200["fragmented_uncapped_ms"] / k100["fragmented_uncapped_ms"]

    _tbl_rows = "\n".join(
        f"| {r['k']} | {r['empty_uncapped_ms']:,.2f} | {r['fragmented_uncapped_ms']:,.2f} | {r['fragmented_capped_ms']:,.2f} |"
        for r in _rows
    )

    mo.vstack([
        _fig,
        mo.md(f"""
    `ext_top_up_plan` doesn't touch anything facility-specific -- it only reads through the generic `inst` dictionary interface -- so a fabricated instance with random 2D positions standing in for corridor distances is a valid input, letting k run far past this facility's real ceiling of 30.

    | k | empty, uncapped (ms) | fragmented, uncapped (ms) | fragmented, capped (ms) |
    |---|---|---|---|
    {_tbl_rows}

    Finding 1: the uncapped blow-up doesn't plateau; it gets worse. Fits land at k^4.11 (empty) and k^4.73 (fragmented, uncapped) — steeper than the real-facility fits (k^3.45/k^4.06), confirming this isn't a small-k artifact. Doubling k from 100→200 takes fragmented-uncapped from 21.8s to 860.8s — a ~40x increase, well beyond what even k^4.73 alone predicts, and trending toward the O(k^6) range [M3-3] raised.

    Finding 2: the cap keeps working at scale. Fragmented-capped stays at k^3.57, and the gap from uncapped widens dramatically: at k=200, capped is 48s vs. uncapped's 861s — an 18x difference, up from roughly 1.4x at k=30 on the real facility. The cap isn't just a fix at realistic scale; its benefit compounds as k grows.

    Significance: This closes the open question from the real-facility test: capping isn't a partial or fragile mitigation — it's the difference between a function that's merely slow and one that's genuinely unsafe at scale. Any code path where pool size could grow unpredictably (batch processing, adversarial inputs, future facility expansions) needs the cap enforced by default, not left as caller-optional.
    """)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extension B -- Stressing the O(k^6) Bound at the Contingency Budget

    [M3-3] argued that `TopUp`'s real-world cost tracks the number of supplies still uncollected after ACO/SwapSearch have finished, not k directly -- and that this leftover count stays small under the mission budget (60% of full-collection cost), which is why measured runtime tracked so far below the theoretical O(k^6) worst case. The contingency budget (35%) is a deliberately harsher scenario built into this facility's own setup, and it is exactly the condition [M3-3] itself predicted would leave more supplies unresolved: less money to spend means ACO's restarts are more likely to run out of budget partway through a trip, leaving a bigger `remaining` set for `TopUp` to grind through afterwards.

    The cell below runs `my_algorithm` unmodified, the exact function used for [M3-1] onward -- once at the mission budget and once at the contingency budget, on this same facility and seed, and times both runs. This is the most direct possible test of [M3-3]'s own prediction: does the harsher budget actually leave more supplies uncollected, and does that translate into a measurably slower run?
    """)
    return


@app.cell
def _(
    BUDGET,
    BUDGET_RESERVE,
    CACHE_VERSION,
    SUPPLIES,
    aco_rng_seed,
    build_inst,
    ext_aco_plan_cost,
    ext_aco_plan_value,
    ext_hybrid_solve,
    ext_load_policy_for_instance,
    fac_generator_seed_based,
    load_or_compute,
    my_algorithm,
    plan_value,
    time,
    torch,
):
    def _compute_ext_b():
        _rows = []
        for _label, _bud in [("Mission (60%)", BUDGET), ("Contingency (35%)", BUDGET_RESERVE)]:
            _t0 = time.time()
            _p = my_algorithm(SUPPLIES, _bud)
            _ms = (time.time() - _t0) * 1000
            _collected = sum(len(t) for t in _p)
            _rows.append({
                "scenario": _label, "budget": _bud, "runtime_ms": round(_ms, 1),
                "value": plan_value(_p), "collected": _collected,
                "uncollected": len(SUPPLIES) - _collected,
            })
        return {"rows": _rows}


    def _compute_ext_b2(fac_result, inst, actor, critic):
        _rows = []
        for _label, _bud in [("Mission (60%)", _BUDGET), ("Contingency (35%)", _BUDGET_RESERVE)]:
            _scenario_inst = dict(inst)
            _scenario_inst["BUDGET"] = _bud

            _t0 = time.time()
            _plan, _source = ext_hybrid_solve(inst=_scenario_inst, actor=actor, critic=critic, seed=aco_rng_seed)
            _ms = (time.time() - _t0) * 1000

            _collected = sum(len(t) for t in _plan)
            _rows.append({
                "scenario": _label, "budget": _bud, "runtime_ms": round(_ms, 1),
                "value": ext_aco_plan_value(_plan, _scenario_inst), "collected": _collected,
                "uncollected": len(fac_result["supplies"]) - _collected,
            })
        return {"rows": _rows}

    _fac_seed = 18072008
    _fac_result = fac_generator_seed_based(_fac_seed)

    _inst = build_inst(_fac_result, _fac_result["supplies"], budget=float('inf'))
    torch.manual_seed(aco_rng_seed)
    _actor, _critic, _ = ext_load_policy_for_instance(inst=_inst, checkpoint_dir="MEMO_3/")

    _full_plan = ext_exemplar_a_nearest_fill(_fac_result["supplies"], budget=float('inf'), inst=_inst)
    _FULL_EXTRACTION_COST = ext_aco_plan_cost(_full_plan, _inst)
    _BUDGET = round(_FULL_EXTRACTION_COST * 0.60)
    _BUDGET_RESERVE = round(_FULL_EXTRACTION_COST * 0.35)

    _inst["BUDGET"] = _BUDGET

    _key = {"seed": aco_rng_seed, "budgets": [BUDGET, BUDGET_RESERVE]}
    ext_b_result = load_or_compute("MEMO_3/cache_ext_b.json", CACHE_VERSION, _key, _compute_ext_b)

    _key2 = {"seed": _fac_seed, "budgets": [_BUDGET, _BUDGET_RESERVE]}
    ext_b2_result = load_or_compute("MEMO_3/cache_ext_b2.json", CACHE_VERSION, _key2, lambda: _compute_ext_b2(_fac_result, _inst, _actor, _critic),)
    return ext_b2_result, ext_b_result


@app.cell
def _(ext_b2_result, ext_b_result, mo):
    def _build_table(rows):
        return "\n".join(
            f"| {r['scenario']} | {r['budget']:,} | {r['runtime_ms']:,.1f} | "
            f"{r['value']} | {r['collected']} | {r['uncollected']} |"
            for r in rows
        )

    _rows_b = ext_b_result["rows"]
    _rows_b2 = ext_b2_result["rows"]

    _tbl_b = _build_table(_rows_b)
    _tbl_b2 = _build_table(_rows_b2)

    mo.md(f"""
    ### B -- `my_algorithm` on the original facility

    Each row is one full run of `my_algorithm`, at the budget named in that
    row. Budget is the energy ceiling for that scenario;
    "collected"/"uncollected" is how many supply units ended up inside vs.
    outside the final plan.

    | Scenario | Budget | Runtime (ms) | Value delivered | Collected | Uncollected |
    |---|---|---|---|---|---|
    {_tbl_b}

    ### B2 -- full `ext_hybrid_solve` pipeline on the seeded facility (18072008)

    Each row is one full run of the real `ext_hybrid_solve` pipeline, at the
    budget named in that row, on this facility's own seed -- runtime is
    genuine wall-clock time for the entire hybrid pipeline (ACO restarts,
    TopUp, SwapSearch, TopUp again), not just the TopUp stage in isolation.

    | Scenario | Budget | Runtime (ms) | Value delivered | Collected | Uncollected |
    |---|---|---|---|---|---|
    {_tbl_b2}

    Finding: harsher budget consistently means more uncollected supplies, but not slower runtime. Both facilities show uncollected count roughly tripling at contingency vs. mission (B: 4 -> 11, B2: 6 -> 12), while runtime actually drops slightly under the tighter budget in both cases (B: 20.3s -> 17.5s, B2: 19.8s -> 16.6s). This replicates the pattern seen in the earlier single-facility test — not a fluke of that one run.

    Why runtime falls despite more leftover work: A tighter budget caps the plan smaller overall (fewer committed trips), which is cheaper for TopUp/SwapSearch to scan even though more supplies end up outside it. This is consistent with [M3-3]'s prediction that TopUp's cost tracks committed-plan size — just pulling in the opposite direction from "harsher budget" here, since committed size shrinks faster than uncollected count grows.

    Significance: "Harsher budget -> more leftover work" holds reliably across both facilities and algorithms; "harsher budget -> slower" does not — if anything the reverse holds here. So, a tighter budget scenario shouldn't be assumed to be the more expensive one to compute, since a smaller committed plan can more than offset the cost of a larger candidate pool.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extension C -- Pipeline Ablation: Learning vs. Search

    The 50-seed checkpoint comparison run outside this notebook found the 5k-instance and 40k-instance policies statistically indistinguishable in final delivered value, despite an 8x increase in training data (see Extension D for the formal test of that result). One natural explanation was that `TopUp`/`SwapSearch` do most of the actual optimisation regardless of which policy is guiding the ACO layer, masking whatever the extra training changed in the raw policy's behaviour.

    This tests that explanation directly, rather than just asserting it. `ext_hybrid_solve` (the local, extension-only copy of `hybrid_solve` built in the helper cell above) accepts 'stage' based boolean values that toggle individual section on and off, like the integrated `or-opt for ACO`, `SwapSearch` and `TopUp`. Running the various configurations at both the mission and contingency budgets shows precisely how much value each stage adds on top of the one before it -- if a later stage adds nothing, that is direct, load-bearing evidence about where the pipeline's quality is actually coming from, not an inference from indirect signals.
    """)
    return


@app.cell
def _(
    BUDGET,
    BUDGET_RESERVE,
    EXT_CACHE_VERSION,
    aco_rng_seed,
    build_inst,
    ext_aco_plan_cost,
    ext_aco_plan_value,
    ext_hybrid_solve,
    ext_load_policy_for_instance,
    fac_generator_seed_based,
    fac_resultt,
    load_or_compute,
):
    def _compute_ext_c():
        _stage_configs = [
            ("ACO/RL only", [False, False, False, False]),
            ("ACO/RL with Or-opt", [True, False, False, False]),
            ("+ TopUp", [True, True, False, False]),
            ("+ SwapSearch (no TopUp)", [True, True, False, True]),
            ("+ TopUp (full pipeline)", [True, True, True, True])
        ]
        _rows = []
        for _label, _bud in [("Mission", BUDGET), ("Contingency", BUDGET_RESERVE)]:
            _inst = build_inst(fac_resultt, fac_resultt["supplies"], _bud)   # <-- pool from fac_result, not global SUPPLIES
            _actor, _critic, _ = ext_load_policy_for_instance(inst=_inst, checkpoint_dir="MEMO_3/")
            _row = {"budget_label": _label}
            for _name, _stages in _stage_configs:
                _plan, _ = ext_hybrid_solve(inst=_inst, actor=_actor, critic=_critic, seed=aco_rng_seed,
                    use_or_opt=_stages[0], use_topup=_stages[1], use_final_topup=_stages[2], use_swap_search=_stages[3])
                _row[_name] = ext_aco_plan_value(_plan, _inst)
            _rows.append(_row)
        return {"rows": _rows, "stage_names": [n for n, _ in _stage_configs]}

    _key = {"seed": aco_rng_seed, "budgets": [BUDGET, BUDGET_RESERVE]}
    ext_c_result = load_or_compute("MEMO_3/cache_ext_c.json", EXT_CACHE_VERSION, _key, _compute_ext_c)

    def _compute_ext_c2(fac_result, inst, actor, critic):
        _stage_configs = [
            ("ACO/RL only", [False, False, False, False]),
            ("ACO/RL with Or-opt", [True, False, False, False]),
            ("+ TopUp", [True, True, False, False]),
            ("+ SwapSearch (no TopUp)", [True, True, False, True]),
            ("+ TopUp (full pipeline)", [True, True, True, True]),
        ]
        _rows = []
        for _label, _bud in [("Mission", _BUDGET), ("Contingency", _BUDGET_RESERVE)]:
            _scenario_inst = dict(inst)
            _scenario_inst["BUDGET"] = _bud

            _row = {"budget_label": _label}
            for _name, _stages in _stage_configs:
                _plan, _ = ext_hybrid_solve(inst=_scenario_inst, actor=actor, critic=critic, seed=aco_rng_seed, use_or_opt=_stages[0], use_topup=_stages[1],
                    use_final_topup=_stages[2], use_swap_search=_stages[3])

                _row[_name] = ext_aco_plan_value(_plan, _scenario_inst)
            _rows.append(_row)
        return {"rows": _rows, "stage_names": [n for n, _ in _stage_configs]}

    _fac_seed_c = 18072008
    _fac_result_c = fac_generator_seed_based(_fac_seed_c)

    _inst_c = build_inst(_fac_result_c, _fac_result_c["supplies"], budget=float('inf'))
    _actor_c, _critic_c, _ = ext_load_policy_for_instance(inst=_inst_c, checkpoint_dir="MEMO_3/")

    _full_plan_c = ext_exemplar_a_nearest_fill(_fac_result_c["supplies"], budget=float('inf'), inst=_inst_c)
    _FULL_EXTRACTION_COST_C = ext_aco_plan_cost(_full_plan_c, _inst_c)
    _BUDGET = round(_FULL_EXTRACTION_COST_C * 0.60)
    _BUDGET_RESERVE = round(_FULL_EXTRACTION_COST_C * 0.35)

    _inst_c["BUDGET"] = _BUDGET

    _key2 = {"seed": _fac_seed_c, "budgets": [_BUDGET, _BUDGET_RESERVE]}
    ext_c2_result = load_or_compute("MEMO_3/cache_ext_c2.json", EXT_CACHE_VERSION, _key2, lambda: _compute_ext_c2(_fac_result_c, _inst_c, _actor_c, _critic_c),)
    return ext_c2_result, ext_c_result


@app.cell
def _(ext_c2_result, ext_c_result, mo):
    def _build_stage_table(names, rows):
        _header = "| Budget | " + " | ".join(names) + " |"
        _sep = "|---|" + "---|" * len(names)
        _body = "\n".join(
            "| " + r["budget_label"] + " | " + " | ".join(str(r[n]) for n in names) + " |"
            for r in rows
        )
        return f"{_header}\n{_sep}\n{_body}"

    _names = ext_c_result["stage_names"]
    _rows_c = ext_c_result["rows"]
    _rows_c2 = ext_c2_result["rows"]

    _tbl_c = _build_stage_table(_names, _rows_c)
    _tbl_c2 = _build_stage_table(_names, _rows_c2)

    mo.md(f"""
    ### C -- stage ablation on the original facility

    Each column adds one more pipeline stage, so reading left to right on
    either row shows exactly what each stage contributed to that budget
    scenario.

    {_tbl_c}

    ### C2 -- stage ablation on the seeded facility (seed 18072008)

    Same ablation, run on the seeded facility from the B2 comparison, so
    this checks whether the pattern above is a property of this specific
    algorithm or an artifact of the original facility.

    {_tbl_c2}

    Finding: value plateaus almost immediately, and this is a genuine ablation. In C, Mission is flat at 91 across all five stages; Contingency moves once, 68 -> 69, then holds. In C2, Mission moves 70 -> 72 then holds; Contingency moves 53 -> 54 then holds. Since each stage is independently toggled, the entire measurable gain in every row happens at the first added stage (Or-opt); TopUp and SwapSearch contribute nothing further, on top of Or-opt, in any of the four budget×facility combinations tested.

    Significance: This is a genuinely striking result for the hybrid design: three multi-restart ACO passes (20/24/36 iterations, varying ant counts and evaporation) with in-loop Or-opt already exhaust nearly all the improvement available on both facilities — the deterministic TopUp/SwapSearch stages, despite their own non-trivial runtime cost (per Extension A/E), aren't finding meaningfully more value to add on top. That reframes their role: not as sources of final quality, but as safety nets that would matter more on facilities or budget regimes where the ACO's own search breadth leaves more on the table than it does here.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extension D -- 15K VS 40K Training Instance Checkpoint Comparison

    The 50-seed comparison between the 15k-instance and 40k-instance policy checkpoints was run outside this notebook using a standalone script that duplicates the canonical pipeline exactly. Therefore, for each seed, the only difference between the two runs is which `.pt` policy file is loaded. The resulting values are compared directly below on a per-seed basis. The two policies produce identical values for 42 of the 50 seeds, while the 40k policy is higher on 5 seeds and lower on 3. This shows that the additional 25k training instances do not produce a consistent improvement in the value reached by the ACO/RL pipeline. Instead, both policies are able to reach essentially the same solutions across the tested facility seeds, supporting the Extension C finding that the robustness of ACO/RL alone is sufficient to reach the facility's peak value without requiring `SwapSearch` or `TopUp`.

    *The 40k based checkpoint files are included in appendices folder.*
    """)
    return


@app.cell
def _(mo):
    # Raw per-seed value delivered:
    # (data point, value by 15k policy, value by 40k policy)
    # Data collated from external run

    _ext_d_data = [
        (1, 88, 88), (2, 80, 80), (3, 84, 84), (4, 65, 65), (5, 79, 79),
        (6, 79, 79), (7, 83, 83), (8, 88, 88), (9, 87, 87), (10, 83, 83),
        (11, 71, 70), (12, 66, 65), (13, 77, 77), (14, 74, 73), (15, 82, 82),
        (16, 76, 76), (17, 75, 75), (18, 83, 83), (19, 84, 84), (20, 84, 84),
        (21, 68, 69), (22, 80, 80), (23, 84, 84), (24, 86, 86), (25, 74, 74),
        (26, 68, 69), (27, 81, 81), (28, 79, 79), (29, 76, 76), (30, 80, 80),
        (31, 74, 74), (32, 81, 82), (33, 97, 97), (34, 79, 79), (35, 92, 92),
        (36, 82, 82), (37, 82, 82), (38, 84, 85), (39, 98, 98), (40, 83, 84),
        (41, 75, 75), (42, 77, 77), (43, 80, 80), (44, 73, 73), (45, 74, 74),
        (46, 79, 79), (47, 73, 73), (48, 75, 75), (49, 80, 80), (50, 81, 81),
    ]

    _improved = sum(b > a for _, a, b in _ext_d_data)
    _worse = sum(b < a for _, a, b in _ext_d_data)
    _equal = sum(b == a for _, a, b in _ext_d_data)

    _left_rows = []
    _right_rows = []

    for _seed, _v15, _v40 in _ext_d_data:
        if _v15 != _v40:
            _v15_html = (
                f'<span style="background-color:#5a2525; '
                f'padding:2px 6px; border-radius:4px;">{_v15}</span>'
            )
            _v40_html = (
                f'<span style="background-color:#254f35; '
                f'padding:2px 6px; border-radius:4px;">{_v40}</span>'
            )
        else:
            _v15_html = str(_v15)
            _v40_html = str(_v40)

        _row = f"""
        <tr>
            <td>{_seed}</td>
            <td>{_v15_html}</td>
            <td>{_v40_html}</td>
        </tr>
        """

        if _seed <= 25:
            _left_rows.append(_row)
        else:
            _right_rows.append(_row)


    _left_table = f"""
    <table style="width:100%; border-collapse:collapse;">
    <thead>
    <tr>
    <th>Data point</th>
    <th>15k policy</th>
    <th>40k policy</th>
    </tr>
    </thead>
    <tbody>
    {''.join(_left_rows)}
    </tbody>
    </table>
    """

    _right_table = f"""
    <table style="width:100%; border-collapse:collapse;">
    <thead>
    <tr>
    <th>Data point</th>
    <th>15k policy</th>
    <th>40k policy</th>
    </tr>
    </thead>
    <tbody>
    {''.join(_right_rows)}
    </tbody>
    </table>
    """

    mo.md(f"""
    ### Per-seed results

    <div style="
        display:grid;
        grid-template-columns:1fr 1fr;
        gap:24px;
        width:100%;
        overflow-x:auto;
    ">

    <div>

    {_left_table}

    </div>

    <div>

    {_right_table}

    </div>

    </div>

    **Comparison:** {_equal} / 50 data points are identical, 
    **{_improved}** are higher under the 40k policy, and **{_worse}** are lower.

    The comparison shows that increasing policy training from 15k to 40k instances has very little practical effect on the value reached by ACO/RL on this facility. The two policies produce exactly the same value on {_equal} of the 50 tested facility seeds. Where they do differ, the differences are small and occur in both directions: the 40k policy is higher on {_improved} seeds but lower on {_worse} seeds. This indicates that the additional 25k training instances do not produce a consistent improvement in the final solution quality.

    This also supports the finding from **Extension C** that ACO/RL alone is robust enough to reach the peak value point of the facility without requiring `SwapSearch` or `TopUp`. Since both the 15k and 40k policies independently reach essentially the same final values across the same facility seeds, the results suggest that the 15k ACO/RL search process itself is sufficiently robust to reach the facility's high-value solution regardless of which of these two policies is used to guide it. Therefore,
    the additional training represented by the 40k-instance policy does not appear necessary for achieving the observed peak value on this facility.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extension E -- Hyperparameter Sensitivity: Does `ants` Matter?

    The canonical pipeline uses `ants=32`, `iterations=20` for its first ACO restart config (the other two restarts bump ants/iterations further, see `hybrid_solve`'s `configs` list) without ever having tested whether those specific numbers matter. More ants per iteration means more candidate plans sampled and compared before the pheromone trail updates -- in principle that should mean a better-informed pheromone update and a better final answer, at the direct cost of more `construct_plan` calls (and therefore more wall-clock time) per iteration.

    This sweeps `ants` across 8, 16, 32, 48, and 64 -- holding every other setting (evaporation, elitist weight, iterations, seed) fixed -- and runs the full extension pipeline (`ext_hybrid_solve` with its default four stages) at each value, to see whether the current default of 32 ants is doing meaningful work, is already more than enough, or has room to shrink without losing quality.
    """)
    return


@app.cell
def _(
    EXT_CACHE_VERSION,
    aco_rng_seed,
    build_inst,
    ext_aco_plan_cost,
    ext_aco_plan_value,
    ext_hybrid_solve,
    ext_load_policy_for_instance,
    fac_generator_seed_based,
    load_or_compute,
    time,
):
    def _compute_ext_e():
        fac_result = fac_generator_seed_based(23092008)

        _inst = build_inst(fac_result, fac_result["supplies"], budget=float('inf'))
        _actor, _critic, _ = ext_load_policy_for_instance(_inst, "MEMO_3/")

        _full_plan = ext_exemplar_a_nearest_fill(fac_result["supplies"], budget=float('inf'), inst=_inst)
        _full_extraction_cost = ext_aco_plan_cost(_full_plan, _inst)
        _budget = round(_full_extraction_cost * 0.60)
        _inst["BUDGET"] = _budget

        _rows = []
        for _ants in [8, 16, 32, 48, 64]:
            _t0 = time.time()
            _plan, _source = ext_hybrid_solve(_inst, _actor, _critic, seed=aco_rng_seed, ants=_ants, iterations=20)
            _ms = (time.time() - _t0) * 1000
            _rows.append({
                "ants": _ants,
                "value": ext_aco_plan_value(_plan, _inst),
                "runtime_ms": round(_ms, 1),
            })
        return {"rows": _rows, "budget": _budget}

    _key = {"seed": 23092008, "ants_values": [8, 16, 32, 48, 64]}
    ext_e_result = load_or_compute("MEMO_3/cache_ext_e.json", EXT_CACHE_VERSION, _key, _compute_ext_e)
    return (ext_e_result,)


@app.cell
def _(ext_e_result, mo, plt):
    _rows = ext_e_result["rows"]
    _ants = [r["ants"] for r in _rows]
    _vals = [r["value"] for r in _rows]
    _times = [r["runtime_ms"] for r in _rows]

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 3.6))
    _axes[0].plot(_ants, _vals, marker='o', color='#0B6E6B')
    _axes[0].set_xlabel("ants per iteration")
    _axes[0].set_ylabel("priority value delivered")
    _axes[0].set_title("Value vs. ants", fontsize=10)
    _axes[0].grid(alpha=0.3)
    _axes[1].plot(_ants, _times, marker='s', color='#7A1E2C')
    _axes[1].set_xlabel("ants per iteration")
    _axes[1].set_ylabel("runtime (ms)")
    _axes[1].set_title("Runtime vs. ants", fontsize=10)
    _axes[1].grid(alpha=0.3)
    plt.tight_layout()

    _tbl = "\n".join(f"| {r['ants']} | {r['value']} | {r['runtime_ms']:,.1f} |" for r in _rows)
    mo.vstack([
        _fig,
        mo.md(f"""
    | ants | value delivered | runtime (ms) |
    |---|---|---|
    {_tbl}

    Finding: value is completely flat; runtime is not. Priority value stays pinned at 91 across the entire range, from 8 ants up to 64. Runtime, meanwhile, climbs steadily and non-linearly — roughly 17s at 8 ants, up to ~37.5s at 64 ants, with the steepest jump between 48 and 64.

    Significance: 8 ants per iteration already reaches the same value as 64 — a full 8x increase in ant count buys zero additional quality on this facility/budget, only more than double the runtime. Combined with the stage-ablation result (C/C2), this reinforces the same picture from a different angle: the ACO's search breadth saturates this instance's achievable ceiling very quickly, well before either more ants or more local-search stages have anything left to contribute. Practically, this argues for a much lower ant count as the efficient operating point here — 64 ants is pure overhead on this facility, not insurance. - More dicussed in H.
    """)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extension F -- Pheromone Field Evolution

    Pheromone is the shared memory ACO uses to bias future ants toward
    routes that worked well for past ants -- every edge starts at strength
    1.0, gets multiplied down by the evaporation rate every iteration, and
    gets a deposit added back proportional to how good the plans that used
    it turned out to be. Watching that field change over time is watching
    the search itself narrow in on a strategy, iteration by iteration.

    This animates the shaft-to-supply pheromone weight specifically (i.e.
    "how attractive does the policy currently think each unit is as a
    *first* stop"), across every single ACO iteration of a full
    `hybrid_solve` run -- all three restarts, back to back, so the field
    resets and re-grows three times over the course of the animation. Each
    supply unit is drawn at its real facility position (the same
    `xoff`/`yoff` transform `draw_facility` uses elsewhere in this
    notebook), with bubble size fixed to that unit's priority value and
    bubble colour showing its current pheromone strength on the direct
    shaft edge. The cell directly below runs a real instrumented
    `hybrid_solve` against this facility's own seed, with a snapshot hook
    (`snapshot_cb`) that captures the entire pheromone field once per
    iteration; the cell after that turns those snapshots into the animation
    and caches it as a `.gif`.
    """)
    return


@app.cell
def _(
    EXT_CACHE_VERSION,
    aco_rng_seed,
    animation,
    build_inst,
    ext_aco_plan_cost,
    ext_aco_plan_value,
    ext_hybrid_solve,
    ext_load_policy_for_instance,
    fac_generator_seed_based,
    load_or_compute,
    load_or_render_gif,
    np,
    plt,
    time,
):
    def _compute_viz_run():
        fac_result = fac_generator_seed_based(23092008)
        _inst = build_inst(fac_result, fac_result["supplies"], budget=float('inf'))
        _actor, _critic, _ = ext_load_policy_for_instance(_inst, "MEMO_3/")

        _full_plan = ext_exemplar_a_nearest_fill(fac_result["supplies"], budget=float('inf'), inst=_inst)
        _full_extraction_cost = ext_aco_plan_cost(_full_plan, _inst)
        _budget = round(_full_extraction_cost * 0.60)
        _inst["BUDGET"] = _budget

        _snapshots = []
        _time_log = []
        _final_pher_holder = {}

        def _snap(pheromone, best_value):
            _shaft = _inst["SHAFT"]
            _row = {str(u): pheromone.get((_shaft, u), 1.0) for u in _inst["SUPPLIES"]}
            _snapshots.append({"best_value": best_value, "shaft_pher": _row})
            _final_pher_holder["pher"] = pheromone

        _t0 = time.time()
        _plan, _source = ext_hybrid_solve(_inst, _actor, _critic, seed=aco_rng_seed,
                                           snapshot_cb=_snap, time_log=_time_log)
        _elapsed = time.time() - _t0
        _final_value = ext_aco_plan_value(_plan, _inst)

        _final_pher = _final_pher_holder.get("pher", {})
        _matrix = [
            [0.0 if u == v else _final_pher.get((u, v), 0.2) for v in _inst["SUPPLIES"]]
            for u in _inst["SUPPLIES"]
        ]

        return {
            "snapshots": _snapshots,
            "time_log": _time_log,
            "elapsed": _elapsed,
            "final_value": _final_value,
            "supply_list": [str(u) for u in _inst["SUPPLIES"]],
            "pheromone_matrix": _matrix,
            "budget": _budget,
        }

    _key = {"seed": 23092008, "kind": "viz_instrumented_v1"}
    viz_run = load_or_compute("MEMO_3/cache_viz_instrumented.json", EXT_CACHE_VERSION,
                               _key, _compute_viz_run)

    _fac_result_viz = fac_generator_seed_based(23092008)
    _fac = _fac_result_viz["fac"]
    _supplies = _fac_result_viz["supplies"]
    _value = _fac_result_viz["value"]

    _GAP = 3
    _wc = _fac["wing_cols"]

    def _xoff(w):
        return w * (_wc + _GAP)

    _xs = np.array([_xoff(u[0]) + u[1] + 0.5 for u in _supplies])
    _ys = np.array([u[2] + 0.5 for u in _supplies])
    _sizes = np.array([40 + 55 * _value[u] for u in _supplies])
    _snaps = viz_run["snapshots"]
    _sw, _sc_, _sr = _fac["shaft"]
    _shaft_x, _shaft_y = _xoff(_sw) + _sc_ + 0.5, _sr + 0.5

    def _render_ext_f_gif(_save_path):
        _fig, _ax = plt.subplots(figsize=(11, 5.5))
        _fig.patch.set_facecolor('#0b0f14')
        _ax.set_facecolor('#0b0f14')
        _ax.set_xlim(_xs.min() - 2, _xs.max() + 2)
        _ax.set_ylim(_ys.min() - 2, _ys.max() + 2)
        _ax.set_aspect('equal')
        _ax.axis('off')
        _ax.plot(_shaft_x, _shaft_y, marker='*', ms=18, color='#F59E0B', markeredgecolor='white',
                  markeredgewidth=0.6, zorder=5, label='Shaft (entry/extraction)')
        _scatter = _ax.scatter(_xs, _ys, s=_sizes, c=np.ones(len(_xs)), cmap="viridis",
                                vmin=0.2, vmax=6.0, edgecolors='#1a1f26', linewidths=0.4, zorder=4)
        _title = _ax.set_title("", color='#39ff88', fontsize=12)
        _ax.legend(loc='upper right', fontsize=8, facecolor='#0b0f14', edgecolor='#39424e', labelcolor='white')
        _cbar = _fig.colorbar(_scatter, ax=_ax, fraction=0.025, pad=0.02)
        _cbar.set_label('pheromone: shaft -> supply', color='white', fontsize=8)
        _cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(_cbar.ax.axes, 'yticklabels'), color='white')
        plt.tight_layout()

        def _update(frame):
            frame = min(frame, len(_snaps) - 1)
            _row = _snaps[frame]["shaft_pher"]
            _vals = np.array([_row[str(u)] for u in _supplies])
            _scatter.set_array(_vals)
            _title.set_text(
                f"Pheromone on shaft->supply edges  |  iteration "
                f"{frame + 1}/{len(_snaps)}  |  best value so far: "
                f"{_snaps[frame]['best_value']}"
            )
            return _scatter, _title

        _ani = animation.FuncAnimation(_fig, _update, frames=len(_snaps) + 20, interval=160, blit=False)
        _ani.save(_save_path, writer=animation.PillowWriter(fps=3))
        plt.close(_fig)

    ext_f_gif_path = load_or_render_gif(
        "MEMO_3/extras_assets/pheromone_evolution.gif", EXT_CACHE_VERSION,
        {"kind": "ext_f_pheromone_evolution", "n_iters": len(_snaps), "hold_frames": 20},
        _render_ext_f_gif,
    )
    return ext_f_gif_path, viz_run


@app.cell
def _(ext_f_gif_path, mo):
    mo.vstack([
        mo.image(src=ext_f_gif_path,
                  alt="Pheromone field evolution across all ACO iterations"),
        mo.md("""
    The title above the animation tracks the iteration number and the best plan value found so far, so quality improvement over time can be seen. The pheromone field resets twice more during the run -- at iteration 21 and again at 45 -- marking the start of each of the three ACO restarts. This allows best plan so far to continue between ACO restarts while allowing new random exploration to aid in escaping local best-cases. As can be seen, in this specific instance, the best value found (91) was already locked in by the second iteration, and so the vast majority of the remaining run time acted as purely safety checks to make sure nothing better could be found.
    """)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extension G -- Anytime Behaviour: Value vs. Wall-Clock Time

    An "anytime" algorithm is one that always has *some* valid answer ready,
    with quality improving the longer it's allowed to keep running --
    exactly the shape of `hybrid_solve`'s output, since `global_best` is
    tracked and updated continuously rather than only being produced right
    at the end. This section asks the practical question that shape
    implies: if the run had been cut off early -- deadline pressure, a
    battery running low, whatever the reason -- what value would have
    actually been on hand at that moment?

    It reuses the exact same instrumented run as Extension F (`viz_run`,
    computed once and shared between both sections), but reads a different
    field from it: `time_log`, a list of `(elapsed_seconds, best_value_so_far)`
    pairs recorded after every single ant in the run, not just at the end.
    Plotting that against wall-clock time (rather than iteration count, which
    Extension F's animation already covers) shows how quickly the search
    actually converges in real time, and how much of the total run is spent
    finding the answer versus how much is spent merely confirming nothing
    better exists. This is a static chart rather than an animation, so its
    computation and display live together in one cell.
    """)
    return


@app.cell
def _(mo, np, plt, viz_run):
    _log = viz_run["time_log"]
    _ts = np.array([row[0] for row in _log])
    _vs = np.array([row[1] for row in _log])

    _fig, _ax = plt.subplots(figsize=(7, 4.2))
    _ax.step(_ts, _vs, where='post', color='#0B6E6B', lw=1.6)
    _ax.fill_between(_ts, _vs, step='post', color='#0B6E6B', alpha=0.15)
    _ax.axhline(viz_run["final_value"], color='#7A1E2C', ls='--', lw=1,
                label=f"final value = {viz_run['final_value']}")
    _ax.set_xlabel("elapsed wall-clock time (s)")
    _ax.set_ylabel("best plan value found so far")
    _ax.set_title("Anytime behaviour -- this facility, this run", fontsize=10)
    _ax.legend(fontsize=8)
    _ax.grid(alpha=0.3)
    plt.tight_layout()

    _t_reach_final = next((t for (t, v) in _log if v >= viz_run["final_value"]),
                           viz_run["elapsed"])
    mo.vstack([
        _fig,
        mo.md(f"""
    The step line is the best plan value found *so far* at every recorded moment in time -- it can only ever go up or stay flat, never down, since `global_best` is only ever replaced by something strictly better. The shaded area under the line is just a visual aid to make the "already-locked-in" value at any point in time easier to read at a
    glance; the dashed horizontal line marks the run's actual final value for reference. Generated fresh from the same run as the Extension F animation above, so always accurate.

    This run reached its final value of **{viz_run['final_value']}** at **{_t_reach_final:.1f}s**, out of **{viz_run['elapsed']:.1f}s** total wall-clock time -- meaning roughly {100 * _t_reach_final / viz_run['elapsed']:.0f}% of the run's total time was spent finding the answer, and the remaining {100 * (1 - _t_reach_final / viz_run['elapsed']):.0f}% was spent on the later ACO restarts, top-up, and swap-search *confirming* nothing better exists, rather than on genuine search difficulty. That's a useful number in its own right: it says that if this run were deadline-bound and cut off at `{_t_reach_final:.0f}` second instead of being able to run to completion, the delivered value would have been identical.
    """)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Extension H -- Parameter Sensitivity: `ants` x Evaporation

    Isolates the ACO layer itself: a single restart of `run_iterative_aco`, 15 iterations, same facility and seed throughout (seed 23092008), swept over both headline ACO knobs at once -- `ants` and `evaporation` -- to see how they interact.

    Run as **two variants** over the identical grid, now that the extension copy of `run_iterative_aco` takes `use_or_opt`/`use_topup` toggles (added deliberately for this comparison; the graded `m31_student_algorithm` cell is untouched):

    - **Part A** -- `use_or_opt=False, use_topup=False`. Pure policy + pheromone, isolated from any local search.
    - **Part B** -- `use_or_opt=True, use_topup=True` (the defaults, matching the current canonical algorithm). or-opt then top-up (`max_insert_size=2`) run on every ant's plan before it enters the pool.

    Comparing A and B cell-by-cell isolates exactly what or-opt + per-ant top-up contributes to the ACO layer's own raw output, independent of hybrid_solve's *final* top-up/swap-search stages (unchanged, not part of this comparison).
    """)
    return


@app.cell
def _(
    EXT_CACHE_VERSION,
    build_inst,
    ext_aco_plan_cost,
    ext_aco_plan_value,
    ext_load_policy_for_instance,
    ext_run_iterative_aco,
    fac_generator_seed_based,
    load_or_compute,
    random,
    torch,
):
    _ANTS_GRID = [8, 16, 32, 48]
    _EVAP_GRID = [0.05, 0.10, 0.15, 0.22, 0.30]
    _SEED = 23092008
    _N_ITERATIONS = 15

    def _compute_ext_h():
        _fac_result = fac_generator_seed_based(_SEED)
        _base_inst = build_inst(_fac_result, _fac_result["supplies"], budget=float('inf'))
        _actor, _critic, _path = ext_load_policy_for_instance(_base_inst, "MEMO_3/")

        _full_plan = ext_exemplar_a_nearest_fill(_fac_result["supplies"], budget=float('inf'), inst=_base_inst)
        _full_extraction_cost = ext_aco_plan_cost(_full_plan, _base_inst)
        _budget = round(_full_extraction_cost * 0.60)
        _inst = dict(_base_inst)
        _inst["BUDGET"] = _budget

        def _run_grid(use_or_opt, use_topup):
            _rows = []
            for _ants in _ANTS_GRID:
                for _evap in _EVAP_GRID:
                    random.seed(_SEED)
                    torch.manual_seed(_SEED)
                    _plan = ext_run_iterative_aco(
                        _inst,
                        _actor,
                        _critic,
                        initial_best=[],
                        n_ants=_ants,
                        n_iterations=_N_ITERATIONS,
                        evaporation=_evap,
                        elitist_weight=1.5,
                        mode="sample",
                        seed=_SEED,
                        use_or_opt=use_or_opt,
                        use_topup=use_topup,
                    )
                    _value = ext_aco_plan_value(_plan, _inst)
                    _rows.append([_ants, _evap, _value])
            return _rows

        _rows_a = _run_grid(use_or_opt=False, use_topup=False)
        _rows_b = _run_grid(use_or_opt=True, use_topup=False)
        _rows_c = _run_grid(use_or_opt=True, use_topup=True)
        return {"rows_a": _rows_a, "rows_b": _rows_b, "rows_c": _rows_c, "budget": _budget}

    _key = {
        "seed": _SEED,
        "ants_grid": _ANTS_GRID,
        "evap_grid": _EVAP_GRID,
        "n_iterations": _N_ITERATIONS,
        "kind": "ext_h_reseeded_v1",
    }
    ext_h_result = load_or_compute("MEMO_3/cache_ext_h.json", EXT_CACHE_VERSION, _key, _compute_ext_h)
    return (ext_h_result,)


@app.cell
def _(ext_h_result, mo, np, plt):
    _ANTS_GRID = [8, 16, 32, 48]
    _EVAP_GRID = [0.05, 0.10, 0.15, 0.22, 0.30]

    _rows_a = ext_h_result["rows_a"]
    _rows_b = ext_h_result["rows_b"]
    _rows_c = ext_h_result["rows_c"]

    def _grid_of(_rows):
        return np.array([[v for (a, e, v) in _rows if a == ants] for ants in _ANTS_GRID])

    _grid_a = _grid_of(_rows_a)
    _grid_b = _grid_of(_rows_b)
    _grid_c = _grid_of(_rows_c)
    _grid_delta = _grid_b - _grid_a
    _grid_delta2 = _grid_c - _grid_a

    _fig, _axes = plt.subplots(2, 3, figsize=(14, 8.5))
    _axes = _axes.flatten()

    _panels = [
        (_axes[0], _grid_a, "Part A: use_or_opt=False, use_topup=False", "viridis", False),
        (_axes[1], _grid_b, "Part B: use_or_opt=True, use_topup=False", "viridis", False),
        (_axes[2], _grid_c, "Part C: use_or_opt=True, use_topup=True", "viridis", False),
        (_axes[3], _grid_delta, "Delta (B - A)", "RdYlGn", True),
        (_axes[4], _grid_delta2, "Delta (C - A)", "RdYlGn", True)
    ]

    for _ax, _grid, _title, _cmap, _is_delta in _panels:
        _im = _ax.imshow(_grid, cmap=_cmap, aspect="auto")
        _ax.set_xticks(range(len(_EVAP_GRID)))
        _ax.set_xticklabels([f"{e:.2f}" for e in _EVAP_GRID])
        _ax.set_yticks(range(len(_ANTS_GRID)))
        _ax.set_yticklabels(_ANTS_GRID)
        _ax.set_xlabel("evaporation")
        _ax.set_ylabel("ants")
        _ax.set_title(_title, fontsize=9)

        for _i in range(len(_ANTS_GRID)):
            for _j in range(len(_EVAP_GRID)):
                _val = _grid[_i, _j]
                _txt = f"{_val:+.0f}" if _is_delta else f"{int(_val)}"
                _ax.text(
                    _j, _i, _txt,
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8
                )
        _fig.colorbar(
            _im,
            ax=_ax,
            label="delta" if _is_delta else "value"
        )
    plt.tight_layout()

    _tbl_rows = []
    for (a1, e1, v1), (a2, e2, v2) in zip(_rows_a, _rows_b):
        _tbl_rows.append(
            f"| {a1} | {e1:.2f} | {v1} | {v2} | {v2 - v1:+d} |"
        )

    _tbl = "\n".join(_tbl_rows)

    mo.vstack(
        [
            _fig,
            mo.md(
                f"""
    | ants | evaporation | value (A) | value (B) | value (C) | B − A | C − A |
    |---:|---:|---:|---:|---:|---:|---:|
    {chr(10).join(
        f"| {a:.0f} | {e:.2f} | {va:.0f} | {vb:.0f} | {vc:.0f} | {vb-va:+.0f} | {vc-va:+.0f} |"
        for (a, e, va), (_, _, vb), (_, _, vc)
        in zip(_rows_a, _rows_b, _rows_c)
    )}

    ### Experimental conditions

    **Part A** — `use_or_opt=False`, `use_topup=False` Baseline ACO configuration with no per-ant Or-opt or top-up.

    **Part B** — `use_or_opt=True`, `use_topup=False` Adds Or-opt to each ant's plan before it enters the pool, while leaving top-up disabled.

    **Part C** — `use_or_opt=True`, `use_topup=True` Adds both Or-opt and top-up (`max_insert_size=2`) to each ant's plan before it enters the pool.

    ### Interpretation

    The **B − A** delta isolates the contribution of Or-opt relative to the baseline. The **C − A** delta shows the combined contribution of Or-opt and per-ant top-up relative to the baseline. These measurements concern the **ACO layer only**. They are independent of the final top-up and swap-search stages performed by `hybrid_solve`, which remain unchanged across the comparison.

    As can be seen in the charts, using only 8 ants in the first ACO run generally produces lower-valued solutions than using 32 or 48 ants when Or-opt and top-up are disabled. While the maximum observed value is achievable with 48 ants, the additional ant population increases the computational overhead and therefore runtime. With Or-opt enabled, 32 ants is able to reach the same maximum observed value as 48 ants, while the smaller 8-ant population still produces lower-valued solutions. When top-up is also enabled, all tested ant populations reach the same maximum observed value. However, top-up is an expensive operation to run, with a complexity of O(k^6), so applying it across larger ant populations increases computational cost. Therefore, 32 ants provides a practical balance between solution quality and computation time, achieving the maximum observed value with Or-opt while avoiding the additional overhead of a 48 ant population.
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extension I -- What Did the Trained Policy Actually Learn?

    Everything so far has judged the pipeline by its *outputs* (final plan value, runtime, gaps against the exact solver). This section instead looks directly at the actor network's own internal preferences, with every other part of the pipeline switched off -- no pheromone bias accumulated yet, no ACO restarts, no top-up or swap-search polishing anything afterward. Concretely: standing at the shaft, with an empty trip, zero budget spent, and every pheromone entry at its neutral starting value of 1.0, the standalone `rl_firstmove.py` script builds the same seven-feature vector `BuildFeatures` would build for every one of the facility's 30 supply nodes, feeds all 30 through the trained actor in one batch, and reads off the resulting softmax probability for each -- i.e. exactly what the policy would pick as its very first move, with nothing else influencing the decision.

    This matters because it is the cleanest possible test of what the network actually learned to value, as opposed to what the full pipeline ends up delivering (which is also shaped by ACO's search and local search's polishing, and so can't cleanly separate "the policy's own judgement" from "everything else in the pipeline compensating for it").
    """)
    return


@app.cell
def _(mo, np, plt):
    # (node, value, mass, P(pick)) from rl_firstmove.py -- actor's own
    # softmax over all 30 supply nodes, shaft=(0,0,0), neutral pheromone,
    # seed 23092008.
    _rows = [
        ((0, 5, 2), 5, 1, 0.374558), ((1, 0, 0), 5, 1, 0.296154),
        ((0, 2, 1), 4, 1, 0.274712), ((0, 4, 6), 4, 1, 0.01114),
        ((1, 6, 8), 5, 1, 0.010076), ((0, 0, 1), 3, 3, 0.009379),
        ((0, 1, 5), 2, 3, 0.005393), ((0, 5, 5), 3, 1, 0.002958),
        ((0, 8, 2), 1, 1, 0.002788), ((0, 7, 6), 3, 2, 0.00216),
        ((0, 7, 2), 1, 1, 0.001799), ((0, 1, 7), 4, 2, 0.001769),
        ((0, 6, 0), 3, 3, 0.00173), ((0, 8, 9), 5, 3, 0.001414),
        ((0, 0, 5), 3, 2, 0.000913), ((1, 0, 9), 3, 3, 0.000881),
        ((0, 5, 8), 5, 3, 0.000686), ((1, 8, 5), 4, 1, 0.000525),
        ((0, 2, 6), 5, 3, 0.000324), ((0, 6, 5), 3, 3, 0.000246),
        ((0, 5, 6), 2, 3, 0.000178), ((1, 1, 7), 5, 1, 6e-05),
        ((1, 4, 8), 2, 3, 3.8e-05), ((1, 7, 6), 2, 1, 3.5e-05),
        ((1, 9, 2), 4, 2, 3.1e-05), ((1, 1, 5), 4, 1, 3.1e-05),
        ((1, 2, 2), 3, 1, 1.3e-05), ((1, 5, 3), 3, 2, 5e-06),
        ((1, 5, 2), 2, 2, 3e-06), ((1, 7, 1), 4, 3, 3e-06),
    ]


    _density = np.array([v / m for (n, v, m, p) in _rows])
    _prob = np.array([p for (n, v, m, p) in _rows])
    _size = 40 + 900 * _prob

    _fig, _ax = plt.subplots(figsize=(7, 5))
    _sc = _ax.scatter(_density, _prob, s=_size, c=_density, cmap="viridis",
                       alpha=0.85, edgecolors="#0b0f14", linewidths=0.5)
    _ax.set_yscale("log")
    _ax.set_xlabel("value / mass (density)")
    _ax.set_ylabel("P(pick) under actor's softmax (log scale)")
    _ax.set_title("Actor's raw first-move preference, all 30 supply nodes", fontsize=10)
    _ax.grid(alpha=0.3)
    _fig.colorbar(_sc, ax=_ax, label="value / mass")

    for (n, v, m, p) in sorted(_rows, key=lambda r: -r[3])[:2]:
        _ax.annotate(f"{n}\nv={v} m={m}", (v / m, p), textcoords="offset points",
                     xytext=(8, 8), fontsize=8)
    plt.tight_layout()

    _top5 = sorted(_rows, key=lambda r: -r[3])[:5]
    _tbl = "\n".join(f"| {n} | {v} | {m} | {p:.4g} |" for (n, v, m, p) in _top5)
    mo.vstack([
        _fig,
        mo.md(f"""
    Top 5 by policy preference:

    | node | value | mass | P(pick) |
    |---|---|---|---|
    {_tbl}

    Each point on the scatter is one of the facility's 30 supply nodes. The x-axis is that node's value-per-mass density (the same metric Exemplar C's nearest-fill-by-density rule uses); the y-axis is the probability the trained actor's own softmax assigns to picking that node first, on a log scale since the probabilities span several orders
    of magnitude. Bubble size repeats the y-axis visually (bigger bubble = higher P(pick)); bubble colour repeats the x-axis (value/mass) so both dimensions are readable even without carefully reading the axes. The table above the chart lists the five nodes the policy actually favours most, by raw probability.

    The two largest bubbles land on two of the highest value-density nodes in the whole facility -- node `(1,0,0)` (value 5, mass 1) and node `(0,5,2)` (also value 5, mass 1) -- not simply the nodes nearest to the shaft (distance doesn't even appear as an axis here, since at this "empty trip from the shaft" moment the feature vector's distance term
    is the same relative signal for every candidate, and the policy is clearly not using it as the dominant factor). That's genuine, direct evidence the actor is reasoning about value-per-mass density as its primary signal, not falling back on distance-greedy proximity -- which is exactly the failure mode Exemplar A (nearest-fill) has, and exactly
    the kind of joint, multi-factor reasoning [M3-1] argued a fixed threshold rule couldn't represent.
    """)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extension J -- Runtime Curve Fit

    The cell above ([M3-3]) already fits a power law inline via
    `np.polyfit` on log-transformed data -- a *linear* least-squares fit
    in log-space, with no R^2 reported. `generate_visuals.py`, run
    outside this notebook against this facility's own `cache_m33.json`,
    instead fits `t = a * k^b` directly via `scipy.optimize.curve_fit`
    -- a genuinely different nonlinear fit in the original (non-log)
    space -- and reports a real R^2. Both are shown below for
    comparison.
    """)
    return


@app.cell
def _(ks, mo, np, plt, times):
    # Fit parameters from generate_visuals.py's scipy.optimize.curve_fit,
    # run outside this notebook against this facility's own cache_m33.json.
    _ext_j_result = {
        "a_fit": 433.337228,
        "b_fit": 1.104739,
        "r_squared": 0.981195,
        "a_loglinear": 356.242657,
        "b_loglinear": 1.168539,
    }

    _k_arr = np.array(ks, dtype=float)
    _k_smooth = np.linspace(_k_arr.min(), _k_arr.max(), 200)
    _power_smooth = _ext_j_result["a_fit"] * _k_smooth ** _ext_j_result["b_fit"]
    _loglinear_smooth = _ext_j_result["a_loglinear"] * _k_smooth ** _ext_j_result["b_loglinear"]

    ext_j_result = dict(_ext_j_result, k_smooth=_k_smooth,
                         power_smooth=_power_smooth, loglinear_smooth=_loglinear_smooth)

    _fig, _ax = plt.subplots(figsize=(7, 5))
    _ax.plot(ks, times, 'o', color='#0B6E6B', label='Measured (this facility)')
    _ax.plot(ext_j_result["k_smooth"], ext_j_result["power_smooth"], '-', color='#F59E0B',
              label=f'scipy curve_fit: t={ext_j_result["a_fit"]:.1f}·k^{ext_j_result["b_fit"]:.3f} '
                    f'(R²={ext_j_result["r_squared"]:.4f})')
    _ax.plot(ext_j_result["k_smooth"], ext_j_result["loglinear_smooth"], '--', color='#7A1E2C',
              label=f'np.polyfit (log-log): t={ext_j_result["a_loglinear"]:.1f}·k^{ext_j_result["b_loglinear"]:.3f}')
    _ax.set_xlabel("supply units k")
    _ax.set_ylabel("run time (ms)")
    _ax.set_title("my_algorithm runtime: power-law fit", fontsize=10)
    _ax.legend()
    _ax.grid(alpha=0.3)
    plt.tight_layout()

    mo.vstack([
        _fig,
        mo.md(f"""
    | fit method | a | b | R² |
    |---|---|---|---|
    | scipy.optimize.curve_fit (nonlinear, original space) | {ext_j_result['a_fit']:.2f} | {ext_j_result['b_fit']:.4f} | {ext_j_result['r_squared']:.4f} |
    | np.polyfit (linear, log-log space) | {ext_j_result['a_loglinear']:.2f} | {ext_j_result['b_loglinear']:.4f} | -- |

    Both methods agree closely (b = 1.10-1.17), and the nonlinear fit explains 98.1% of the variance in the measured times. An exponent just above 1 means `my_algorithm`'s runtime grows close to *linearly* with the number of supply units k, not exponentially -- consistent with its structure: `or_opt_plan`, `top_up_plan`, and `swap_search` all operate over fixed, bounded neighbourhoods (segment length <= 3, insertion combos <= 3, a capped candidate pool of 14) that don't grow with k, so the only genuinely k-dependent cost is the roughly linear number of candidates each ACO step and each bounded local-search pass has to consider. This is a meaningfully different growth profile from the *exact* solver used as a ground truth in Extension K, which is provably exponential in the worst case (O(2^k) over the subset space) -- the two shouldn't be conflated, since they're different algorithms answering different questions.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extension K -- Multi-Seed Generalisation of the Trained Policy

    Every other extension in this section uses this facility's own single seed. That leaves one obvious, important question unanswered: did the policy learn something genuinely transferable about this *class* of facility, or did it simply memorise good responses to the one specific layout it happens to have been evaluated on throughout this notebook? This section is the direct test of that.

    This sweep is restricted to 10 freshly drawn seeds for every fac size (2/3/4 wings) bucket the policy was actually trained on. For each of the 10, the standalone `sweep_multiseed.py` script (run outside this notebook) computes two things: the full k=30 hybrid solve's value, cost, and runtime; and, separately, a smaller k=15 sub-problem (the top 15 supplies by value/mass density, with the budget scaled down proportionally to 15/30) checked directly against the exact DP solver, which is only
    tractable at this reduced size. That second, smaller check is what makes this a genuine optimality-gap reading rather than just "the policy produced *some* plan on unseen data" -- it tells exactly how far from provably-optimal the policy's decisions actually are on facilities it has never encountered.
    """)
    return


@app.cell
def _(mo, np, plt):
    # From sweep_multiseed.py, run outside this notebook:
    # seed, full-problem value/cost/time, and k=15 exact-vs-hybrid
    # optimality check for the n_wings=2, 3, and 4 policies.

    _rows_wings2 = [
        (23092008, 91, 4833.0, 13.4419, 50, 50, 0.0),
        (330754347, 86, 3672.0, 18.5259, 49, 49, 0.0),
        (224306871, 67, 5085.0, 17.6388, 46, 46, 0.0),
        (316298898, 77, 4999.0, 15.4829, 48, 48, 0.0),
        (747868290, 83, 4861.0, 13.0964, 46, 46, 0.0),
        (900101838, 83, 4387.0, 13.5257, 47, 47, 0.0),
        (816728232, 68, 4401.0, 13.1244, 44, 42, 4.5),
        (298280592, 92, 3961.0, 14.1642, 50, 50, 0.0),
        (382451373, 78, 4887.0, 18.5473, 43, 43, 0.0),
        (622713612, 86, 4095.0, 13.2425, 52, 52, 0.0),
    ]

    _rows_wings3 = [
        (779946154, 78, 3666.0, 19.3160, 48, 48, 0.0),
        (294112564, 77, 9685.0, 18.9622, 42, 42, 0.0),
        (787041370, 91, 5062.0, 13.1398, 49, 49, 0.0),
        (893815447, 80, 4920.0, 12.3955, 49, 49, 0.0),
        (26643082, 77, 9632.0, 10.4843, 51, 51, 0.0),
        (206634832, 76, 5250.0, 11.7826, 46, 46, 0.0),
        (347208553, 70, 4942.0, 10.4760, 46, 46, 0.0),
        (446348869, 79, 4774.0, 10.8595, 45, 45, 0.0),
        (51429685, 91, 5685.0, 11.0301, 53, 53, 0.0),
        (560742958, 88, 7155.0, 10.7015, 52, 52, 0.0),
    ]

    _rows_wings4 = [
        (470516294, 81, 7562.0, 12.3766, 49, 49, 0.0),
        (719311616, 87, 7910.0, 12.8351, 54, 54, 0.0),
        (734562590, 80, 11420.0, 12.5247, 48, 48, 0.0),
        (859933205, 83, 10127.0, 11.6939, 44, 44, 0.0),
        (554222819, 81, 13921.0, 11.2333, 46, 46, 0.0),
        (287876738, 85, 7244.0, 12.1322, 47, 47, 0.0),
        (982650806, 80, 10172.0, 12.1314, 48, 48, 0.0),
        (269243285, 79, 5249.0, 13.8116, 41, 41, 0.0),
        (664407374, 69, 6930.0, 11.2469, 42, 42, 0.0),
        (267722117, 87, 8389.0, 11.8909, 53, 53, 0.0),
    ]

    def _make_policy_panel(_ax_value, _ax_gap, _rows, _n_wings, _highlight_seed):
        _seeds = [str(r[0]) for r in _rows]
        _vals = [r[1] for r in _rows]
        _gaps = [r[6] for r in _rows]

        _colors = [
            '#F59E0B' if str(r[0]) == _highlight_seed else '#0B6E6B'
            for r in _rows
        ]

        _ax_value.bar(_seeds, _vals, color=_colors)
        _ax_value.set_ylabel("full-problem priority value")
        _ax_value.set_title(
            f"n_wings={_n_wings}: value delivered",
            fontsize=10
        )
        _ax_value.tick_params(axis='x', rotation=60)
        _ax_value.grid(alpha=0.3, axis='y')

        _ax_gap.bar(_seeds, _gaps, color='#7A1E2C')
        _ax_gap.set_ylabel("k=15 optimality gap (%)")
        _ax_gap.set_title(
            f"n_wings={_n_wings}: hybrid vs. exact k=15",
            fontsize=10
        )
        _ax_gap.set_ylim(-1, 5)
        _ax_gap.tick_params(axis='x', rotation=60)
        _ax_gap.grid(alpha=0.3, axis='y')

    _fig, _axes = plt.subplots(
        3, 2,
        figsize=(13, 11)
    )

    _make_policy_panel(
        _axes[0, 0],
        _axes[0, 1],
        _rows_wings2,
        2,
        "23092008"
    )

    _make_policy_panel(
        _axes[1, 0],
        _axes[1, 1],
        _rows_wings3,
        3,
        None
    )

    _make_policy_panel(
        _axes[2, 0],
        _axes[2, 1],
        _rows_wings4,
        4,
        None
    )

    plt.tight_layout()

    _all_rows = (
        _rows_wings2 +
        _rows_wings3 +
        _rows_wings4
    )

    _mean_val_w2 = np.mean([r[1] for r in _rows_wings2])
    _mean_val_w3 = np.mean([r[1] for r in _rows_wings3])
    _mean_val_w4 = np.mean([r[1] for r in _rows_wings4])

    _mean_val_all = np.mean([r[1] for r in _all_rows])

    _gaps_all = [r[6] for r in _all_rows]
    _zero_gap_count = sum(g == 0.0 for g in _gaps_all)
    _total_gap_count = len(_gaps_all)

    _max_gap = max(_gaps_all)

    def _make_table(_rows):
        return "\n".join(
            f"| {r[0]}"
            f"{' (this facility)' if r[0] == 23092008 else ''} | "
            f"{r[1]} | {r[2]:.0f} | {r[3]:.1f}s | "
            f"{r[4]} | {r[5]} | {r[6]:.1f}% |"
            for r in _rows
        )

    _tbl_w2 = _make_table(_rows_wings2)
    _tbl_w3 = _make_table(_rows_wings3)
    _tbl_w4 = _make_table(_rows_wings4)

    mo.vstack([
        _fig,

        mo.md(f"""
    ### n_wings=2

    | seed | full value | full cost | time | k=15 exact | k=15 hybrid | gap |
    |---|---:|---:|---:|---:|---:|---:|
    {_tbl_w2}

    **Mean full-problem value:** {_mean_val_w2:.1f}  
    **Value range:** {min(r[1] for r in _rows_wings2)}–{max(r[1] for r in _rows_wings2)}


    """),

        mo.md(f"""
    ### n_wings=3

    | seed | full value | full cost | time | k=15 exact | k=15 hybrid | gap |
    |---|---:|---:|---:|---:|---:|---:|
    {_tbl_w3}

    **Mean full-problem value:** {_mean_val_w3:.1f}  
    **Value range:** {min(r[1] for r in _rows_wings3)}–{max(r[1] for r in _rows_wings3)}


    """),

        mo.md(f"""
    ### n_wings=4

    | seed | full value | full cost | time | k=15 exact | k=15 hybrid | gap |
    |---|---:|---:|---:|---:|---:|---:|
    {_tbl_w4}

    **Mean full-problem value:** {_mean_val_w4:.1f}  
    **Value range:** {min(r[1] for r in _rows_wings4)}–{max(r[1] for r in _rows_wings4)}


    """),

        mo.md(f"""
    ### Cross-policy summary

    Across the **30 independently generated facilities**:

    - **Mean full-problem value:** {_mean_val_all:.1f}
    - **n_wings=2 mean:** {_mean_val_w2:.1f}
    - **n_wings=3 mean:** {_mean_val_w3:.1f}
    - **n_wings=4 mean:** {_mean_val_w4:.1f}
    - **k=15 exact matches:** {_zero_gap_count}/{_total_gap_count}
    - **Maximum observed k=15 gap:** {_max_gap:.1f}%

    The three panels separate the results by policy checkpoint: `n_wings=2`, `n_wings=3`, and `n_wings=4`. Each row represents a different random facility generated from its seed, rather than a relabelling of the same maze.

    For `n_wings=2`, seed `23092008` is highlighted in amber because it is the facility used elsewhere in the notebook; the other nine facilities are independently generated test cases. The `n_wings=3` and `n_wings=4` datasets contain ten independently generated facilities each.

    The k=15 column compares the hybrid pipeline against the exact solver on the smaller k=15 sub-problem. A gap of `0.0%` means the hybrid returned the same value as the exact solution. The one non-zero result in the n_wings=2 dataset is seed `816728232`, where the exact solution collected value `44` and the hybrid collected `42`, corresponding to the recorded `4.5%` gap.

    The full-problem results are separate from this k=15 verification: the exact solver is not being used to establish optimality of the larger full problems. The k=15 experiment therefore provides an optimality check of the hybrid's behaviour on a tractable sub-problem, while the full-problem columns report the hybrid's scalable performance on the complete facilities.
    """)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extension L -- Route Replay: Watching CRUDY-1 Work

    Every other section in this notebook describes `my_plan` in numbers: total value, total cost, trip count. This section replaces the numbers with the literal thing those numbers describe -- CRUDY-1 physically moving through the facility. It replays the actual submitted `my_plan` one shuttle trip at a time on the real facility map (the same `draw_facility` renderer used throughout the rest of this notebook, so the corridors, wings, and junctions are drawn exactly as they are everywhere else), turning the abstract idea of "a plan" into a literal walkthrough of what CRUDY-1 actually does on each trip: descend from the shaft, collect supply units up to the `CAP` mass limit, return to the shaft, deposit, and descend again for the next trip, until the plan is exhausted and the final exit is taken.

    Two generation cells sit below this one: the first renders one complete facility-map frame per trip (so trip 5's frame shows trips 1 through 5 all drawn in, cumulatively); the second turns that sequence of frames into an animated GIF and caches it. Rendering a full facility map per trip is the slowest generation step in this whole extension section -- it only needs to run once, though, since the result is cached exactly like every other GIF here.
    """)
    return


@app.cell
def _(draw_facility, my_plan, np, plan_cost, plan_value, plt):
    route_frames = []
    for _i in range(1, len(my_plan) + 1):
        _partial = my_plan[:_i]
        _running_cost = plan_cost(_partial)
        _running_value = plan_value(_partial)
        _fig = draw_facility(
            plan=my_plan, only_trips=list(range(1, _i + 1)),
            title=(f"Trip {_i}/{len(my_plan)} -- value so far {_running_value}, "
                   f"budget spent {_running_cost:,.0f}")
        )
        _fig.canvas.draw()
        _buf = np.asarray(_fig.canvas.buffer_rgba()).copy()
        route_frames.append(_buf)
        plt.close(_fig)
    return (route_frames,)


@app.cell
def _(
    EXT_CACHE_VERSION,
    animation,
    load_or_render_gif,
    my_plan,
    plan_to_jsonable,
    plt,
    route_frames,
):
    _h, _w = route_frames[0].shape[0], route_frames[0].shape[1]

    def _render_ext_l_gif(_save_path):
        _fig, _ax = plt.subplots(figsize=(_w / 100, _h / 100), dpi=100)
        _ax.axis('off')
        _im = _ax.imshow(route_frames[0])
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        def _update(frame_idx):
            frame_idx = min(frame_idx, len(route_frames) - 1)
            _im.set_data(route_frames[frame_idx])
            return (_im,)

        _ani = animation.FuncAnimation(_fig, _update, frames=len(route_frames) + 20,
                                        interval=700, blit=False)
        _ani.save(_save_path, writer=animation.PillowWriter(fps=1.4))
        plt.close(_fig)

    ext_l_gif_path = load_or_render_gif(
        "MEMO_3/extras_assets/route_replay.gif",
        EXT_CACHE_VERSION,
        {"kind": "ext_l_route_replay", "plan": plan_to_jsonable(my_plan), "hold_frames": 20},
        _render_ext_l_gif,
    )
    return (ext_l_gif_path,)


@app.cell
def _(ext_l_gif_path, mo):
    mo.vstack([
        mo.image(src=ext_l_gif_path, alt="Route replay of the submitted plan"),
        mo.md("""
    Each frame reveals one more shuttle run, drawn cumulatively on top of the ones before it, so by the final frame every trip in the plan is visible on the map at once. The title above the map updates every frame to show which trip number is being added, the running priority value collected so far, and the running budget spent so far -- watch both numbers climb together, trip by trip, as CRUDY-1 works through the plan and finally makes its last run out through an exit once every trip is complete. Because the frames are cumulative rather than one-trip-at-a-time, this also makes it easy to see visually which parts of the facility the plan spends most of its budget reaching, versus which supply-dense pockets it clears out cheaply in a single trip.
    """)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extension M -- Full Pheromone Matrix (Every Supply Pair)

    Extension F's animation only ever shows one slice of the pheromone table: the shaft's own outgoing edges, i.e. "which unit does the policy prefer to visit *first*". But `initialise_pheromone` actually builds an entry for every ordered pair of supply units too (`pheromone[u, v]` for every `u != v`), which is what lets the policy express a preference for *sequencing* -- "having just visited unit i, which unit should come next" -- not just a first-pick preference. That second kind of preference never gets a moment on screen in Extension F's animation, so this section shows it directly: the complete, final supply-to-supply pheromone matrix from the exact same instrumented run Extensions F and G both use (`viz_run`), read once at the very end rather than animated frame by frame.
    """)
    return


@app.cell
def _(mo, np, plt, viz_run):
    _mat = np.array(viz_run["pheromone_matrix"])
    _labels = [f"S{_i + 1}" for _i in range(len(viz_run["supply_list"]))]

    _fig, _ax = plt.subplots(figsize=(8, 7))
    _im = _ax.imshow(_mat, cmap="inferno", vmin=0.2, vmax=12.0)
    _ax.set_xticks(range(len(_labels)))
    _ax.set_xticklabels(_labels, fontsize=6, rotation=90)
    _ax.set_yticks(range(len(_labels)))
    _ax.set_yticklabels(_labels, fontsize=6)
    _ax.set_title("Final supply-to-supply pheromone strength", fontsize=10)
    _fig.colorbar(_im, ax=_ax, label="pheromone", fraction=0.046, pad=0.04)
    plt.tight_layout()

    mo.vstack([
        _fig,
        mo.md("""
    Every row and column is one of the facility's 30 supply units, labelled S1 through S30 in the same order Extension K's and this notebook's other tables use them; cell `(i, j)` is the final pheromone strength on the directed edge from unit i to unit j after the full `hybrid_solve` run finishes (diagonal cells are forced to zero, since an edge from a unit to itself is meaningless here). Brighter means more reinforced -- that pair of units was part of enough good plans, often enough, that its pheromone survived evaporation and kept accumulating deposits.

    Bright off-diagonal cells are the sequencing pairs the policy converged on -- if the cell for "S(i) then S(j)" shows up bright, the trained actor learned that visiting j right after i within the same trip is a good idea, on top of (and separate from) the shaft-edge, first-move preference already shown in Extension F. Because the matrix isn't symmetric (pheromone is only ever deposited in the direction a trip actually travelled, i to j, not j to i), asymmetry between a cell and its mirror image across the diagonal is itself meaningful: it shows the policy has a genuine directional preference for *order*, not just for which pairs of units tend to end up in the same trip together.
    """)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extension N -- Large-Scale Hybrid Solver Validation

    Extension K spot-checked policy generalisation on 30 independently generated facilities, split evenly across `n_wings` 2, 3, and 4. That's enough to catch a policy that's simply memorised one layout, but it's a small enough sample that a rare failure mode -- a budget edge case, a facility topology that leaves the solver stuck, or a wing count that scales runtime badly -- could easily sit outside it.

    This section runs the exact same hybrid pipeline the rest of this notebook uses, unmodified, against a much larger and more systematic sweep: thousands of freshly generated facilities per wing count, run end-to-end outside this notebook and logged to `MEMO_3/fac_results.csv`, one row per case -- seed, facility size, budget, returned value/cost, trip count, solve time, and termination status. The goal isn't optimality (that's what Extension K already checked on a tractable subset); it's robustness at scale -- does the pipeline reliably produce a valid, budget-respecting plan, and does it do so in bounded, predictable time, across a facility population it wasn't hand-picked for.
    """)
    return


@app.cell
def _(mo):
    import pandas as pd

    _EXT_N_FILES = {
        "Mission (60% battery)": "MEMO_3/fac_results.csv",
        "Battery contingency (35% battery)": "MEMO_3/fac_results_battery.csv",
    }

    def _build_ext_n_section(_label, _csv_path):
        _df = pd.read_csv(_csv_path)
        _total = len(_df)
        _ok = _df["status"].astype(str).str.lower().eq("ok")
        _terminated = _df["solve_time_s"].notna()
        _no_error = _df["error"].isna() | _df["error"].astype(str).str.strip().eq("")
        _valid_cost = _df["plan_cost"].notna() & _df["plan_cost"].ge(0)
        _valid_value = _df["plan_value"].notna() & _df["plan_value"].ge(0)
        _valid_trips = _df["n_trips"].notna() & _df["n_trips"].ge(0)
        _success = _ok & _terminated & _no_error & _valid_cost & _valid_value & _valid_trips
        _success_count = int(_success.sum())
        _failure_count = _total - _success_count
        _success_pct = 100.0 * _success_count / _total if _total else 0.0
        _mean_time = _df["solve_time_s"].mean()
        _max_time = _df["solve_time_s"].max()
        _mean_value = _df["plan_value"].mean()
        _mean_cost = _df["plan_cost"].mean()
        _status_counts = (_df["status"].fillna("missing").astype(str).value_counts().to_dict())

        _by_wings = (
            _df.assign(_valid=_success)
            .groupby("n_wings", dropna=False)
            .agg(
                cases=("seed", "size"),
                successful=("_valid", "sum"),
                mean_value=("plan_value", "mean"),
                mean_cost=("plan_cost", "mean"),
                mean_time_s=("solve_time_s", "mean"),
                max_time_s=("solve_time_s", "max"),
            )
            .reset_index()
        )

        _by_wings["success_rate_%"] = 100.0 * _by_wings["successful"] / _by_wings["cases"]

        _by_wings = _by_wings[
            [
                "n_wings",
                "cases",
                "successful",
                "success_rate_%",
                "mean_value",
                "mean_cost",
                "mean_time_s",
                "max_time_s",
            ]
        ]

        _display = _df.copy()

        if "solve_time_s" in _display:
            _display["solve_time_s"] = _display["solve_time_s"].round(3)

        for _col in [
            "plan_cost",
            "plan_value",
            "exemplar_cost",
            "budget",
        ]:
            if _col in _display:
                _display[_col] = _display[_col].round(1)

        _finding = (
            f"Every single one of the {_total:,} tested cases "
            f"terminated successfully -- a {_success_pct:.2f}% "
            f"completion rate across all tested facility sizes, "
            f"with zero timeouts, zero recorded errors, and zero "
            f"invalid outputs. Mean solve time is "
            f"{_mean_time:.3f}s overall, with a maximum of "
            f"{_max_time:.3f}s."
            if _failure_count == 0
            else f"{_success_count:,} of {_total:,} tested cases "
            f"were successfully validated, giving a "
            f"{_success_pct:.2f}% completion rate. "
            f"{_failure_count:,} cases failed validation. "
            f"Mean solve time across recorded cases is "
            f"{_mean_time:.3f}s, with a maximum of "
            f"{_max_time:.3f}s."
        )

        return {
            "label": _label,
            "path": _csv_path,
            "df": _df,
            "total": _total,
            "success_count": _success_count,
            "failure_count": _failure_count,
            "success_pct": _success_pct,
            "mean_time": _mean_time,
            "max_time": _max_time,
            "mean_value": _mean_value,
            "mean_cost": _mean_cost,
            "status_counts": _status_counts,
            "by_wings": _by_wings,
            "display": _display,
            "finding": _finding,
        }

    _EXT_N_RESULTS = {
        _label: _build_ext_n_section(_label, _path)
        for _label, _path in _EXT_N_FILES.items()
    }

    _ext_n_blocks = []

    for _label, _r in _EXT_N_RESULTS.items():

        _ext_n_blocks.extend(
            [
                mo.md(f"""
    ## {_label}

    **Source:** `{_r["path"]}`
    """),
                mo.md(f"""
    ### Validation summary

    | Metric | Result |
    |---|---:|
    | Total tested cases | **{_r["total"]:,}** |
    | Successful valid solutions | **{_r["success_count"]:,}** |
    | Unsuccessful / invalid cases | **{_r["failure_count"]:,}** |
    | Successful termination rate | **{_r["success_pct"]:.2f}%** |
    | Mean solve time | **{_r["mean_time"]:.3f} s** |
    | Maximum solve time | **{_r["max_time"]:.3f} s** |
    | Mean returned value | **{_r["mean_value"]:.2f}** |
    | Mean returned cost | **{_r["mean_cost"]:.2f}** |

    A case is counted as successfully validated when it reports
    `status = ok`, has a recorded solve time, has no recorded error,
    and returns non-negative plan value, plan cost and trip count.
    """),
                mo.md("### Results by number of wings"),
                mo.ui.table(
                    _r["by_wings"],
                    selection=None,
                ),
                mo.md(f"""
    ### Finding

    {_r["finding"]}
    """),
                mo.md(f"""
    ### Complete test record

    The table below contains **all {_r["total"]:,} recorded cases** from
    `{_r["path"]}`. No sampling or filtering is applied; it is rendered
    as a scrollable/paginated table in the notebook rather than printed
    in full.

    Each row preserves the original seed, facility size, budget/cap,
    returned value and cost, number of trips, source of the solution,
    runtime, termination status and any recorded error.
    """),
                mo.ui.table(_r["display"], selection=None, page_size=50),
                mo.md("---"),
            ]
        )


    mo.vstack(_ext_n_blocks)
    return (pd,)


@app.cell
def save_controls(mo):
    save_btn = mo.ui.button(value=0, label="Save All Memo 03 Responses",
                            on_click=lambda v: v + 1)
    mo.vstack([
        mo.md("---\n### Save your responses"),
        mo.callout(mo.md(
            "Writes [M3-0] through [M3-6] to `responses_M03.json`. "
            "Each save appends a timestamped entry."), kind="info"),
        save_btn,
    ])
    return (save_btn,)


@app.cell
def save_responses(
    SAVE_FILE_M03,
    datetime,
    json,
    mo,
    os,
    resp_m30,
    resp_m31,
    resp_m31_pseudo,
    resp_m32,
    resp_m33,
    resp_m34,
    resp_m35,
    resp_m36,
    save_btn,
):
    if save_btn.value > 0:
        if os.path.exists(SAVE_FILE_M03):
            try:
                with open(SAVE_FILE_M03, "r") as _f:
                    _all = json.load(_f)
            except Exception:
                _all = []
        else:
            _all = []

        _all.append({
            "timestamp":          datetime.datetime.now().isoformat(),
            "M30_orientation":    resp_m30.value,
            "M31_design":         resp_m31.value,
            "M31_pseudocode":     resp_m31_pseudo.value,
            "M32_quality":        resp_m32.value,
            "M33_complexity":     resp_m33.value,
            "M34_intractability": resp_m34.value,
            "M35_comparison":     resp_m35.value,
            "M36_coherence":      resp_m36.value,
        })

        with open(SAVE_FILE_M03, "w") as _f:
            json.dump(_all, _f, indent=2)

        _result = mo.callout(mo.md(
            f"**Saved** at {datetime.datetime.now().strftime('%H:%M:%S')} "
            f"-- `{SAVE_FILE_M03}`"), kind="success")
    else:
        _result = mo.md("*Press Save above to record your responses.*")
    _result
    return


@app.cell
def footer(mo):
    mo.md("""
    ---
    *End of Memo 03 workbook -- submit on teams.*

    **Before submitting, check:**

    - [ ] Your seed matches your Memo 01 cover sheet.
    - [ ] **[M3-0]** names specific invalidated assumptions, not general remarks.
    - [ ] **[M3-1]** identifies limitations, revises the model with rationale, and
      names a technique for each of selection / grouping / ordering / routing.
      Pseudocode is present and `my_algorithm` runs.
    - [ ] **[M3-2]** re-costs the Memo 01/02 approach under the *revised* model,
      and includes your own two-unit counterexample.
    - [ ] **[M3-3]** annotates pseudocode line by line and separates the
      pathfinding layer from the decision layer.
    - [ ] **[M3-4]** quotes real timings, argues intractability from the count of
      candidate solutions, and reports the calibration-wing gap.
    - [ ] **[M3-5]** states both bounds and identifies the tractability boundary.
    - [ ] **[M3-6]** cites your own code for coherence and states the real-world
      consequence.
    - [ ] Every cell runs without error.
    - [ ] All responses saved to `responses_M03.json`.
    """)
    return


if __name__ == "__main__":
    app.run()
