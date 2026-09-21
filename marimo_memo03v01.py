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

    return


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
                                        continue  # no-op, same position

                                    new_dst = dst_base[:pos] + seg_variant + dst_base[pos:]
                                    new_dst_cost = trip_cost(new_dst)

                                    # Delta cost: only the affected trip(s) changed --
                                    # everything else in the plan is untouched, so
                                    # there's no need to re-sum the whole plan.
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
                    plan_for_ant = or_opt_plan(plan_for_ant, inst, max_segment=3)   # <-- new: cheap routing polish
                    plan_for_ant = top_up_plan(plan_for_ant, inst, max_insert_size=2)  # now has freed budget to spend
                    
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
    return aco_rng_seed, my_algorithm, my_plan


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
    #### S3-B Side Memo
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
    return


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
