import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import pulp as pl

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(layout="wide")

st.title("🚆 Paradip Yard Time Expanded Network Optimizer")

st.markdown("""
This dashboard:
- Takes user inputs for node capacities
- Takes edge capacities and travel times
- Takes mandatory holding/processing times
- Builds the railway network graph
- Runs the Time Expanded Network optimization
- Displays maximum throughput
- Shows time-sliced optimized flows
""")

# =========================================================
# DEFAULT VALUES
# =========================================================

DEFAULT_NODE_CAPACITY = {
    1: 6,
    2: 2,
    3: 2,
    4: 8
}

DEFAULT_HOLDING_HOURS = {
    1: 1,
    2: 2,
    3: 2,
    4: 1
}

DEFAULT_EDGE_CAPACITY = {
    (1, 2): 4,
    (1, 3): 3,
    (2, 4): 2,
    (3, 4): 2
}

DEFAULT_TRAVEL_TIME = {
    (1, 2): 1,
    (1, 3): 3,
    (2, 4): 7,
    (3, 4): 2
}

# =========================================================
# SESSION STATE
# =========================================================

if "node_capacity" not in st.session_state:
    st.session_state.node_capacity = DEFAULT_NODE_CAPACITY.copy()

if "holding_hours" not in st.session_state:
    st.session_state.holding_hours = DEFAULT_HOLDING_HOURS.copy()

if "edge_capacity" not in st.session_state:
    st.session_state.edge_capacity = DEFAULT_EDGE_CAPACITY.copy()

if "travel_hours" not in st.session_state:
    st.session_state.travel_hours = DEFAULT_TRAVEL_TIME.copy()

# =========================================================
# RESET BUTTON
# =========================================================

st.sidebar.header("Controls")

if st.sidebar.button("Reset to Default Values"):
    st.session_state.node_capacity = DEFAULT_NODE_CAPACITY.copy()
    st.session_state.holding_hours = DEFAULT_HOLDING_HOURS.copy()
    st.session_state.edge_capacity = DEFAULT_EDGE_CAPACITY.copy()
    st.session_state.travel_hours = DEFAULT_TRAVEL_TIME.copy()
    st.rerun()

# =========================================================
# USER INPUTS
# =========================================================

st.sidebar.header("Node Information")

node_capacity = {}
holding_hours = {}

for node in [1, 2, 3, 4]:
    node_capacity[node] = st.sidebar.number_input(
        f"Node {node} Capacity",
        min_value=1,
        max_value=50,
        value=st.session_state.node_capacity[node],
        key=f"nodecap_{node}"
    )
    holding_hours[node] = st.sidebar.number_input(
        f"Node {node} Holding Hours",
        min_value=0,
        max_value=24,
        value=st.session_state.holding_hours[node],
        key=f"hold_{node}"
    )

# =========================================================
# EDGE INPUTS
# =========================================================

st.sidebar.header("Edge Capacities")

edge_capacity = {}

for edge in DEFAULT_EDGE_CAPACITY:
    edge_capacity[edge] = st.sidebar.number_input(
        f"Edge {edge[0]}→{edge[1]} Capacity",
        min_value=1,
        max_value=50,
        value=st.session_state.edge_capacity[edge],
        key=f"edge_{edge}"
    )

# =========================================================
# TRAVEL TIMES
# =========================================================

st.sidebar.header("Travel Times (Hours)")

travel_hours = {}

for edge in DEFAULT_TRAVEL_TIME:
    travel_hours[edge] = st.sidebar.number_input(
        f"{edge[0]}→{edge[1]} Travel Time",
        min_value=1,
        max_value=24,
        value=st.session_state.travel_hours[edge],
        key=f"time_{edge}"
    )

# =========================================================
# BASIC SETS
# =========================================================

# BUG FIX 1: Use range(24) so time indices are 0..23.
# Using range(25) (i.e. t up to 24) caused storage variables
# s[i, 24] to be referenced but never defined (they were only
# created for range(24) = 0..23), crashing at flow conservation.
T = range(24)
physical_nodes = [1, 2, 3, 4]

# =========================================================
# NETWORK VISUALIZATION
# =========================================================

st.header("📌 Physical Railway Network")

G = nx.DiGraph()

for node in physical_nodes:
    G.add_node(
        node,
        capacity=node_capacity[node],
        holding=holding_hours[node]
    )

for (i, j), cap in edge_capacity.items():
    G.add_edge(
        i, j,
        capacity=cap,
        travel_time=travel_hours[(i, j)]
    )

# =========================================================
# NODE POSITIONS
# =========================================================

pos = {
    1: (0, 0),
    2: (1, 1),
    3: (1, -1),
    4: (2, 0)
}

# =========================================================
# DRAW GRAPH
# =========================================================

fig, ax = plt.subplots(figsize=(12, 6))

nx.draw(
    G, pos,
    with_labels=False,
    node_size=5500,
    node_color="skyblue",
    font_size=16,
    font_weight="bold",
    arrowsize=25,
    ax=ax
)

node_labels = {
    node: (
        f"Node {node}\n"
        f"Cap={node_capacity[node]}\n"
        f"Hold={holding_hours[node]}h"
    )
    for node in G.nodes()
}

nx.draw_networkx_labels(
    G, pos,
    labels=node_labels,
    font_size=11,
    font_weight="bold",
    ax=ax
)

edge_labels = {
    (i, j): (
        f"Cap={edge_capacity[(i,j)]}\n"
        f"Travel={travel_hours[(i,j)]}h"
    )
    for (i, j) in edge_capacity
}

nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=edge_labels,
    font_size=10,
    ax=ax
)

plt.title(
    "Paradip Yard Railway Network",
    fontsize=18,
    fontweight="bold"
)

st.pyplot(fig)

# =========================================================
# OPTIMIZATION
# =========================================================

if st.button("🚀 Run Optimization"):

    model = pl.LpProblem("TEN_Max_Throughput", pl.LpMaximize)

    # =====================================================
    # DECISION VARIABLES
    # =====================================================

    x = {}

    # Source injection arcs: virtual source S → node 1
    for t in T:
        x["S", 1, t] = pl.LpVariable(f"x_S_1_{t}", lowBound=0)

    # Travel arcs: (i, j, departure_time)
    # A train departing node i at time t arrives at node j at
    # t + holding_hours[i] + travel_hours[(i,j)].
    # Only create the variable when the arrival fits in horizon.
    for (i, j), travel_time in travel_hours.items():
        hold = holding_hours[i]
        effective_time = travel_time + hold
        for t in T:
            if t + effective_time <= max(T):
                x[i, j, t] = pl.LpVariable(
                    f"x_{i}_{j}_{t}", lowBound=0
                )

    # Storage variables: units waiting at node i at end of period t
    s = {}
    for i in physical_nodes:
        for t in T:
            s[i, t] = pl.LpVariable(f"s_{i}_{t}", lowBound=0)

    # =====================================================
    # OBJECTIVE: maximise units reaching node 4
    # =====================================================

    model += pl.lpSum(
        x[i, j, t]
        for (i, j, t) in x
        if j == 4
    )

    # =====================================================
    # EDGE CAPACITY CONSTRAINTS
    # =====================================================

    for (i, j) in travel_hours:
        for t in T:
            if (i, j, t) in x:
                model += x[i, j, t] <= edge_capacity[(i, j)]

    # =====================================================
    # STORAGE CAPACITY CONSTRAINTS
    # =====================================================

    for i in physical_nodes:
        for t in T:
            model += s[i, t] <= node_capacity[i]

    # =====================================================
    # FLOW CONSERVATION
    # BUG FIX 2: The original outflow loop iterated over edges
    # as  `for (i2, j), ...` but then checked `if i2 == i` —
    # where the outer loop variable is also named `i`.  Python's
    # scoping means the inner `i2` shadows nothing; the check
    # `i2 == i` compared the *inner* loop variable against
    # itself, always True, so every edge's outflow was collected
    # for every node.  Renamed inner variables to `src`/`dst`
    # to avoid the collision.
    # =====================================================

    for node in physical_nodes:

        if node == 4:          # sink — no conservation needed
            continue

        for t in T:

            inflow = []
            outflow = []

            # --- Incoming travel arcs arriving at `node` at time t ---
            for (src, dst), travel_time in travel_hours.items():
                if dst != node:
                    continue
                hold = holding_hours[src]
                effective_time = travel_time + hold
                dep_t = t - effective_time
                if (src, dst, dep_t) in x:
                    inflow.append(x[src, dst, dep_t])

            # --- Source injection (only node 1) ---
            if node == 1 and ("S", 1, t) in x:
                inflow.append(x["S", 1, t])

            # --- Outgoing travel arcs departing `node` at time t ---
            for (src, dst) in travel_hours:
                if src != node:          # BUG FIX 2 — correct guard
                    continue
                if (src, dst, t) in x:
                    outflow.append(x[src, dst, t])

            # --- Storage carry-over ---
            if t > 0:
                inflow.append(s[node, t - 1])

            # BUG FIX 3: guard t < max(T) so we never reference
            # s[node, 24] which does not exist when T = range(24).
            if t < max(T):
                outflow.append(s[node, t])

            # At the final time step allow leftover stock
            if t == max(T):
                model += pl.lpSum(inflow) >= pl.lpSum(outflow)
            else:
                model += pl.lpSum(inflow) == pl.lpSum(outflow)

    # =====================================================
    # SOLVE
    # =====================================================

    solver = pl.PULP_CBC_CMD(msg=0)
    model.solve(solver)

    # =====================================================
    # RESULTS
    # =====================================================

    st.header("📈 Optimization Results")

    st.metric(
        "Maximum Throughput",
        round(pl.value(model.objective), 2)
    )

    st.success(f"Solver Status: {pl.LpStatus[model.status]}")

    # =====================================================
    # SOURCE INJECTION RESULTS
    # Shows how many units the virtual source pushes into
    # node 1 at each hour of the planning horizon.
    # =====================================================

    st.header("🔛 Source Injections into Node 1")

    injection_data = []

    for t in T:
        var = x.get(("S", 1, t))
        if var is None:
            continue
        val = var.value()
        if val is not None and val > 1e-6:
            injection_data.append({
                "Hour (t)":        t,
                "Injected Units":  round(val, 2)
            })

    if injection_data:
        injection_df = pd.DataFrame(injection_data).sort_values("Hour (t)")
        st.dataframe(injection_df, use_container_width=True)

        st.subheader("📊 Injections Over Time")
        st.bar_chart(injection_df.set_index("Hour (t)"))
    else:
        st.info("No source injections in the optimal solution.")

    # =====================================================
    # FLOW RESULTS
    # Skip "S" source arcs here — they have no entry in
    # travel_hours or holding_hours (handled above instead).
    # =====================================================

    st.header("🚄 Time-Sliced Flow Movements")

    flow_data = []

    for (i, j, t), var in x.items():

        if i == "S":          # virtual source arc — shown above
            continue

        val = var.value()

        if val is not None and val > 1e-6:
            effective_time = travel_hours[(i, j)] + holding_hours[i]
            arrival_time   = t + effective_time

            flow_data.append({
                "Departure Time": t,
                "Arrival Time":   arrival_time,
                "From":           i,
                "To":             j,
                "Holding Hours":  holding_hours[i],
                "Travel Hours":   travel_hours[(i, j)],
                "Effective Time": effective_time,
                "Flow":           round(val, 2)
            })

    flow_df = pd.DataFrame(flow_data)

    if not flow_df.empty:
        flow_df = flow_df.sort_values(by="Departure Time")
        st.dataframe(flow_df, use_container_width=True)

        st.subheader("📊 Flow Over Time")
        chart_df = (
            flow_df.groupby("Departure Time")["Flow"]
            .sum()
            .reset_index()
        )
        st.line_chart(chart_df.set_index("Departure Time"))

    # =====================================================
    # STORAGE RESULTS
    # =====================================================

    st.header("🏗️ Storage Utilization")

    storage_data = []

    for (i, t), var in s.items():
        val = var.value()
        if val is not None and val > 1e-6:
            storage_data.append({
                "Time":        t,
                "Node":        i,
                "Stored Flow": round(val, 2)
            })

    storage_df = pd.DataFrame(storage_data)

    if not storage_df.empty:
        st.dataframe(storage_df, use_container_width=True)

        st.subheader("📊 Storage Over Time")
        storage_chart = (
            storage_df.groupby("Time")["Stored Flow"]
            .sum()
            .reset_index()
        )
        st.area_chart(storage_chart.set_index("Time"))

    # =====================================================
    # BOTTLENECK ANALYSIS
    # =====================================================

    st.header("⚠️ Bottleneck Analysis")

    bottleneck_data = []

    for (i, j), cap in edge_capacity.items():
        utilized = 0
        for t in T:
            if (i, j, t) in x:
                val = x[i, j, t].value()
                if val:
                    utilized += val

        utilization_percent = (utilized / (cap * len(T))) * 100

        bottleneck_data.append({
            "Edge":                f"{i}→{j}",
            "Capacity per Hour":   cap,
            "Total Utilized Flow": round(utilized, 2),
            "Utilization %":       round(utilization_percent, 2)
        })

    bottleneck_df = pd.DataFrame(bottleneck_data)
    st.dataframe(bottleneck_df, use_container_width=True)
