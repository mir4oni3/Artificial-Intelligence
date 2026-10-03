import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium", app_title="AI - Uninformed Search")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Uninformed (Blind) Search

    **Agenda**

    1. Basic concepts: state space, representation, evaluating strategies, global vs local, uninformed vs informed
    2. Graphs in Python and the data structures behind search
    3. Depth-First Search (DFS)
    4. Breadth-First Search (BFS)
    5. Uniform-Cost Search (UCS)
    6. Depth-Limited Search (DLS) and Iterative Deepening Search (IDS)
    7. Summary of the algorithms
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup: imports and helper code

    The helpers below build graphs, compute positions for trees, and draw a graph with the white / blue / red
    colouring.

    You do not need to understand the code. The algorithms in the next sections are what matters.
    """)
    return


@app.cell
def _():
    # The imports used in the rest of the notebook
    import heapq
    import sys
    import time
    from collections import deque
    from dataclasses import dataclass, field

    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D

    return Line2D, deque, heapq, nx, plt


@app.cell
def _(Line2D, nx, plt):
    # Helper code that can be skipped

    def make_undirected(edges):
        graph = {}
        for edge in edges:
            u, v = edge[0], edge[1]
            cost = edge[2] if len(edge) == 3 else 1
            graph.setdefault(u, []).append((v, cost))
            graph.setdefault(v, []).append((u, cost))
        for node in graph:
            graph[node].sort()                    # children in alphabetical order
        return graph


    def make_tree(children):
        graph = {}
        for parent, kids in children.items():
            graph.setdefault(parent, [])
            for kid in kids:
                graph[parent].append((kid, 1))
                graph.setdefault(kid, [])
        return graph


    def tree_layout(graph, root):
        positions = {}
        next_leaf_x = [0]                         # a list so the inner function can modify it

        def place(node, depth):
            kids = [kid for kid, _ in graph[node]]
            if not kids:
                positions[node] = (next_leaf_x[0], -depth)
                next_leaf_x[0] += 1
            else:
                for kid in kids:
                    place(kid, depth + 1)
                xs = [positions[kid][0] for kid in kids]
                positions[node] = (sum(xs) / len(xs), -depth)   # parent centred above its children

        place(root, 0)
        return positions


    def draw_graph(graph, positions, state=None, goals=(), title="", show_weights=False,
                   annotations=None, show_legend=True, figsize=(8, 4.5)):
        state = state or {}
        visited = state.get("visited", set())
        expanded = state.get("expanded", set())
        current = state.get("current")
        annotations = annotations if annotations is not None else state.get("labels", {})

        nx_graph = nx.Graph()
        for u, neighbours in graph.items():
            nx_graph.add_node(u)
            for v, cost in neighbours:
                nx_graph.add_edge(u, v, weight=cost)
        nodes = list(nx_graph.nodes)

        fills = ["#c0504d" if n in expanded else "#4f81bd" if n in visited else "white" for n in nodes]
        outlines = ["orange" if n == current else "green" if n in goals else "black" for n in nodes]
        widths = [4 if n == current or n in goals else 1.2 for n in nodes]

        fig, ax = plt.subplots(figsize=figsize)
        nx.draw_networkx_edges(nx_graph, positions, ax=ax, width=2)
        nx.draw_networkx_nodes(nx_graph, positions, ax=ax, nodelist=nodes, node_size=700,
                               node_color=fills, edgecolors=outlines, linewidths=widths)
        for node in nodes:
            x, y = positions[node]
            text_colour = "white" if node in visited or node in expanded else "black"
            ax.text(x, y, str(node), ha="center", va="center", fontsize=11,
                    fontweight="bold", color=text_colour)
            if node in annotations:
                ax.annotate(str(annotations[node]), (x, y), xytext=(0, 20), textcoords="offset points",
                            ha="center", fontsize=10, fontweight="bold")
        if show_weights:
            edge_labels = {(u, v): data["weight"] for u, v, data in nx_graph.edges(data=True)}
            nx.draw_networkx_edge_labels(nx_graph, positions, edge_labels=edge_labels, ax=ax, font_size=10)
        if show_legend:
            legend = [("white", "node"), ("#4f81bd", "visited"), ("#c0504d", "expanded")]
            handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=colour, markeredgecolor="black",
                              markersize=12, label=label) for colour, label in legend]
            handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="white",
                                  markeredgecolor="orange", markeredgewidth=3, markersize=12, label="current"))
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1), frameon=False)
        ax.set_title(title)
        ax.margins(0.12)
        ax.axis("off")
        fig.tight_layout()
        return fig


    def reconstruct_path(parent, goal):
        path = [goal]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        return path[::-1]

    return draw_graph, make_tree, make_undirected, reconstruct_path, tree_layout


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 1. Basic concepts

    ## 1.1 State space

    - **State**: a representation of the task in the process of its solution (a node in a graph).
        - **Initial State** - written as **S** (the starting node of the graph)
        - **Intermediate State** - any capital letter except **S** and **G**, any state in between (a node in the graph)
        - **Goal State** - written as **G**; if there are more: **G1**, **G2**, etc. (the node we want to reach)
    - **Successor Function**: function, returning a list of all reachable states from a given state with a single move.
    - **Path Cost**: the total cost of a given path (sum of weights of each edge that lies in the given path in the graph)
    - **State Space**: the set of all possible states that can be obtained from a given initial state.

    > A **solution** is a sequence of actions leading from the initial state to a goal state.

    A concrete example - route finding on a map:

    | Concept | Route finding |
    |---|---|
    | State | the city you are currently in |
    | Initial state | the city you start from |
    | Goal state | the city you want to reach |
    | Successor function | drive along one road to a neighbouring city |
    | Path cost | sum of the kilometres of all roads driven |
    | Solution | the list of roads to drive, in order |

    ## 1.2 Representation of the state space

    The state space is represented as a **graph** or a **tree**, where each state is a **node** and each
    application of the successor function is an **edge**.

    When the state space is represented as a tree, it is called a **Search Tree**:

    - the *initial state* is the *root* of the tree
    - the *goal states* are the *leaves*

    Two things to keep in mind, because they explain most of the bugs you will write:

    - Any graph can be "unrolled" into a search tree starting from S. The **same state can appear many
      times** in that tree, once for every path that reaches it.
    - If the graph has a **cycle**, the unrolled tree is **infinite** (A - B - A - B - ...). This is why graph
      algorithms keep a `visited` structure: to never process the same state twice.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.3 Evaluating search algorithms

    Search algorithms are evaluated along the following dimensions:

    - **Completeness**: Always finds a solution, if one exists
    - **Optimality**: Always finds a least-cost solution, if one exists
    - **Time Complexity**: maximum number of nodes generated during the search
    - **Space Complexity**: maximum number of nodes in memory at any one moment

    We look at the **worst-case** complexity (Big O notation). Time and space complexity are measured in terms of:

    - **b** - maximum branching factor of the search tree (maximum number of children of a node)
    - **d** - depth of the shallowest goal (in weighted graphs, the least-cost goal may be deeper)
    - **m** - maximum depth of the state space (may be ∞)

    > "Tree search" and "graph search" are two variants of an algorithm, not two kinds of state spaces. Tree search does not keep visited states, so on a state space with cycles it can revisit the same states forever (A → B → A → …); its search tree can be infinite even when the state space is finite. Graph search keeps an explored set and expands each state at most once, so its cost is also bounded by the total number of states - so if the graph is finite, the search is also finite (doesn't explore a cycle over and over again). This is why DFS with an explored set (using graph search) is complete on finite state spaces.

    In the following example the initial state S is at level 0:

    - b = 3 (S has 3 children, no node has more)
    - m = 3 (the deepest node, G1, is at level 3)
    - d = 2 (the nearest goal state, G2, is at level 2)
    """)
    return


@app.cell
def _(deque, draw_graph, make_tree, tree_layout):
    example_tree = make_tree({
        "S": ["A", "B", "C"],
        "A": ["D", "E"],
        "C": ["F", "G2"],
        "D": ["G1"],
    })


    def tree_stats(graph, root, goals):
        branching = max(len(children) for children in graph.values())

        depth = {root: 0}                 # BFS from the root to get the level of every node
        queue = deque([root])
        while queue:
            node = queue.popleft()
            for child, _ in graph[node]:
                depth[child] = depth[node] + 1
                queue.append(child)

        max_depth = max(depth.values())
        goal_depth = min(depth[goal] for goal in goals)
        return branching, goal_depth, max_depth


    example_b, example_d, example_m = tree_stats(example_tree, "S", goals=["G1", "G2"])
    print(f"b = {example_b}, d = {example_d}, m = {example_m}")

    draw_graph(example_tree, tree_layout(example_tree, "S"), goals=["G1", "G2"],
               title=f"b = {example_b}, d = {example_d}, m = {example_m}", show_legend=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.4 Global search vs local search

    - **Global Search**: search strategies that look all over the state space. If necessary, all states will be traversed.
    - **Local Search**: they only look at the local area, so they can only see states in this area. If the
      solution is outside of it, they will not be able to find it.


    ### Local search

    - Local search in AI is an **optimization algorithm** used to find the optimal (or near optimal) solution more **quickly**.
    - Local search algorithms are used when we care **only about a solution, not the path** to a solution (it returns a final state, not a path).
    - It can be used on problems that can be formulated as finding a solution **maximizing a criterion**
      among a number of candidate solutions(optimization problems).
    - Local search algorithms **move from solution to solution** in the space of candidate solutions (the search
      space) by applying local changes, until a solution deemed optimal is found or a time bound is elapsed.
    - It is usually **neither complete nor optimal**. It can get stuck in a **local optimum**: a state better than all its neighbours but worse than the best solution. Randomness, restarts and occasionally accepting a worse move are ways to escape.
    - It uses **constant memory** and works on **huge or even continuous state spaces** where global search is hopeless.
     > Gradient descent, the algorithm used to train neural networks, used in the implementation of LLMs is a local search algorithm.

    Everything in this seminar is **global** search. Local search comes back later with genetic algorithms.

    ## 1.5 Uninformed (blind) search vs informed (heuristic) search

    - **Uninformed Search**: uninformed strategies use **only the information available in the problem
      definition**. They have no information on where the goal state might be (if one exists).
        - Examples: DFS, BFS, UCS, DLS, IDS - today's topic.
    - **Informed Search**: informed strategies have information on the goal state which helps in more
      efficient searching. This information is obtained by a function (**heuristic**) that estimates how
      close a state is to the goal state.
        - Examples: Greedy Best-First Search, A*, Beam Search, Hill Climbing - next week's topic.

    ## 1.6 Graph and tree traversal: the vocabulary

    - **Node**: graphs and trees consist of nodes and edges. When a node is neither "visited" nor "expanded",
      we do not mark it in any way - it stays **white**.
    - **Visited (generated) node**: when we visit a node, or in other words **add it to the data structure** we use, we call
      it a "visited node". We mark it in **blue**.
    - **Expanded node**: when we expand a node, or in other words **remove it from the data structure** we use
      and add its children to this data structure (if the node has children), we call it an "expanded node".
      We mark it in **red**.

    The data structure holding the visited-but-not-yet-expanded nodes is called the **frontier** (or fringe). **The only thing that differs between DFS, BFS and UCS is which kind of data structure the
    frontier is.** Everything else is the same loop.

    > We can think of time/space complexity as a measurement of the total number of generated/visited nodes in the worst case (not just expanded).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 2. Graphs in Python

    ## 2.1 Representing a graph

    We will store every graph as a dictionary that maps a node to the list of its neighbours, together with
    the cost of the edge:

    ```python
    graph = {
        "S": [("A", 7), ("B", 9), ("C", 14)],
        "A": [("B", 10), ("D", 15), ("S", 7)],
        ...
    }
    ```

    This is called an **adjacency list**. Getting the neighbours of a node is `graph[node]`, which is `O(1)`.
    For unweighted graphs every cost is `1` and we just ignore it (`for neighbour, _ in graph[node]`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 3. Depth-First Search (DFS)

    ## 3.1 The idea

    - **Go deep first**: DFS explores as far as possible down one branch of a tree or graph before it
      backtracks. It is like exploring a maze by always choosing the first available path until you hit a dead end.
    - **Uses a stack**: the algorithm uses a stack (FILO) to keep track of the nodes it needs to visit. When it
      hits a dead end, it takes the most recently visited node from the stack and continues from there.

    The stack can be explicit (a `list`), or implicit - the **call stack** of a recursive function.

    ## 3.2 Recursive DFS (Tree Search)

    On the binary tree DFS goes all the way down the leftmost branch (S, A, C, G), backtracks one level (H),
    backtracks two levels (D, I, J), and only then touches the right half of the tree.
    """)
    return


@app.cell
def _(draw_graph, make_tree, tree_layout):
    def dfs_order(graph, node):
        order = [node]
        for child, _ in graph[node]: # looping over 2-element tuples where the second element is not important
            order += dfs_order(graph, child)
        return order

    binary_tree = make_tree({
        "S": ["A", "B"],
        "A": ["C", "D"], "B": ["E", "F"],
        "C": ["G", "H"], "D": ["I", "J"], "E": ["K", "L"], "F": ["M", "N"],
    })
    binary_tree_pos = tree_layout(binary_tree, "S")

    print("DFS order:", " ".join(dfs_order(binary_tree, "S")))
    draw_graph(binary_tree, binary_tree_pos, title="1. binary tree", show_legend=False, figsize=(8, 3.5))
    return binary_tree, binary_tree_pos


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.3 Recursive DFS (Graph Search)

    On a graph we must remember which nodes were already visited, otherwise the cycles (S - C - F - G - S,
    C - E - H - G - F - C) would make us walk in circles forever.

    The recursion: visit a node, then for every neighbour that is still unvisited, recursively do the same.
    If the recursive call found the goal, stop and pass `True` all the way up. The current **path** from the
    start to the node we are in is exactly the contents of the stack.
    """)
    return


@app.function
def dfs_recursive(graph, start, goal): # wrap the recursive dfs
    visited = {start}
    expanded = set()
    path = [start] # the path from start to the current node

    def explore(node): # recursive function to explore the graph
        if node == goal:
            return True # goal found, stop the recursion

        expanded.add(node)
        for neighbour, _ in graph[node]:
            if neighbour not in visited:
                visited.add(neighbour)
                path.append(neighbour)
                if explore(neighbour): # recursive call
                    return True
                path.pop()
        return False

    return list(path) if explore(start) else None


@app.cell
def _(draw_graph, make_undirected):
    unweighted_graph = make_undirected([
        ("A", "B"), ("A", "S"), ("S", "C"), ("S", "G"), ("C", "D"),
        ("C", "F"), ("C", "E"), ("G", "F"), ("G", "H"), ("E", "H"),
    ])
    unweighted_graph_pos = {
        "A": (0.5, 4), "B": (0, 3), "S": (1, 3), "C": (0.6, 2), "G": (1.4, 2),
        "D": (0.2, 1), "F": (1.0, 1), "E": (0.6, 0), "H": (1.4, 0),
    }

    print("DFS order:", " ".join(dfs_recursive(unweighted_graph, "A", "G")))
    draw_graph(unweighted_graph, unweighted_graph_pos, title="2. unweighted graph", show_legend=False, figsize=(8, 3.5))
    return unweighted_graph, unweighted_graph_pos


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    DFS found **A - S - C - E - H - G**, even though **A - S - G** exists. DFS returns the *first* path it
    stumbles upon, not the shortest one. This is what "not optimal" means.

    If we do not stop at G, DFS visits the whole graph in the order **A, B, S, C, D, E, H, G, F** - this is the
    traversal shown in the unweighted slides.

    > We may run into stack overflow exceptions for deep DFS recursions, so in these cases, iterative DFS with our own stack (not relying on the function call stack) may be preferred.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.4 Properties of DFS

    - **Complete? No, in general**
        - In infinite spaces, It can go deep down an infinite-depth branch.
        - In finite spaces, if doing a tree search, DFS can go around in a loop if there is a cycle.
        - In finite spaces, if doing a graph search (keep track of visited) ⇒ **complete**.
    - **Optimal? No**
        - it returns the first solution it finds, as we just saw.
    - **Time? $O(b^m)$**
        - in the worst case the goal is the last node visited, so we visit
      all $b^m$ nodes ($b^m$ is the upper bound of nodes for trees with depth $m$).
        - Terrible if $m$ is much larger than $d$, but if solutions are **dense**,
      it may be much faster than breadth-first.
    - **Space? $O(bm)$**
        - we say this is linear in space, because it is linear in $m$ (we consider $b$ to be fixed)
        - the $b$ factor comes from storing all siblings before expanding the next node, and we do this $m$ times in total - at most $b$ siblings for each of the $m$ levels, $O(bm)$
        - it is also possible to directly explore the first unvisited child, before generating all other children, achieving $O(m)$ (like we did in the code example above), but often we just generate all nodes because it is simpler and the memory saving is negligible.

    > A **dense graph** is a graph in which the number of edges is close to the maximal number of edges.

    ### Example for an infinite state space

    Find a sequence of actions that transforms an initial positive integer **N** into a target integer **T**
    using only the operations:

    - Operation A: "Multiply by 2" ($x \to 2x$)
    - Operation B: "Subtract 1" ($x \to x - 1$)

    Start **N = 5**, target **T = 8** (5 - 1 = 4, 4 · 2 = 8). The state space contains every positive integer,
    so it is infinite. If DFS always tries "multiply by 2" first, it falls into the trap
    5 → 10 → 20 → 40 → 80 → ... and never comes back.

    **Fix**: if the current value exceeds the target, do not continue (with $x > T$ you can only get back to $T$
    by subtracting, so forbid further multiplying). Now every state is between 1 and $2T$, the
    space is finite, and DFS with a check for repeated states on the path is complete.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4. Breadth-First Search (BFS)

    ## 4.1 The idea

    BFS explores the state space **level by level**: first the start, then everything 1 step away, then
    everything 2 steps away, and so on. The frontier is a **queue (FIFO)**: nodes are expanded in the same
    order in which they were visited.

    ## 4.2 BFS (Tree Search)
    """)
    return


@app.cell
def _(binary_tree, binary_tree_pos, deque, draw_graph):
    def bfs_order(graph, root):
        order = []
        queue = deque([root]) # we use a deque as a queue
        while queue:
            node = queue.popleft()
            order.append(node)
            queue.extend(child for child, _ in graph[node])
        return order


    print("BFS order:", " ".join(bfs_order(binary_tree, "S")))
    draw_graph(binary_tree, binary_tree_pos, title="3. binary tree", show_legend=False, figsize=(8, 3.5))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.3 BFS (Graph Search)

    The code is the same as iterative DFS with one change: `stack.pop()` becomes `queue.popleft()`.
    """)
    return


@app.cell
def _(deque, reconstruct_path):
    def bfs(graph, start, goal, log=None):
        queue = deque([start])
        visited = {start}
        expanded = set()
        parent = {start: None}

        while queue:
            node = queue.popleft()
            if node == goal:
                return reconstruct_path(parent, goal) # follow the parent pointers back from goal to start

            expanded.add(node)
            for neighbour, _ in graph[node]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    parent[neighbour] = node
                    queue.append(neighbour)
        return None

    return (bfs,)


@app.cell
def _(bfs, draw_graph, unweighted_graph, unweighted_graph_pos):
    bfs_path = bfs(unweighted_graph, "A", "G")
    print("BFS path:", " -> ".join(bfs_path))
    draw_graph(unweighted_graph, unweighted_graph_pos, title="4. unweighted graph", show_legend=False, figsize=(8, 3.5))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    BFS found **A - S - G**, the path with the fewest edges. If we let it traverse the whole graph, the visited
    order is **A, B, S, C, G, D, E, F, H** - the same as in the unweighted slides.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.4 Properties of BFS

    - **Complete? Yes** (if $b$ is finite, otherwise no)
    - **Optimal? Yes** (if the cost is equal for all edges, not in general)
    - **Time?** $1 + b + b^2 + b^3 + \ldots + b^d + b(b^d - 1) = O(b^{d+1})$
      - If the goal is the last node at level $d$, we visit all nodes up to level $d$, and we also generate the
      $b$ children of every node at level $d$ except the last one (the goal).
    - **Space? $O(b^{d+1})$**
      - it keeps every node in memory.
    - **Data structure: queue**

    > **Space is the big problem**: BFS is exponential, where DFS was linear.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Goal test on generation vs on expansion

    Our BFS tests `node == goal` when a node is **removed** from the queue (on expansion). That is where the
    extra $b(b^d - 1)$ term comes from: we generate the whole level $d + 1$ before we reach the goal.

    BFS can be modified to apply the goal test **when a node is generated** (when it is added to the queue).
    For BFS this is still optimal (for unit step costs) and the time becomes $O(b^d)$.
    For UCS in the next section this trick is **not** allowed, and we will see why.

    ---

    # 5. Uniform-Cost Search (UCS)

    ## 5.1 The idea

    BFS finds the path with the fewest **edges**. When edges have different costs, that is not the cheapest path.

    Uniform-cost search always selects for expansion the frontier node with the **lowest path cost from the
    start** node, written $g(n)$.

    **Frontier**: a **priority queue**, where the priority of a node $v$ is the path cost
    from the start node to $v$.

    When we reach a node that is already in the frontier through a cheaper path, we lower its cost
    (**relaxation**). `heapq` cannot change the priority of an element that is already inside, so we just push
    the node again with the new cost, and when an old, more expensive copy is popped later we skip it.

    The algorithm relies on the key property that the cost $g(u)$ at the exact moment node $u$ is expanded (popped off the priority queue) is the absolute lowest cost path from the start node to $u$. I have put a (*) comment where this property is used.

    This algorithm is almost identical to **Dijkstra's algorithm**, the main difference being is that UCS terminates upon
    finding a solution state, instead of finding the shortest path to all states.
    """)
    return


@app.cell
def _(heapq, reconstruct_path):
    def ucs(graph, start, goal):
        frontier = [(0, start)]              # (path cost g, node) - heap sorts by path cost
        best = {start: 0}                    # cheapest known cost to every visited node
        parent = {start: None}
        expanded = set()

        while frontier:
            cost, node = heapq.heappop(frontier)
            if node in expanded:             # (*) - an old, more expensive copy, skip it
                continue
            if node == goal:                 # (*) - expanded the goal, so cheapest path has been found   
                return reconstruct_path(parent, goal), cost

            expanded.add(node)
            for neighbour, step_cost in graph[node]:
                if neighbour in expanded: # (*) - already expanded, so we have found the cheapest path to it
                    continue
                new_cost = cost + step_cost
                old_cost = best.get(neighbour)
                if old_cost is None or new_cost < old_cost: # relaxation
                    best[neighbour] = new_cost
                    parent[neighbour] = node
                    heapq.heappush(frontier, (new_cost, neighbour))

        return None, float("inf") # goal not reachable

    return (ucs,)


@app.cell
def _(draw_graph, make_undirected, ucs):
    weighted_graph = make_undirected([
        ("S", "A", 7), ("S", "B", 9), ("S", "C", 14), ("A", "B", 10), ("A", "D", 15),
        ("B", "C", 2), ("B", "D", 11), ("C", "G", 9), ("D", "G", 6),
    ])
    weighted_graph_pos = {
        "S": (0, 2), "A": (2, 1), "B": (2, 3), "C": (1, 4.2), "D": (4, 3.4), "G": (3, 4.5),
    }

    ucs_path, ucs_cost = ucs(weighted_graph, "S", "G")
    print("UCS path:", " -> ".join(ucs_path), f"(cost = {ucs_cost})")
    draw_graph(weighted_graph, weighted_graph_pos, title="5. weighted graph", show_weights=True, show_legend=False, figsize=(8, 3.5))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.2 Properties of UCS

    Let us define the **optimal path cost** as $C^*$, and the **minimal cost of an edge** in the state space
    graph as $\varepsilon$.

    - **Complete? Yes**.
      - We require every weight $w$ in the graph to be $w \ge \varepsilon > 0$ to avoid infinite paths with finite cost, for example $1+1/2+1/4+1/8+...=2$.
    - **Optimal? Yes**.
      - We can think of the algorithm as iteratively exploring all paths with a fixed start node in sorted order (increasing total path cost). The first time that a path appears with an end node being the target node, we know this is the cheapest such path, since it was found before any other path with the same end node.
      > NOTE: UCS is optimal only for graphs with non-negative weights.
    - **Time?** $O(b^{\lceil C^*/\varepsilon \rceil})$
      - To guarantee finding an optimal path of cost $C^*$, UCS must explore all paths whose path cost is less than or equal to $C^*$. Every step along any path adds at least $\varepsilon$ to the total path cost. So to reach the path with cost $C^*$, we must do no more than $C^*/\varepsilon$ steps. Since this is in general not an integer, we take the ceil, and the upper bound on the maximum number of steps becomes $\lceil C^*/\varepsilon \rceil$. Since the branching factor is $b$, this means that at every one of those steps we add at most $b$ nodes in the frontier, in the worst case.

    - **Space?** $O(b^{\lceil C^*/\varepsilon \rceil})$
      - All of the generated states must be kept in the frontier, and as we saw above, these states are $O(b^{\lceil C^*/\varepsilon \rceil})$.
    ---

    # 6. Depth-Limited Search (DLS) and Iterative Deepening Search (IDS)

    ## 6.1 Depth-Limited Search

    **DLS = DFS with depth limit $l$**, i.e. nodes at depth $l$ have no successors.

    It fixes DFS's problem with infinite depth: the search can never go deeper than $l$. But if the goal is
    deeper than $l$, DLS cannot find it.

    On a graph we still need to avoid loops. A global `visited` set won't work - a node first reached
    by a long path (and cut off) would be skipped when reached later by a shorter path (this is in the case of a cycle where there are more than one paths from node A to node B). So we only check that
    the node is not already on the **current path**.
    """)
    return


@app.function
def dls(graph, start, goal, limit):
    path = [start]
    cutoff = False

    def recurse(node, depth_left):
        nonlocal cutoff # without it, cutoff = True below would create a new local variable instead of updating dls's
        if node == goal:
            return True

        if depth_left == 0:                      # a node at depth l has no successors
            if graph[node]:                      # if there are successors that are cut off
                cutoff = True
            return False

        for neighbour, _ in graph[node]:
            if neighbour in path:                # avoid loops along the current path
                continue
            path.append(neighbour)
            if recurse(neighbour, depth_left - 1):
                return True
            path.pop()
        return False

    if recurse(start, limit):
        return list(path)

    return "cutoff" if cutoff else None


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Properties of DLS

    - **Complete?**
        - Yes, if $l \ge d$
        - No, if $l \lt d$
    - **Optimal? No**
        - it is a DFS, it returns the first solution within the limit, not the shallowest
    - **Time? $O(b^l)$**
        - as DFS, but with $m = l$
    - **Space? $O(bl)$**
        - as DFS, but with $m = l$

    > DLS searches in a subset of the state space, but it is **NOT** a local search algorithm!
    > It still looks at every state in that subspace systematically, and it still returns a path.

    ## 6.2 Iterative Deepening Search

    How do we pick $l$ when we do not know $d$? We do not - we **iterate over the limit** in DLS:
    $l = 0, 1, 2, \ldots$ until DLS stops returning "cutoff".
    """)
    return


@app.function
def ids(graph, start, goal, max_limit=100):
    for limit in range(max_limit + 1):
        result = dls(graph, start, goal, limit)
        if result != "cutoff":
            return result, limit          # a path, or None if the whole space was searched
    return None, max_limit


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Every iteration starts from scratch, and the top levels are searched again and again.
    That looks wasteful, but, surprisingly, it barely matters in practice.

    ### Properties of IDS

    - **Complete? Yes**
    - **Optimal? Yes**, if step cost = const
      - It effectively performs a level-by-level search of the state space,
      like BFS. A goal at depth $d$ is found in iteration $l = d$, and no goal was found in the iterations before,
      so there is no shallower one.
    - **Time? $O(b^d)$**
      - Nodes at level 0 are generated in all $d + 1$ iterations, nodes at level 1 in $d$ iterations, ..., nodes at level $d$ only once: $b^0 + (b^0 + b^1) + (b^0 + b^1 + b^2) + \ldots + (b^0 + b^1 + \ldots + b^d)$
      $= (d+1)b^0 + db^1 + (d-1)b^2 + \ldots + b^d = O(b^d)$
    - **Space? $O(bd)$, i.e. linear space!** - it is a DFS at every iteration.

    IDS combines the advantages of both - the **memory of DFS** and the **completeness and optimality of BFS**.
    It is the preferred uninformed search when the state space is large and the depth of the solution is unknown.

    ## 6.3 IDS vs BFS

    Let $b = 10$ and $d = 5$, solution at the far right leaf.

    $N_{BFS} = b^0 + b^1 + \ldots + b^d + b(b^d - 1) = 1 + 10 + 100 + 1{,}000 + 10{,}000 + 100{,}000 + 999{,}990 = 1{,}111{,}101$

    $N_{IDS} = (d+1)b^0 + db^1 + \ldots + b^d = 6 + 50 + 400 + 3{,}000 + 20{,}000 + 100{,}000 = 123{,}456$

    - IDS does better because the other nodes at depth $d$ are not expanded - the repeated work on the upper levels is small, because the upper you are, the less nodes there are (exponential branching factor $b$ has not fully kicked in yet).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 7. Summary of the algorithms

    | Criterion | DFS | BFS | UCS | DLS | IDS |
    |---|---|---|---|---|---|
    | **Complete** | No | Yes* (if $b$ is finite) | Yes* (if step cost $\ge \varepsilon$) | Yes* (if $l \ge d$) | Yes |
    | **Optimal** | No | Yes* (if step cost = 1) | Yes | No | Yes* (if step cost = 1) |
    | **Time** | $O(b^m)$ | $O(b^{d+1})$ | $O(b^{\lceil C^*/\varepsilon \rceil})$ | $O(b^l)$ | $O(b^d)$ |
    | **Space** | $O(bm)$ | $O(b^{d+1})$ | $O(b^{\lceil C^*/\varepsilon \rceil})$ | $O(bl)$ | $O(bd)$ |
    | **Data structure** | Stack | Queue | Priority Queue | Stack | Stack |

    Some general rules where a given algorithm may be preferred:

    - Edge costs differ and you need the cheapest path → **UCS**
    - Equal costs, the space is small enough to fit in memory → **BFS**
    - Equal costs, big space, unknown solution depth → **IDS**
    - You only need *some* solution, the space is finite, and all solutions are at a known depth → **DFS**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 8. Homework: Frog Leap Puzzle

    ## 8.1 The problem

    The game has $2N + 1$ fields. In the beginning there are $N$ frogs looking to the left, placed on the
    rightmost $N$ fields, and $N$ frogs looking to the right, placed on the leftmost $N$ fields. The aim of the
    game is to swap the places of the frogs and reach the opposite configuration.

    Rules:

    - Each frog can only move in the direction it looks at.
    - A frog can jump into the free field in front of it, or jump over one frog into the free field behind it.

    **Input**: $N$ - the number of frogs looking in one direction.

    **Output**: all the configurations needed to get from the start to the final state.

    Sample input: `2`

    Sample output:

    ```
    >>_<<
    >_><<
    ><>_<
    ><><_
    ><_<>
    _<><>
    <_><>
    <<>_>
    <<_>>
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
