# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.23.3",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="AI - Genetic Algorithms")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Genetic Algorithms (GAs)

    **Agenda**

    1. What is a genetic algorithm: definition, key concepts, GAs as local search (the link to local beam search)
    2. The main steps of a GA: the reproduction cycle, the 8-queens example
    3. Selection: selection pressure, roulette wheel, SUS, tournament, rank-based, elitism, truncation
    4. Crossover: standard (one-point, two-point, uniform), order crossovers for permutations, real-valued genes
    5. Mutation: operators and the mutation rate
    6. Putting it all together: the genetic algorithm, evolving a string, evolving an image
    7. Summary
    8. Homework (choose one): the Travelling Salesman Problem or the Knapsack Problem
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup: imports and helper code

    The helpers draw chess boards with queens, roulette wheels and crossovers, print chromosomes, and implement
    tournament selection (section 3.4), which the genetic algorithm in section 6 uses. You do not need to understand the
    drawing code.
    """)
    return


@app.cell
def _():
    # The imports used in the rest of the notebook
    import itertools

    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image, ImageDraw

    return Image, ImageDraw, itertools, np, plt


@app.cell
def _(np, plt):
    # Helper code that can be skipped

    def draw_queens(queen_rows, title=""):
        # queen_rows[col] = the row of the queen in column col (row 0 is the top row)
        n = len(queen_rows)
        fig, ax = plt.subplots(figsize=(4, 4))
        for row in range(n):
            for col in range(n):
                colour = "#f0d9b5" if (row + col) % 2 == 0 else "#b58863"
                ax.add_patch(plt.Rectangle((col, n - 1 - row), 1, 1, color=colour))
        for col, row in enumerate(queen_rows):
            ax.text(col + 0.5, n - 1 - row + 0.5, "♛", ha="center", va="center", fontsize=224 // n)
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title)
        plt.close(fig)
        return fig


    def fmt(chromosome):
        # [1, 2, 3] -> "1 2 3"
        return " ".join(str(gene) for gene in chromosome)


    def draw_roulette_wheel(labels, fitness_scores, pointers, title=""):
        # pointers are positions on the wheel between 0 and the total fitness; the wheel starts at the top, clockwise
        total = sum(fitness_scores)
        selected = np.searchsorted(np.cumsum(fitness_scores), pointers, side="right")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(fitness_scores, labels=[f"{label}\n{score / total:.0%}" for label, score in zip(labels, fitness_scores)],
               colors=plt.cm.Pastel1.colors, startangle=90, counterclock=False, labeldistance=0.6,
               wedgeprops={"edgecolor": "white", "linewidth": 2}, textprops={"fontsize": 12})
        for number, (position, index) in enumerate(zip(pointers, selected), 1):
            angle = np.radians(90 - 360 * position / total)
            direction = np.array([np.cos(angle), np.sin(angle)])
            ax.annotate("", xy=1.02 * direction, xytext=1.35 * direction, arrowprops={"arrowstyle": "-|>", "lw": 2})
            ax.text(*1.5 * direction, f"{number}: {labels[index]}", ha="center", va="center", fontsize=11,
                    fontweight="bold")
        ax.set_xlim(-1.6, 1.6)
        ax.set_ylim(-1.6, 1.6)
        ax.set_title(f"{title}\nselected: {' '.join(labels[i] for i in selected)}")
        plt.close(fig)
        return fig


    def draw_crossover(ax, parent1, parent2, child1, child2, cuts, title=""):
        # each gene is coloured by the parent it comes from; the dashed lines are the crossover points
        colours = ("#4f81bd", "#f79646")
        segments = [sum(position >= cut for cut in cuts) for position in range(len(parent1))]
        rows = [("Parent 1", parent1, [0] * len(parent1)), ("Parent 2", parent2, [1] * len(parent2)),
                ("Child 1", child1, [segment % 2 for segment in segments]),
                ("Child 2", child2, [1 - segment % 2 for segment in segments])]
        for row, (name, genes, origins) in enumerate(rows):
            y = -row - (0.7 if row >= 2 else 0)          # a gap between the parents and the children
            ax.text(-0.3, y + 0.5, name, ha="right", va="center", fontsize=11)
            for position, (gene, origin) in enumerate(zip(genes, origins)):
                ax.add_patch(plt.Rectangle((position, y), 1, 1, facecolor=colours[origin], edgecolor="white", lw=2))
                ax.text(position + 0.5, y + 0.5, str(gene), ha="center", va="center", color="white",
                        fontsize=12, fontweight="bold")
        for cut in cuts:
            ax.plot([cut, cut], [-3.9, 1.2], color="black", linestyle="--", lw=1.5)
        ax.set_xlim(-2.6, len(parent1) + 0.2)
        ax.set_ylim(-4, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title)


    def tournament_selection(population, fitness_scores, num_selected, rng, k=3):
        selected = []
        for _ in range(num_selected):
            tournament = rng.choice(len(population), size=k, replace=False)   # k different individuals
            winner = max(tournament, key=lambda i: fitness_scores[i])
            selected.append(population[winner])
        return selected

    return (
        draw_crossover,
        draw_queens,
        draw_roulette_wheel,
        fmt,
        tournament_selection,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 1. What is a genetic algorithm?

    ## 1.1 Definition

    - A **search and optimization technique** inspired by natural evolution.
    - It uses the principles of **selection**, **crossover** and **mutation**.
    - It **evolves a population of possible solutions** to find the best (fittest) one.

    ## 1.2 Key concepts

    Each solution (**chromosome**) is represented by a set of parameters (**genes**). The **fitness** of each chromosome measures how well it solves the problem.

    | Term | Meaning | Example |
    |---|---|---|
    | **Gene** | a single parameter or feature that defines part of a solution | a weight, a value, a decision variable; `7` |
    | **Chromosome** | a complete set of genes - one potential solution | $[x_1, x_2, x_3, \dots]$; `3 7 1 9 4` |
    | **Population** | a group of chromosomes - many solutions tested at once | `3 7 1 9 4`, `8 2 6 5 0`, `1 1 7 3 6` |
    | **Fitness function** | a score for each chromosome - **higher is better** | For N-Queens: the number of non-attacking pairs of queens |
    | **Generation** | the population at one iteration of the algorithm | |

    ## 1.3 Genetic algorithms as local search

    > **Genetic Algorithms = stochastic beam search + generating successors from pairs of states**

    Recall beam search from the lecture about informed search: keep $l$ states, generate their successors, keep
    the best $l$. A GA differs in two ways:

    - **Stochastic**: the individuals that survive are chosen *randomly*, with a bias towards the fitter ones (selection),
      instead of always taking the top $l$. This keeps more diversity and avoids getting stuck too early.
    - **Successors from pairs**: a new state is created by **combining two parents** (crossover), not just by changing one
      state. The idea is that if two parents each solved a different part of the problem well, a child may get both
      good parts.

    Like every local search, a GA works with **complete states** (every chromosome is a full candidate solution), the path
    is irrelevant, and it gives **no guarantee** of finding the optimum.

    ```
    Population
      1 | 1 2 3 4 5 |      <- the 5 in chromosome 1 is a gene
      2 | 2 1 3 4 5 |      <- the whole row is a chromosome
      3 | 1 2 5 4 3 |
        ...
      n | 2 3 5 4 1 |
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 2. The main steps of a genetic algorithm

    ## 2.1 The reproduction cycle

    **Initialization** → **Evaluation** → **Selection** → **Crossover** → **Mutation** → **Replacement** → **Evaluation** and repeat until the population converges


    1. **Initialization**: create a population of random chromosomes.
    2. **Evaluation**: compute the fitness of every chromosome.
    3. **Selection**: choose chromosomes (parents), preferring the fitter ones, based on the evaluation - **repetition is allowed**.
    4. **Crossover**: combine pairs of parents into children.
    5. **Mutation**: make small random changes to the children.
    6. **Replacement**: the children (and possibly the best old individuals) create the next generation.
    7. **Convergence check**: stop when a chromosome is good enough, when a number of generations has passed, or when
       the best fitness has not improved for a while. Otherwise go to step 2.

    ## 2.2 Example: 8-queens

    - **Chromosome**: a string of 8 digits - digit $i$ is the row of the queen in column $i$. E.g.
      `24748552`.
    - **Fitness**: the number of **non-attacking pairs** of queens. There are $\binom{8}{2} = 28$ pairs in total, so the perfect solution without any conflicts has fitness 28.

    We have 4 chromosomes that form the first generation:

    ```
    (a) Initial      (b) Fitness     (c) Selection     (d) Crossover      (e) Mutation
        population       function

    24748552              24(31%)      327|52411         32748552           32748152
    32752411              23(29%)      247|48552         24752411           24752411
    24415124              20(26%)      32752|411         32752124           32252124
    32543213              11(14%)      24415|124         24415411           24415417
    ```

    - The selection probability is proportional to the fitness (b). This means that the probability of the first chromosome to be selected is $\frac{24}{24 + 23 + 20 + 11} = \frac{24}{78} = 31\%$
    - Based on these probabilities, parents are selected randomly (c). In this example `32752411` is chosen twice, `32543213` is not chosen at all
    - After selection, each parent pair is cut at a random crossover point (c). In this example, this happens after the 3rd gene for the first parent pair, after the 5th for the
      second parent pair.
    - Each pair of parents creates a pair of children where the first child takes the first part of parent A and the second part of parent B, and the other child takes the first part of parent B and the second part of parent A (d).
    - Each child mutates with a small probability (e). Here, mutation is implemented as one random gene(digit) being replaced by another gene (digit).
    - After the mutation, the resulting population is the initial population for the next generation
    """)
    return


@app.cell
def _(draw_queens, itertools):
    def queens_fitness(chromosome):
        # the number of non-attacking pairs of queens
        rows = [int(gene) for gene in chromosome]
        n = len(rows)
        attacking = sum(1 for c1, c2 in itertools.combinations(range(n), 2)
                        if rows[c1] == rows[c2] or abs(rows[c1] - rows[c2]) == c2 - c1)
        return n * (n - 1) // 2 - attacking


    _population = ["24748552", "32752411", "24415124", "32543213"]
    _scores = [queens_fitness(x) for x in _population]
    for _x, _s in zip(_population, _scores):
        print(f"{_x}: fitness {_s}, selection probability {_s / sum(_scores):.0%}")

    print()

    # Hand picked possible children from the 4 chromosomes above
    for _x in ["32748552", "24752411", "32752124", "24415411", "32748152", "32252124", "24415417"]:
        print(f"{_x}: fitness {queens_fitness(_x)}")

    # digit 1 is the bottom row, so the row from the top is 8 - digit
    draw_queens([8 - int(gene) for gene in "24748552"], title="24748552, fitness 24")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that one generation does not have to improve everything: the best parent had fitness 24, and the best child
    `32748152` also has 24. The crossover point fell in a place where the left part of `32752411` and the right part of
    `24748552` fit together well, and the mutation of the 6th gene happened to help. Many children are worse than their
    parents - selection removes them in the next generation.

    ---

    # 3. Selection

    **Selection** is the stage of a GA in which individual chromosomes are chosen from the population to be parents (using crossover).

    ## 3.1 Selection pressure

    **Selection pressure** refers to how strongly the algorithm favors the best-performing chromosomes when choosing
    who gets to be a parent and create the next generation. It is the degree of bias towards fitter individuals in the
    selection process.

    - **High selection pressure** → strongly favors the best individuals
        - faster convergence, but risk of **premature convergence**: loss of diversity, stuck in local optima
    - **Low selection pressure** → individuals are chosen more randomly
        - slower convergence, but better **exploration** of the search space (more genetic diversity)

    **Premature convergence** means that the population becomes (almost) identical copies of one mediocre individual
    before the optimum is found. Crossover of two identical parents gives the same individual again, so from that
    moment only mutation can bring anything new.

    All examples below use the same population: **A - E with fitness 10, 20, 30, 25, 15** (higher is better).

    The selection function in the helpers, `tournament_selection` (section 3.4), takes the population, the fitness
    scores, the number of individuals to select, and a random number generator, and returns the selected individuals.
    It works with **indices** into the population, so it works for any kind of chromosome (strings, lists, images).
    """)
    return


@app.cell
def _(np):
    population = ["A", "B", "C", "D", "E"]
    fitness_scores = [10, 20, 30, 25, 15]  # higher is better
    rng = np.random.default_rng(42)  # a random number generator with a fixed seed: the same results every run
    return fitness_scores, population, rng


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.2 Roulette wheel selection

    Chromosomes are selected with a **probability proportional to their fitness**. Imagine a roulette wheel where each
    chromosome has a slot size corresponding to its fitness value; fitter individuals have larger slots and are more
    likely to be selected.

    1. Calculate the total fitness of the population.
    2. Compute each individual's selection probability:

       $$p_i = \frac{f_i}{\sum_j f_j}$$

    3. Spin the wheel (generate a random number) to select individuals based on these probabilities.

    Example: the total fitness is $10 + 20 + 30 + 25 + 15 = 100$, so A gets 10% of the wheel, B 20%, C 30%, D 25% and
    E 15%.

    | Pros | Cons |
    |---|---|
    | Simple | Premature convergence if a single individual is overwhelmingly fit (it takes over the wheel) |
    | Works well when fitness differences are moderate | Sensitive to scaling: small differences in fitness can have a large effect |

    Adding 1000 to every fitness does not change which individual is better, but the wheel becomes almost uniform (19.8% to 20.2%) - the selection pressure is gone. The
    opposite also happens: if fitness values are close to 0, tiny differences become huge relative differences.

    Selecting 5 parents means spinning the wheel 5 times. Each arrow below is one spin:
    """)
    return


@app.cell
def _(draw_roulette_wheel, fitness_scores, population, rng):
    _spins = rng.uniform(0, sum(fitness_scores), size=5)    # 5 independent spins, one per parent
    draw_roulette_wheel(population, fitness_scores, _spins, title="Roulette wheel: 5 separate spins")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.3 Stochastic Universal Sampling (SUS)

    A variation of roulette wheel selection that ensures a more even and consistent spread of selected individuals.
    Imagine the roulette wheel as before, but instead of one spin, place **multiple evenly spaced pointers** on it
    (equal to the number of parents needed). **Spin the wheel once** - all pointers select individuals simultaneously.

    For $n$ parents the pointers are $\frac{\text{total fitness}}{n}$ apart. With our total fitness 100 and $n = 5$
    parents, the pointers are 20 apart, so C (30% of the wheel) is always hit by 1 or 2 pointers, while with 5 separate
    roulette spins it could be selected anywhere from 0 to 5 times.

    | Pros | Cons |
    |---|---|
    | Reduces randomness in selection (less variance) | Slightly more complex to implement than standard roulette selection |
    | Ensures even sampling proportional to fitness | |

    The same wheel with SUS - one spin moves all 5 pointers together:
    """)
    return


@app.cell
def _(draw_roulette_wheel, fitness_scores, np, population, rng):
    _spacing = sum(fitness_scores) / 5
    _pointers = rng.uniform(0, _spacing) + _spacing * np.arange(5)   # one spin, then evenly spaced
    draw_roulette_wheel(population, fitness_scores, _pointers, title="SUS: one spin, 5 evenly spaced pointers")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.4 Tournament selection

    Randomly select a small group (a **tournament**) of $k$ individuals, and pick the **best one from that group** to be
    a parent. Repeat for every parent needed. Selection pressure can be controlled by changing $k$:

    - Larger $k$ → stronger selection pressure
    - Smaller $k$ → more diversity ($k = 1$ is a completely random choice)

    | Pros | Cons |
    |---|---|
    | Easy to implement. Selection pressure can be controlled by changing $k$ | May lose diversity if the tournament size is too large |

    It only compares fitness values, it never adds or divides them - so it does not care about scaling, and it works
    even with negative fitness.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In general, with tournaments of size $k$ (without replacement), the $k - 1$ worst individuals can never be selected.
    Compare with roulette (C 30%, A 10%): with $k = 3$, C wins every tournament it takes part in - 60% of them - and the
    two worst individuals, A and E, are never selected. Tournament selection with $k = 3$ has a stronger selection
    pressure than roulette wheel selection here.

    ## 3.5 Rank-based selection

    Instead of using raw fitness values, individuals are **ranked according to their fitness**. The selection probability
    depends on the **rank**, not on the absolute fitness.

    1. Sort the population by fitness.
    2. Assign selection probabilities based on rank (e.g. linearly or exponentially decreasing).
    3. Select individuals using these rank-based probabilities.

    With linear ranking, the worst individual gets rank 1 and the best rank $n$, and the probability of rank $r$ is
    $\frac{r}{1 + 2 + \dots + n} = \frac{r}{n(n+1)/2}$. For 5 individuals the ranks 1 - 5 give probabilities
    $\frac{1}{15}, \dots, \frac{5}{15}$ = 0.067, 0.133, 0.2, 0.267, 0.333.

    | Pros | Cons |
    |---|---|
    | Prevents domination by one very fit chromosome | Can be slower to converge because it doesn't consider big differences between chromosomes |
    | The selection pressure does not depend on the scale of the fitness values | |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When C's fitness grows from 30 to 300, roulette gives it 81% of the wheel - after a few generations the population is
    full of copies of C. With ranks, C is still just "the best" and keeps 33%.

    ## 3.6 Elitism and truncation

    **Elitism**: the **best individuals from the current generation** are automatically passed to the next generation,
    without alteration. It is not a way to choose parents - it is used *together with* one of the methods above.

    | Pros | Cons |
    |---|---|
    | Guarantees that the best solutions are preserved - the best fitness can never get worse | Reduces diversity |
    | Improves convergence speed | |

    **Truncation selection**: select only the **top fraction** of the population (e.g. top 30%) based on fitness. All
    selected individuals have an **equal** probability of being parents.

    1. Sort the population by fitness.
    2. Keep only the top percentage (say, the best 20 - 50%).
    3. Randomly select parents from this truncated set.

    | Pros | Cons |
    |---|---|
    | Very simple and deterministic | Can quickly reduce genetic diversity |
    | Strong selection pressure (fast convergence) | May get stuck in local optima |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.7 Summary of selection methods

    | Method | Key idea | Strength | Weakness |
    |---|---|---|---|
    | **Roulette wheel** | probability proportional to fitness | simple | premature convergence |
    | **Tournament** | the best of a random subset wins | adjustable pressure | can lose diversity |
    | **Rank-based** | selection by rank | stable pressure | slower convergence |
    | **SUS** | evenly spaced roulette sampling | low variance | more complex |
    | **Elitism** | the best always survive | guarantees progress | less diversity |
    | **Truncation** | the top % survive | simple and strong pressure | fast loss of diversity |

    ---

    # 4. Crossover

    **Crossover** is a process of taking more than one parent solution and producing a child solution from them.

    There are many types of crossover, and which ones can be used depends on how the chromosome is **encoded**.

    ## 4.1 Standard crossovers

    **One-point crossover (1PX)**: a single crossover point is chosen randomly, and all genes after that point are
    swapped between the parents.

    **Two-point (and k-point) crossover (2PX)**: two points are chosen randomly, and the genetic material between them
    is exchanged.

    **Uniform crossover (UX)**: a random **mask** decides, gene by gene, from which parent each gene comes - getting a gene
    from one of the parents **randomly**! Child 1 takes the gene from the parent the mask says; child 2 takes it from the
    other one.
    """)
    return


@app.cell
def _():
    def one_point_crossover(parent1, parent2, i):
        return parent1[:i] + parent2[i:], parent2[:i] + parent1[i:]


    def two_point_crossover(parent1, parent2, i, j):
        return (parent1[:i] + parent2[i:j] + parent1[j:],
                parent2[:i] + parent1[i:j] + parent2[j:])


    def uniform_crossover(parent1, parent2, mask):
        # mask[k] = 1 -> child 1 takes gene k from parent 1, otherwise from parent 2 (child 2 the opposite)
        child1 = [a if m == 1 else b for a, b, m in zip(parent1, parent2, mask)]
        child2 = [b if m == 1 else a for a, b, m in zip(parent1, parent2, mask)]
        return child1, child2

    return one_point_crossover, two_point_crossover


@app.cell
def _(draw_crossover, one_point_crossover, plt, two_point_crossover):
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(12, 3.2))
    _parent1, _parent2 = "11010110", "10011011"
    draw_crossover(_left, _parent1, _parent2, *one_point_crossover(_parent1, _parent2, 4), cuts=[4],
                   title="One-point crossover, cut at 4")
    _parent1, _parent2 = "11011010", "10101110"
    draw_crossover(_right, _parent1, _parent2, *two_point_crossover(_parent1, _parent2, 2, 5), cuts=[2, 5],
                   title="Two-point crossover, cuts at 2 and 5")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.2 Crossovers for permutations

    The crossovers above break permutations - it is possible for some genes to be repeated and other genes to be dropped completely. To address this, it is possible to modify the traditional **one and two point crossovers** such that **the created children are valid permutations** - we call them **one point order crossover** and **two point order crossover**. There also exist other separate crossovers that also handle this problem - for example **Partially-Mapped Crossover (PMX)** and **Cycle Crossover (CX)**.

    This is the algorithm that defines **one point order crossover**:

    1. Choose a random cut point.
    2. Child 1 copies the genes **before the cut** from parent 1, unchanged.
    3. The remaining positions are filled with the **missing genes, in the order in which they appear in parent 2** - go
       through parent 2 from the start and skip every gene that child 1 already has.
    4. Child 2 is built the same way, with the roles of the parents swapped.

    ```
    Cut at 3:
    Parent 1:         1 2 3 | 4 5 6 7 8
    Parent 2:         3 7 5 | 1 6 8 2 4

    Plain one-point:  1 2 3 | 1 6 8 2 4     <- 1 and 2 appear twice, 5 and 7 are missing
    Child 1:          1 2 3 | 7 5 6 8 4     <- the missing 4 5 6 7 8, in the order of parent 2
    Child 2:          3 7 5 | 1 2 4 6 8     <- the missing 1 2 4 6 8, in the order of parent 1
    ```

    This modification of one point crossover ensures that **every child is a valid permutation**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.3 Crossovers used when genes are real numbers

    When the genes are real numbers, a child can lie *between* its parents:

    - **Linear / whole arithmetic crossover**: $child_1 = \alpha \cdot P_1 + (1 - \alpha) \cdot P_2$,
      $child_2 = (1 - \alpha) \cdot P_1 + \alpha \cdot P_2$, for some $\alpha \in [0, 1]$ (fixed or random).
    - **Single arithmetic crossover**: the same, but only for one randomly chosen gene; the other genes are copied.

    E.g. with $\alpha = 0.25$: parents `[1.0, 4.0]` and `[3.0, 0.0]` → $child_1 = [0.25 \cdot 1 + 0.75 \cdot 3,
    \ 0.25 \cdot 4 + 0.75 \cdot 0] = [2.5, 1.0]$.

    ---

    # 5. Mutation

    **Mutation** is used to **maintain genetic diversity** from one generation of a population to the
    next. It is analogous to biological mutation - it alters one or more gene values in a chromosome from its initial
    state - a small random change to a single chromosome. It helps explore new areas of the search space that may not be
    reachable through crossover alone. For example if no chromosome in the population has a 7, no crossover can ever
    create one.

    | Operator | Encoding | What it does |
    |---|---|---|
    | **Bit flip** | binary | each bit flips (0 → 1, 1 → 0) with a small probability, typically $\frac{1}{L}$ ($L$ = chromosome length) |
    | **Random resetting** | integer | a gene gets a new random value from the domain - "bit flip" for integers |
    | **Swap** | permutation | exchanges the positions of two randomly chosen genes |
    | **Insertion** | permutation | takes one gene out and inserts it at another position; the genes in between shift |
    | **Reverse** (inversion) | permutation | reverses the order of the genes between two random points |
    | **Scramble** | permutation | randomly shuffles the genes between two random points |
    | **Gaussian** | real | adds a small random value from a normal distribution to a gene |

    Note that random resetting and bit flip **cannot** be used on permutations - they would create duplicate values.
    """)
    return


@app.cell
def _(fmt, np):
    def bit_flip(bits, positions):
        return [1 - bit if k in positions else bit for k, bit in enumerate(bits)]


    def swap_mutation(chromosome, i, j):
        mutated = list(chromosome)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated


    def insertion_mutation(chromosome, i, j):
        mutated = list(chromosome)
        gene = mutated.pop(i)       # take the gene at position i out ...
        mutated.insert(j, gene)     # ... and put it back at position j
        return mutated


    def reverse_mutation(chromosome, i, j):
        return chromosome[:i] + chromosome[i:j + 1][::-1] + chromosome[j + 1:]


    def scramble_mutation(chromosome, i, j, rng):
        return chromosome[:i] + list(rng.permutation(chromosome[i:j + 1])) + chromosome[j + 1:]


    def gaussian_mutation(chromosome, sigma, rng):
        return [round(gene + rng.normal(0, sigma), 2) for gene in chromosome]


    _rng = np.random.default_rng(7)
    _c = [2, 1, 5, 4, 3]
    print("original:          ", fmt(_c))
    print("swap 1 and 3:      ", fmt(swap_mutation(_c, 1, 3)))
    print("insertion 1 -> 3:  ", fmt(insertion_mutation(_c, 1, 3)))
    print("reverse 1 .. 3:    ", fmt(reverse_mutation(_c, 1, 3)))
    print()
    print("bit flip 3 and 6:  ", fmt([1, 1, 0, 1, 0, 1, 1, 0]), "->", fmt(bit_flip([1, 1, 0, 1, 0, 1, 1, 0], {3, 6})))
    print("swap 1 and 6:      ", fmt(list(range(1, 9))), "->", fmt(swap_mutation(list(range(1, 9)), 1, 6)))
    print("reverse 1 .. 5:    ", fmt(list(range(1, 9))), "->", fmt(reverse_mutation(list(range(1, 9)), 1, 5)))
    print("scramble 1 .. 4:   ", fmt(list(range(1, 9))), "->", fmt(scramble_mutation(list(range(1, 9)), 1, 4, _rng)))
    print("gaussian, sigma 0.05:", fmt([0.5, 2.3, 1.8, 3.1]), "->", fmt(gaussian_mutation([0.5, 2.3, 1.8, 3.1], 0.05, _rng)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.1 The mutation rate

    The **mutation rate** is the probability that a single gene is mutated. It
    controls the balance between **exploration** and **exploitation**:

    - **Low rate** → children stay close to their parents, the search converges fast but can get stuck (premature
      convergence). With a rate of 0, nothing new can ever appear.
    - **High rate** → more diversity, but good solutions get destroyed and the GA degrades into a **random search**.

    Standard implementations:

    - Binary encoding: Bit-flip with probability $\frac{1}{L}$ ($L$ = chromosome length) - on average one gene per chromosome changes
    - Real-valued: Gaussian noise added to each variable with probability $\frac{1}{n}$ ($n$ = number of variables); sigma
      depends on the range of the variables
    - Permutation: Control the probability per chromosome of applying one swap / insertion /
      reverse
    - **Adaptive mutation**: if the population diversity is low, increase the mutation rate,
      otherwise decrease it.

    Implementation considerations and common pitfalls:

    - Ensure the mutation keeps the solution **valid** - consider the problem constraints
    - Balance exploration vs exploitation
    - Too high a mutation rate → random search; too low → premature convergence
    - The wrong mutation type for the encoding
    - **premature convergence** is the genetic algorithm version of a local optimum. The usual fixes are random restarts, a larger population, a higher mutation rate, or weaker selection pressure.

    ---

    # 6. Putting it all together

    The `genetic_algorithm` function below works for any problem - it only needs four problem-specific functions:

    - `random_individual()` - function that creates one random chromosome (initialization)
    - `fitness(chromosome)` - fitness function, higher is better (evaluation)
    - `crossover(parent1, parent2)` - returns two children (crossover)
    - `mutate(chromosome)` - returns a (possibly) mutated copy

    Every generation it keeps the `num_elites` best chromosomes unchanged (**elitism**), and fills the rest of the new population with mutated children of parents chosen by **tournament selection**.
    """)
    return


@app.cell
def _(tournament_selection):
    def genetic_algorithm(random_individual, fitness, crossover, mutate, rng, population_size=100,
                          num_generations=500, num_elites=2, tournament_size=3, target_fitness=None):
        population = [random_individual() for _ in range(population_size)]     # initialization
        history = []                                                            # the best fitness per generation

        for generation in range(num_generations):
            scores = [fitness(x) for x in population]                           # evaluation
            best_first = sorted(range(population_size), key=lambda i: scores[i], reverse=True) # indices of decreasing fitness-es
            history.append(scores[best_first[0]])
            if target_fitness is not None and scores[best_first[0]] >= target_fitness:
                break                                                           # convergence check

            next_population = [population[i] for i in best_first[:num_elites]]  # elitism
            while len(next_population) < population_size:
                parent1, parent2 = tournament_selection(population, scores, 2, rng, k=tournament_size)
                for child in crossover(parent1, parent2):
                    next_population.append(mutate(child))
            population = next_population[:population_size]

        best = max(population, key=fitness)
        return best, history

    return (genetic_algorithm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.1 Evolving a string

    A classic toy example: evolve the sentence `the quick brown fox jumps over the lazy dog` from random letters.

    - **Gene**: a character from `abcdefghijklmnopqrstuvwxyz ` (26 letters and a space - 27 possible values)
    - **Chromosome**: a string of the same length as the target (43 characters)
    - **Fitness**: the number of characters that match the target at the same position (43 = solved)
    - **Crossover**: one-point
    - **Mutation**: random resetting - every character is replaced by a random one with probability `mutation_rate`

    A random string matches about $\frac{43}{27} \approx 1.6$ characters, and there are $27^{43} \approx 10^{61}$
    possible strings - trying them at random is hopeless.

    We run it with four different mutation rates:
    """)
    return


@app.cell
def _(genetic_algorithm, np, plt):
    genes = "abcdefghijklmnopqrstuvwxyz "
    target = "the quick brown fox jumps over the lazy dog"


    def evolve_string(mutation_rate, seed=42):
        rng = np.random.default_rng(seed)

        def random_string():
            return "".join(genes[k] for k in rng.integers(len(genes), size=len(target)))

        def fitness(dna):
            return sum(x == y for x, y in zip(dna, target))

        def crossover(a, b):
            c = int(rng.integers(1, len(a)))
            return a[:c] + b[c:], b[:c] + a[c:]

        def mutate(dna):
            return "".join(genes[rng.integers(len(genes))] if rng.random() < mutation_rate else x for x in dna)

        return genetic_algorithm(random_string, fitness, crossover, mutate, rng, population_size=100,
                                 num_generations=300, target_fitness=len(target))


    _fig, _ax = plt.subplots(figsize=(10, 4))
    for _rate in (0, 1 / len(target), 0.1, 0.3):
        _best, _history = evolve_string(_rate)
        print(f"mutation rate {_rate:.3f}: best fitness {_history[-1]:>2} after {len(_history) - 1:>3} generations "
              f"- '{_best}'")
        _ax.plot(_history, label=f"mutation rate {_rate:.3f}")
    _ax.axhline(len(target), color="gray", linestyle="--")
    _ax.set_xlabel("generation")
    _ax.set_ylabel("best fitness (matching characters)")
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - **Rate 0**: no mutation - the population can only recombine the letters that happened to be in the first random
      population. It stops improving after a few generations: **premature convergence**.
    - **Rate $\frac{1}{L}$** (one character per string on average): reaches the target.
    - **Rate 0.1**: progress, but every child loses ~4 correct characters to mutation, so it struggles to finish.
    - **Rate 0.3**: almost a third of every string is randomized each generation - practically a **random search**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.2 Evolving an image

    The same idea applied to pixels:

    - **Individual**: an RGB image with the same size as the target - a 3D array `height × width × 3` of values 0 - 255
    - **Fitness**: minus the sum of the absolute pixel differences to the target (higher is better, 0 is a perfect match)
    - **Crossover**: uniform crossover with a random boolean mask that has the same shape as each individual
    - **Mutation**: random resetting of pixel channels (the lab version), compared with Gaussian mutation
    """)
    return


@app.cell
def _(Image, ImageDraw, genetic_algorithm, np, plt):
    _picture = Image.new("RGB", (32, 24), (30, 60, 120))
    _draw = ImageDraw.Draw(_picture)
    _draw.ellipse((4, 4, 18, 18), fill=(240, 200, 40))
    _draw.rectangle((20, 8, 29, 20), fill=(200, 40, 40))
    target_image = np.array(_picture).astype(float)


    def evolve_image(mutation, seed=0):
        rng = np.random.default_rng(seed)

        def random_image():
            return rng.integers(0, 256, size=target_image.shape).astype(float)

        def fitness(image):
            return -np.sum(np.abs(image - target_image))

        def crossover(a, b):
            mask = rng.random(a.shape) < 0.5
            return np.where(mask, a, b), np.where(mask, b, a)

        def reset(image):                                   # random resetting of 0.5% of the pixel channels
            mask = rng.random(image.shape) < 0.005
            image = image.copy()
            image[mask] = rng.integers(0, 256, size=mask.sum())
            return image

        def gaussian(image):                                # small Gaussian changes to 2% of the pixel channels
            mask = rng.random(image.shape) < 0.02
            return np.clip(image + mask * rng.normal(0, 25, size=image.shape), 0, 255)

        return genetic_algorithm(random_image, fitness, crossover, reset if mutation == "reset" else gaussian, rng,
                                 population_size=100, num_generations=1000)


    _fig, _axes = plt.subplots(1, 3, figsize=(10, 3))
    _axes[0].imshow(target_image.astype(np.uint8))
    _axes[0].set_title("target")
    for _ax, _mutation in zip(_axes[1:], ("reset", "gaussian")):
        _best, _history = evolve_image(_mutation)
        _ax.imshow(_best.astype(np.uint8))
        _ax.set_title(f"{_mutation}: fitness {_history[-1]:.0f}")
        print(f"{_mutation:>8}: fitness {_history[0]:.0f} -> {_history[-1]:.0f}")
    for _ax in _axes:
        _ax.axis("off")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 7. Summary

    Designing a genetic algorithm means making these choices:

    | Component | Question | Options |
    |---|---|---|
    | **Encoding** | how is a solution written as a chromosome? | binary, integer / string, permutation, real |
    | **Fitness** | how good is a chromosome? higher must be better | the objective itself, $\frac{1}{\text{cost}}$, with penalties for invalid solutions |
    | **Selection** | who becomes a parent? | roulette, SUS, tournament, rank, truncation |
    | **Crossover** | how are two parents combined? | must match the encoding - section 4 |
    | **Mutation** | how is diversity kept? | must match the encoding - section 5; the rate controls exploration |
    | **Replacement** | who survives to the next generation? | all children, children + elitism, best of parents and children |
    | **Termination** | when do we stop? | target fitness, number of generations, no improvement |

    - **Selection pressure** and the **mutation rate** decide the balance between exploitation (fast convergence) and
      exploration (diversity). Too much exploitation → premature convergence; too much exploration → random search.
    - Like all local search, a GA gives good solutions, but **no guarantee of optimality**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 8. Homework

    Solve **one** of the two tasks below with a genetic algorithm. **Only one of them is required** - you do not need to
    do both.

    ## 8.1 Task 1: The Travelling Salesman Problem (TSP)

    ### Problem definition

    There are $N$ cities, and **every city is connected to every other city** by a direct road. The distance between two
    cities is the same in both directions (here: the straight-line distance between the two points). The salesman must
    **visit every city exactly once**, and the objective is to **minimize the total distance** travelled.

    He does **not** have to return to the city he started from - we are looking for a path, not a round trip. Since the end
    does not have to equal the beginning, **the start city is not important**: the path can start at any city, and which
    city comes first is simply part of the solution.

    With $N$ cities there are $\frac{N!}{2}$ different paths (every order of the cities, and each path can be travelled in
    two directions). For 15 cities that is $\approx 6.5 \cdot 10^{11}$, for 100 cities $\approx 4.7 \cdot 10^{157}$. TSP is
    NP-hard, and GAs are a classic way to find good (not necessarily optimal) paths.

    - **Input**: $N$ - an integer ($N \le 100$), the number of points (cities) in space.
    - The program **generates $N$ random points** in a two-dimensional coordinate system.
    - We are looking for the **shortest road that passes through each point only once**.
    - **Output**: the length of the currently best path in the population for at least five generations:
        - 1 - on the 1st generation
        - 2, 3, 4 - optional (somewhere in between)
        - 5 - after the last generation

    ### TSP as a genetic algorithm

    - **Chromosome**: a path - a **permutation** of the cities, e.g. `3 1 4 0 2` = start in city 3, then visit 1, 4, 0
      and end in 2. Its length is the sum of the $N - 1$ distances between consecutive cities.
    - **Fitness**: shorter paths must have **higher** fitness. The length itself has to be transformed, e.g.
      $\frac{1}{\text{length}}$; or use a selection method that only compares (tournament) and maximize $-\text{length}$.
    - **Crossover**: an order-coded crossover - one-point order, OX, PMX or CX. Binary crossovers would
      produce paths that visit cities twice.
    - **Mutation**: swap, insertion or reverse. **Reverse** is especially good for TSP: reversing a segment of
      the path changes only the two edges at its ends - the segment itself is the same road travelled backwards.

    ### Tactics to try

    - **A starting configuration**: a population of **60 paths over 14 cities**; every generation keep the **2 best paths
      (elitism)**, pick parents by **tournament selection ($k = 3$)**, combine them with the **one-point order
      crossover**, and mutate each child with the chosen operator (swap, insertion or reverse) with a **mutation rate of
      0.3**. With a mutation rate of 0 the population loses its diversity and often gets stuck early (premature
      convergence).
    - **Precompute the distances**: an $N \times N$ distance matrix is computed once; the length of a path is then just
      $N - 1$ lookups.
    - **Use elitism**: without it the best path can be lost, and the printed best length could go up between generations.
    - **Watch the diversity**: if the best length stops improving very early, the selection pressure is too strong or the
      mutation rate is too low.
    - **Stopping**: a fixed number of generations, or stop when the best length has not improved for a while.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8.2 Task 2: The Knapsack Problem (KP)

    ### Problem definition

    A knapsack can carry at most $M$ kg. There are $N$ items: item $i$ has a weight $m_i$ and a value $c_i$, and each
    item can be taken **at most once** - it is either in the knapsack or not. The goal is to choose the **subset of items
    with the maximum total value**, without the total weight exceeding the capacity:

    $$\text{maximize} \sum_{i=1}^{N} c_i x_i \quad \text{subject to} \quad \sum_{i=1}^{N} m_i x_i \le M, \qquad x_i \in \{0, 1\}$$

    There are $2^N$ subsets - for 100 items that is $\approx 1.3 \cdot 10^{30}$. The knapsack problem is NP-hard.

    - **Input**:
        - One line: `M N` - the capacity of the knapsack and the number of items.
        - Then $N$ lines with two integers each: `mi ci` - the weight and the value of the $i$-th item.
    - **Output**:
        - At least 10 values, one per line - the best value in the population at that moment: the first generation,
          at least eight intermediate generations, and the last generation.
        - An empty line.
        - The final maximum value (a single value), equal to the last value of the block above.

    Example input - a knapsack of 15 kg and 5 items:

    ```
    15 5
    12 4
    2 2
    1 2
    1 1
    4 10
    ```

    The best choice is every item except the first one: weight $2 + 1 + 1 + 4 = 8 \le 15$ and value
    $2 + 2 + 1 + 10 = 15$, so the final maximum value should be `15`.

    ### Knapsack as a genetic algorithm

    - **Chromosome**: a binary string with one bit per item - 1 = take it, 0 = leave it. E.g. `01011` = take items 2, 4
      and 5.
    - **Fitness**: the total value of the chosen items - but what if they are too heavy? Some common options:
        - **death penalty**: an overweight knapsack gets fitness 0
        - **penalty**: subtract something proportional to the excess weight, so "slightly too heavy" is better than
          "much too heavy"
        - **repair**: remove items until the knapsack fits
    - **Crossover**: one-point, two-point or uniform - any combination of bits is a valid chromosome (the weight limit is
      handled by the fitness or by the repair).
    - **Mutation**: bit flip with probability $\frac{1}{N}$.

    ### Tactics to try

    - **Initialization**: if the items are heavy compared to $M$, a random bit string (every item taken with probability
      $\frac{1}{2}$) is almost always overweight. With the death penalty the whole first population then has fitness 0,
      and selection has nothing to work with. Take every item with a smaller probability, or repair the first population.
    - **Repair greedily**: remove the items with the lowest value / weight ratio first, until the knapsack fits. This
      usually works better than removing random items.
    - **Use elitism**: the best value then never decreases, so the printed values never go down and the final maximum is
      the last printed value - exactly what the output format asks for.
    - **Printing**: for at least eight intermediate values, print the best value every $\frac{1}{10}$ of the
      generations, plus the first and the last generation.
    - **Watch the diversity** and choose a **stopping** rule - the same as for TSP.
    """)
    return


if __name__ == "__main__":
    app.run()
