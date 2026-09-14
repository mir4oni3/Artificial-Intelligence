import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", app_title="Python for AI - Lecture 1")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Lecture 1 - Python Basics

    **Agenda**

    1. Jupyter Notebooks
    2. pip and virtual environments
    3. Core Python syntax, dynamic typing, duck typing, references instead of values
    5. Built-in data structures
    6. Writing Pythonic code
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 1. Notebooks

    ## 1.1 What is a notebook

    What you are currently looking at is called a notebook.

    A notebook is two things glued together:

    - a **document** made of cells, where each cell is either Markdown (text, formulas, images) or code
    - a **kernel**, which is just a Python process running in the background that never exits

    When you run a cell, the code is sent to that process, executed, and whatever it produced (printed
    text, a number, a plot, a table) is pasted back under the cell. The process stays alive, so every
    variable you created is still there when you run the next cell.


    Compare with what you are used to:

    | | C++ / Java | Python script | Notebook |
    |---|---|---|---|
    | Edit-run cycle | edit, compile, link, run | edit, run | run one cell |
    | State after running | gone | gone | **still in memory** |
    | Reload a 2 GB dataset to fix a typo in a plot | yes | yes | no |
    | Output | terminal text | terminal text | text, tables, plots, LaTeX, widgets |

    ## 1.2 Why notebooks are used

    1. **Prevent slow data loading and slow calculations.** You read a 500 MB CSV once,
       then spend an hour trying different things with the data against the copy sitting in RAM.
       With a script, every experiment loads the data again.
    2. **Plots appear inline.** ML is tied to tables, plots and other visual data to show results and insights.
       Notebooks allow all of these to be viewed inline, showing each output right under the cell.
    4. **Code plus explanation in one place.** Good for explaining different concepts in a single unified interactive place.

    ## 1.3 Where notebooks are bad
    - **Hidden state.** The kernel remembers everything, including variables from cells you have since
      deleted, and functions you have since edited. Your notebook can work perfectly while being
      completely broken for anyone else.
    - **Out-of-order execution.** You can run cell 10 before cell 3. The `[7]` next to a cell is the
      execution counter, not the position. If those numbers are not increasing top to bottom, be suspicious.
    - **Bad version control.** An `.ipynb` file is JSON with embedded outputs. Git diffs are unreadable
      and merge conflicts are miserable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.4 Running notebooks, option A: Google Colab

    Runs notebooks on Google servers. It is free for general ML use.

    1. Go to [colab.research.google.com](https://colab.research.google.com) and sign in with a Google account.
    2. **File - New notebook**. You now have a running Python 3 kernel with `numpy`,
       `pandas`, `matplotlib`, `scikit-learn`, `torch` and most of the ML stack preinstalled.


    ## 1.5 Running notebooks, option B: VS Code

    Run notebooks locally by using VS Code.

    1. Install [VS Code](https://code.visualstudio.com/).
    2. Install extensions: **Python** and **Jupyter**.
    3. Create a file `file.ipynb` and open it - you get a notebook editor.
    4. Top right, click **Select Kernel - Python Environments** and pick your interpreter (python runtime).
    5. The first time, VS Code will offer to `pip install ipykernel` - say yes.

    ## NOTE:
    We will be using notebooks during seminars, but you won't need to use them during this course. I am showing you what notebooks are so you
    can have a better understanding of the data science stack which is commonly used in practice, and to also make it possible for you to
    understand what I am doing during seminars. Homeworks and other tasks will require you to write straight code, preferably in Python, as scripts or other executable code.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 2. python, pip, virtual environments

    ## 2.1 Running Python

    ```bash
    python3 --version          # or: python --version

    python3 script.py          # run a script
    python3                    # start interactive terminal
    ```

    Python is an interpreted language, so it has no separate compile step.
    """)
    return


@app.cell
def _():
    # This is a cell that can be executed.

    # Show the Python environment information.
    import sys

    print("executable:", sys.executable)
    print("version:   ", sys.version.split()[0])
    print("platform:  ", sys.platform)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.2 `pip`: installing dependencies

    `pip` is Python's package manager. It downloads packages and installs them into the Python environment you are currently using.
    Usage: `pip install <name>`.

    ### The commands you need

    ```bash
    pip install numpy                    # latest version
    pip install numpy pandas matplotlib  # several at once
    pip uninstall numpy

    pip show numpy                       # version, location, dependencies
    pip freeze                           # show everything installed
    ```


    ### Installing from inside a notebook in Colab

    In Colab:

    ```python
    # `%pip install networkx`
    ```

    After installing you may need to restart the kernel
    before the import works.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.3 Virtual environments

    ### The problem

    Using one system-wide Python and one set of installed packages leads to problems. For example different
    projects might need different versions of a package, and only one version can be installed at once.
    Also, sooner or later, you might lose track of what is installed and what is not, making reproducing code harder.
    This also leads to a lot of packages, which might start conflicting with each other.

    ### The solution

    A **virtual environment(venv)** is a folder containing its own Python interpreter and its own installed dependencies.
    A virtual environment can be activated, and then `python` and `pip` refer to the environment python instead of the system python. Packages installed inside it are invisible outside it and vice versa. It is great for segregating different projects, for reproducibility, and for tracking what is installed in different environments. If something goes wrong, you can always delete the venv directory and rebuild it in 30 seconds.

    ### Creating and using one

    ```bash
    cd ~/projects/ai-course        # go to your project
    python3 -m venv .venv          # create the venv; ".venv" is the conventional name
    ```

    This makes a `.venv/` directory. Now you can activate it:

    ```bash
    # Linux / macOS / WSL (bash, zsh)
    source .venv/bin/activate

    # Windows PowerShell
    .venv\Scripts\Activate.ps1

    # Windows cmd.exe
    .venv\Scripts\activate.bat
    ```

    Your prompt gains a `(.venv)` prefix. Verify:

    ```bash
    which python        # Linux/macOS  ->  /home/you/projects/ai-course/.venv/bin/python
    where python        # Windows      ->  ...\ai-course\.venv\Scripts\python.exe
    ```

    Now you can install things:

    ```bash
    pip install --upgrade pip
    pip install numpy pandas matplotlib scikit-learn
    ```

    When you are done:

    ```bash
    deactivate
    ```

    Activation lasts for that terminal session only. Open a new terminal, activate again. There is nothing
    to clean up: deleting `.venv/` fully removes the environment.

    ### Rules

    - Create the venv **inside** the project folder, name it `.venv`.
    - Add `.venv/` to `.gitignore`. Never commit it. It is large and machine-specific.
    - In VS Code, `Ctrl+Shift+P - Python: Select Interpreter` and choose the `.venv` one. Do this once per
      project and the terminal, the debugger, and the notebook kernel all use it.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.4 `requirements.txt`

    A plain text file listing the packages a project needs, one per line. This is how you tell another
    person what to install.

    ```text
    # requirements.txt
    numpy==2.1.3
    pandas==2.2.3
    matplotlib==3.9.2
    scikit-learn==1.5.2
    networkx==3.4.2
    ```

    Install everything in it:

    ```bash
    pip install -r requirements.txt
    ```

    Generate one from what you currently have installed:

    ```bash
    pip freeze > requirements.txt
    ```

    `freeze` dumps *every* installed package with exact versions, including transitive dependencies.


    ### The complete workflow for a new assignment

    ```bash
    mkdir ai-hw1
    cd ai-hw1
    python3 -m venv .venv
    source .venv/bin/activate                       # or the Windows line above
    pip install numpy matplotlib networkx
    pip freeze > requirements.txt
    echo ".venv/" >> .gitignore
    code .                                          # open in VS Code, select the .venv interpreter
    ```

    And for someone receiving your project:

    ```bash
    git clone <repo>
    cd <repo>
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 3. Python intro

    ## 3.1 Objects and names are not binded

    In C++ a variable is a named box of a fixed type at a fixed address. In Python:

    - a **name** is just a label
    - an **object** lives on the heap and has its own type
    - assignment binds a name to an object - it doesn't copy

    So `x = 5` does not mean "make an int slot called x and put 5 in it". It means "make (or reuse) the
    integer object 5, and make the name `x` point at it". The type belongs to the object, not to the name,
    so rebinding the name to something else of another type is legal.
    """)
    return


@app.cell
def _():
    value = 42
    print(value, type(value))

    value = "now I am a string"      # perfectly legal
    print(value, type(value))

    value = [1, 2, 3]
    print(value, type(value))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is what **dynamic typing** means - the type check happens when an operation is executed, not
    before. This means:

    > A Python program can contain a type error in a branch that never runs, and nothing will tell you.
    > Errors happen at the moment the line executes, not at build time.
    """)
    return


@app.cell
def _():
    def broken(flag):
        if flag:
            return 1 + 1
        else:
            return 1 + "one"     # a type error, sitting there quietly


    print(broken(True))          # fine

    try:
        broken(False)            # only now does it blow up
    except TypeError as err:
        print("TypeError:", err)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.2 Names are references. Assignment never copies.


    In C++, `std::vector<int> b = a;` copies the whole vector. In Python, `b = a` copies **nothing** - you
    now have two names for the same object. If the object is mutable, changing it through one name is
    visible through the other.
    """)
    return


@app.cell
def _():
    a_list = [1, 2, 3]
    b_list = a_list            # NOT a copy - same object, two names
    b_list.append(4)

    print("a_list:", a_list)   # [1, 2, 3, 4] - a_list changed too
    print("same object?", a_list is b_list) # `is` checks if two names refer to the same object
    print("ids:", id(a_list), id(b_list)) # every object has a unique id
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To actually copy, use the copy() function. Which kind you need depends on how deep the structure is:
    """)
    return


@app.cell
def _():
    original = [1, 2, 3]

    copied = original.copy()

    copied.append(99)
    print("original untouched:", original)
    print("copy modified:     ", copied)
    return


@app.cell
def _():
    import copy

    # Shallow copies only duplicate the outer container, nested objects are still shared.
    grid = [[0, 0], [0, 0]]
    shallow_grid = grid.copy()
    shallow_grid[0][0] = 7
    print("shallow copy shares the inner lists:", grid)

    grid2 = [[0, 0], [0, 0]]
    deep_grid = copy.deepcopy(grid2)     # recursively copies everything
    deep_grid[0][0] = 7
    print("deepcopy is independent:      ", grid2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is important and should be remembered, because this could lead to logical errors in code.

    ## 3.3 `is` versus `==`

    - `==` compares **values** - it can be predefined
    - `is` compares **identity** - check if two names point at the same object
    """)
    return


@app.cell
def _():
    list_x = [1, 2, 3]
    list_y = [1, 2, 3]

    print("list_x == list_y:", list_x == list_y)   # True: equal contents
    print("list_x is list_y:", list_x is list_y)   # False: two distinct objects
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.4 Mutable and immutable

    | Immutable (cannot be changed after creation) | Mutable (can be changed after creation) |
    |---|---|
    | `int`, `float`, `bool`, `str`, `tuple`, `frozenset`, `bytes` | `list`, `dict`, `set`, most objects you define |

    Only immutable objects can be dictionary keys or set elements.
    """)
    return


@app.cell
def _():
    text = "hello"
    # text[0] = "H"    # TypeError: `str` object does not support item assignment - this is because str is immutable
    text = "H" + text[1:]     # you make a new string instead
    print(text)

    coords_ok = {(0, 0), (1, 2)}          # a set of tuples - fine, because tuples are immutable
    print("set of tuples:", coords_ok)

    try:
        coords_bad = {[0, 0]}             # a set containing a list - not fine, because lists are mutable
    except TypeError as set_err:
        print("TypeError:", set_err)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.5 How arguments are passed

    Python passes references to objects, by value. In practice:

    - **mutating** an argument inside a function is visible to the caller
    - **rebinding** the parameter name inside a function is not

    There is no `&` parameter and no `const`. If you want a function not to modify its argument, either
    do not modify it, or copy it first.
    """)
    return


@app.cell
def _():
    def mutate(items):
        items.append("added")     # modifies the caller's list


    def rebind(items):
        items = ["completely", "new"]   # only rebinds the local name
        return items


    shopping = ["milk"]
    mutate(shopping)
    print("after mutate:", shopping)

    rebind(shopping)
    print("after rebind:", shopping)     # unchanged
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The mutable default argument

    Default argument values are evaluated **once**, when the `def` statement runs, not on each
    call. A mutable default is therefore shared between all calls.
    """)
    return


@app.cell
def _():
    def bad_collect(item, bucket=[]):        # DANGER
        bucket.append(item)
        return bucket


    print(bad_collect("a"))     # [`a`]
    print(bad_collect("b"))     # [`a`, `b`] - the same list
    print(bad_collect("c"))     # [`a`, `b`, `c`]
    return


@app.cell
def _():
    def good_collect(item, bucket=None):     # the correct pattern
        if bucket is None:
            bucket = []
        bucket.append(item)
        return bucket


    print(good_collect("a"))    # [`a`]
    print(good_collect("b"))    # [`b`]
    print(good_collect("c"))    # [`c`]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.6 Duck typing

    > If it walks like a duck and quacks like a duck, treat it as a duck.

    Python does not check whether an object is of a certain type. It checks, at the moment of use, whether
    the operation you asked for works.
    A function works on anything that supports the operations it uses.
    """)
    return


@app.cell
def _():
    def summarise(collection):
        # Works with anything that has a length and can be iterated - no type is ever specified.
        return f"{len(collection)} items, first is {next(iter(collection))!r}"


    print(summarise([1, 2, 3]))
    print(summarise((1, 2, 3)))
    print(summarise("abc"))
    print(summarise({"a": 1, "b": 2}))    # iterating a dict gives its keys
    print(summarise({7, 8, 9}))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.7 Indentation

    A colon `:` opens a block and indentation defines it. Four spaces per level is
    the universal convention.

    Empty blocks need the `pass` keyword, since you cannot write `{}`.

    ```python
    if condition:
        do_something()
        do_another_thing()
    else:
        pass          # a block that does nothing
    ```

    ## 3.8 Arithmetic
    """)
    return


@app.cell
def _():
    # Integers have unlimited precision - no overflow
    print(2 ** 200)
    print(len(str(2 ** 10000)), "digits, still exact")
    return


@app.cell
def _():
    # `/` is float division, even for two ints. Use `//` for integer division.
    print("7 / 2  =", 7 / 2)      # 3.5
    print("7 // 2 =", 7 // 2)     # 3
    print("7 % 2  =", 7 % 2)
    print("2 ** 10 =", 2 ** 10)   # power operator
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Other small differences:

    - There is **no `++` and no `--`**, use `i += 1` instead.
    - `True` and `False` are capitalised; `None` is the null. `and`, `or`, `not` replace `&&`, `||`, `!`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.9 Truthiness

    Any object can be used in a boolean context. The falsy values are: `False`, `None`, zero of any
    numeric type, and **every empty container**. Everything else is truthy.

    This is why Pythonic code writes `if items:` instead of `if len(items) > 0:`, and
    `while frontier:` instead of `while not frontier.empty():`.
    """)
    return


@app.cell
def _():
    frontier = []
    print("empty list is truthy?", bool(frontier))

    frontier = ["start"]
    while frontier:
        node = frontier.pop()
        print("expanding", node)

    for candidate in [0, 1, "", "x", [], [0], {}, None, 0.0]:
        print(f"{candidate!r:>6} -> {bool(candidate)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.10 Scope

    Scope units are **functions**, not blocks. A variable created inside an `if` or a `for`
      is visible after it, in the whole function.
    """)
    return


@app.cell
def _():
    def scope_demo():
        for step in range(3):
            last_seen = step          # both `step` and `last_seen` are function-scoped
        print("after the loop:", step, last_seen)

    scope_demo()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.11 Sequence slicing
    """)
    return


@app.cell
def _():
    # Slicing works on any sequence: [start:stop:step], stop is EXCLUSIVE, negatives count from the end.
    word = "artificial"
    print(word[0:3])      # `art`
    print(word[3:])       # from 3 to the end
    print(word[:3])       # from the start to 3
    print(word[-4:])      # last four
    print(word[::2])      # every second character
    print(word[::-1])     # reversed
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 4. Data structures

    ## 4.1 `list`

    A dynamic array. Heterogeneous (it can hold anything), indexable, sliceable, growable.
    """)
    return


@app.cell
def _():
    primes = [2, 3, 5, 7, 11]

    print(primes[0], primes[-1], primes[1:3])
    print(len(primes), sum(primes), max(primes), min(primes))

    primes.append(13)              # O(1) amortised, at the end
    primes.insert(0, 1)            # O(n), shifts everything
    primes.remove(1)               # removes the first occurrence by value, O(n)
    last_prime = primes.pop()      # O(1), removes and returns the last
    print(primes, "popped", last_prime)

    print("index of 7:", primes.index(7))
    print("11 in primes:", 11 in primes)      # O(n) linear scan
    print("count of 3:", primes.count(3))

    primes.reverse()
    print("reversed in place:", primes)
    primes.sort()
    print("sorted in place:  ", primes)
    print("sorted copy:      ", sorted(primes, reverse=True))
    return


@app.cell
def _():
    mixed = [1, "two", 3.0, [4], None]        # legal, though rarely a good idea
    print(mixed)

    nested = [[1, 2, 3], [4, 5, 6]]
    print(nested[1][2])

    zeros_row = [0] * 5
    print(zeros_row)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.2 `tuple`

    An immutable sequence. You can think of them as immutable equivelants of lists. Use it for a fixed-size group of values that belong together: a coordinate, an
    RGB colour, and so on.

    Tuples are **hashable** (as long as their contents are), so they can be
    dictionary keys and set elements(sets are explained later in this notebook).
    """)
    return


@app.cell
def _():
    position = (3, 4)
    row, column = position          # unpacking
    print(row, column)

    print(position[0], len(position))
    # position[0] = 9               # TypeError: tuples are immutable

    single = (42,)                  # a one element tuple NEEDS the trailing comma
    not_a_tuple = (42)              # this is just the integer 42 in parentheses
    print(type(single), type(not_a_tuple))

    visited_cells = {(0, 0), (1, 2)}    # tuples in a set - useful for tracking visited items in a graph
    visited_cells.add((3, 4))
    print((1, 2) in visited_cells)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.3 `dict`

    A hash map. Keys must be hashable (immutable); values can be anything. Since Python 3.7 dictionaries
    preserve insertion order, but do not rely on this functionality in production, since by computer science convention, hashmaps generally
    do not have an ordering, and it would lead to harder understanding of algorithm logic. It is also not compatible with older Python versions.
    """)
    return


@app.cell
def _():
    graph = {
        "A": ["B", "C"],
        "B": ["A", "D"],
        "C": ["A", "D"],
        "D": ["B", "C", "E"],
        "E": ["D"],
    }

    print(graph["A"])                     # KeyError if the key is missing
    print(graph.get("Z"))                 # None instead of an exception
    print(graph.get("Z", []))             # second argument is the default value if the key is missing

    graph["F"] = ["E"]                    # insert or overwrite
    print("F" in graph)                   # membership tests KEYS, in O(1)
    del graph["F"]

    print(list(graph.keys()))
    print(list(graph.values())[:2])
    for node_name, neighbours in graph.items():        # iterating pairs
        print(f"  {node_name} -> {neighbours}")
    return


@app.cell
def _():
    # Counting with a dict, the manual way
    sentence = "the quick brown fox jumps over the lazy dog the end"
    word_counts = {}
    for word_token in sentence.split():
        word_counts[word_token] = word_counts.get(word_token, 0) + 1 # get current count or 0 if not present, then increment and add to the dict
    print(word_counts)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Other useful moves:
    - `dict.update(other)` adds all entries of other into dict
    - `{**a, **b}` merges into a new dict, `dict.pop(key)` removes and returns the removed item
    - `dict.items()` - the dict items as a sequence of two-element tuples
    - `keys()` - a sequence of only the keys
    - `values()` - a sequence of only the values

    Never modify a dict (or a list, or a set) while iterating over it. Iterate over a copy of the keys
    instead: `for key in list(d):`.

    ## 4.4 `set`

    An unordered collection of unique, hashable elements. Membership testing is `O(1)`. In graph search
    this can be used for the `visited` structure. You can think of this as an implementation of mathematical sets - it doesn't contain duplicate items and has no ordering.
    """)
    return


@app.cell
def _():
    visited = set()                    # NOTE: {} is an empty DICT, not an empty set
    visited.add("A")
    visited.add("B")
    visited.add("A")                   # already there, no effect
    print(visited)

    print("A" in visited)              # O(1)
    visited.discard("Z")               # remove if present, no error
    visited.remove("B")                # KeyError if missing

    group_a = {1, 2, 3, 4}
    group_b = {3, 4, 5}
    print("union:       ", group_a | group_b)
    print("intersection:", group_a & group_b)
    print("difference:  ", group_a - group_b)
    print("symmetric:   ", group_a ^ group_b)
    print("subset?      ", {1, 2} <= group_a)

    print("dedupe a list:", set([1, 1, 2, 3, 3, 3]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.5 Comprehensions

    This is very important for writing 'Pythonic' code. A comprehension builds a collection from an
    iterable in one expression. It replaces the `create empty container; loop; append` pattern, and it is
    both shorter and faster (the loop runs in C).

    ```
    [ expression   for item in iterable   if condition ]
       what to      what to loop over      optional filter
       collect
    ```
    """)
    return


@app.cell
def _():
    numbers = range(10)

    squares = [n * n for n in numbers]
    print(squares)

    even_squares = [n * n for n in numbers if n % 2 == 0]
    print(even_squares)

    labels = ["even" if n % 2 == 0 else "odd" for n in numbers]   # ternary in the EXPRESSION slot
    print(labels)

    # Dict and set comprehensions use the same syntax with {}
    square_map = {n: n * n for n in range(5)}
    print(square_map)

    unique_lengths = {len(w) for w in ["a", "bb", "cc", "ddd"]}
    print(unique_lengths)

    # Nested loops: read them left to right, exactly as if they were nested for statements
    grid_cells = [(r, c) for r in range(3) for c in range(2)]
    print(grid_cells)

    matrix = [[1, 2], [3, 4], [5, 6]]
    flattened = [cell for row_values in matrix for cell in row_values]
    print(flattened)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If a comprehension needs more than two `for`/`if` clauses, or does not fit on one or two
    lines, write a normal loop. Readability is more important than being clever.
    """)
    return


@app.cell
def _():
    print(sum(n * n for n in range(1000)))               # no intermediate list is built
    print(any(n > 900 for n in range(1000)))             # stops at the first True
    print(all(n >= 0 for n in range(1000)))
    print(max((n % 7, n) for n in range(20)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.6 Sorting, `key=`, and the argmax pattern

    `sorted(iterable)` returns a new list; `list.sort()` sorts in place. Both accept `key=`, a function
    applied to each element to produce the value that is compared.
    """)
    return


@app.cell
def _():
    people = [("Ann", 31), ("Bob", 24), ("Cleo", 45), ("Dan", 24)]

    print(sorted(people))                                    # tuples compare element by element
    print(sorted(people, key=lambda person: person[1]))      # by age
    print(sorted(people, key=lambda person: -person[1]))     # by age, descending
    print(sorted(people, key=lambda person: (person[1], person[0])))   # age, then name

    words_to_sort = ["banana", "Fig", "apple", "cherry"]
    print(sorted(words_to_sort))                             # uppercase sorts before lowercase
    print(sorted(words_to_sort, key=str.lower))              # case-insensitive
    print(sorted(words_to_sort, key=len))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 5. Writing Pythonic code

    "Pythonic" code is not just for aesthetics, writing code this way is usually shorter, faster (the loop runs in
    C), and less prone to off-by-one errors.

    ## 5.1 Iterate over things, not over indices

    The `for (int i = 0; i < n; i++)` habit is the hardest one to drop, and dropping it removes a whole
    category of bugs.
    """)
    return


@app.cell
def _():
    temperatures = [12.5, 14.0, 9.8, 20.1]

    # What you would write coming from C++/Java:
    for i in range(len(temperatures)):
        print(i, temperatures[i], end="   ")
    print()

    # What you should write when you only need the values:
    for temperature in temperatures:
        print(temperature, end="   ")
    print()

    # When you need the index too: enumerate
    for i, temperature in enumerate(temperatures):
        print(f"{i} {temperature}", end="   ")
    print()

    for i, temperature in enumerate(temperatures, start=1):    # start the counter where you like
        print(f"{i} {temperature}", end="   ")
    print()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | Instead of | Write |
    |---|---|
    | `for i in range(len(xs)): xs[i]` | `for x in xs:` |
    | `i = 0` ... `i += 1` inside the loop | `for i, x in enumerate(xs):` |
    | `if len(xs) > 0:` | `if xs:` |
    | `if xs.count(v) > 0:` / manual search | `if v in xs:` |
    | building a list with `append` in a loop | a comprehension |
    | `tmp = a; a = b; b = tmp` | `a, b = b, a` |

    ## 5.2 Unpacking

    Any iterable can be unpacked into names. This is used constantly for coordinates, `(node, cost)`
    pairs, and multiple return values.
    """)
    return


@app.cell
def _():
    point_3d = (1, 2, 3)
    px, py, pz = point_3d
    print(px, py, pz)

    # Starred unpacking absorbs "the rest"
    head, *tail = [10, 20, 30, 40]
    print(head, tail)

    first, *middle, last = [1, 2, 3, 4, 5]
    print(first, middle, last)

    # Unpacking inside a loop header, over pairs
    path_costs = [("A", 1.0), ("B", 2.5), ("C", 0.4)]
    for step_name, step_cost in path_costs:
        print(step_name, step_cost)

    # Spreading a sequence into function arguments
    def volume(length, width, height):
        return length * width * height

    dimensions = (2, 3, 4)
    print(volume(*dimensions))                                   # * spreads a sequence
    print(volume(**{"length": 2, "width": 3, "height": 4}))      # ** spreads a dict
    return


if __name__ == "__main__":
    app.run()
