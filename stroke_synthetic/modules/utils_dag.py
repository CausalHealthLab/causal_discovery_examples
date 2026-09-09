"""Helper functions for producing and comparing DAGs"""
# Imports
import warnings
from matplotlib.lines import Line2D  # for custom legend
import matplotlib.pyplot as plt
import networkx as nx

# Functions

# #################
# ##### Edges #####
# #################
def categorise_edges_vs_true(edges_disc: list, edges_true: list):
    """
    Decide whether found edges are correct, incorrect, or missing.

    Correct edges are in both lists. Incorrect edges are in only the
    new list, and missing edges are in only the true list.
    Each edge is a tuple (start, end) to show direction.

    Inputs
    ------
    edges_disc - list. Discovered edges.
    edges_true - list. Expected edges.

    Returns
    -------
    d - dict. Dictionary of correct, incorrect, and missing edges.
        Each entry is a list of tuples, one tuple per edge.
    """
    # Store results in this dict:
    d = {}
    d['correct'] = list(set(edges_true) & set(edges_disc))
    d['incorrect'] = list(set(edges_disc) - set(edges_true))
    d['missing'] = list(set(edges_true) - set(edges_disc))
    return d


# #########################
# ##### Plotting DAGs #####
# #########################
def make_legend_entry_for_line(edge_format_dict: dict):
    """
    Make line from input format dictionary for a legend entry.

    The format dictionary is passed to Line2D().
    Example line format dictionary:
    dict(color='k', lw=1, label='Correct', linestyle=None)

    Make a legend of result using: ax.legend(handles=[legend_element])

    Inputs
    ------
    edge_format_dict - dict. Contains formatting for the line.

    Returns
    -------
    legend_element - matplotlib artist. A Line2D instance to match the
                     input formatting.
    """
    legend_element = Line2D([0], [0], **edge_format_dict)
    return legend_element


def make_formatting_for_edges(edges_dict: dict, edge_kw: dict={}):
    """
    Set up formatting for edges to be drawn on a DAG.

    For each type of formatting, create a single list that combines all
    edge categories. These lists can be used in a networkx plot.
    Returns a dict of single lists. For example, if
    there are 10 correct, 5 incorrect, and 2 missing lines, then each
    formatting list in the dict must be 17 items long.

    Expect edges_dict and edge_kw to have the same keys. If there are
    entries in edges_dict that are missing in edge_kw, a warning is
    raised.

    Example inputs:
    edges_dict = {'correct': [('a', 'b'), ('b', 'c')],
                  'incorrect': [('a', 'd')]}
    edge_kw = {'correct': dict(color='k'),
               'incorrect': dict(color='red')}

    Inputs
    ------
    edges_dict - dict. Edges by category. For example, keys could be
                 'correct', 'incorrect', and 'missing'. Values are
                 lists of tuples, one tuple per edge
                 (start node, end node).
    edge_kw    - dict. Formatting for each category of edges. Each
                 value is a dict of format kwargs, for example color
                 or linewidth. 

    Returns
    -------
    d - dict. Each value is a list of the same length. Keys are "edges"
        for a list of edge tuples, and then one key per format kwarg
        in the edge_kw dict of dicts.
    """
    # Sanity check - do the keys of both input dicts match?
    # Expect to see one set of formatting for each type of edge.
    # If an edge type is missing, use None and display a warning.
    edge_types_missing_formatting = set(edges_dict) - set(edge_kw)
    if len(edge_types_missing_formatting) > 0:
        # Which edges are missing?
        m = ', '.join(edge_types_missing_formatting)
        warnings.warn(f'Missing formatting for edge types: {m}')
    
    # Sanity check - are all formatting keywords in all dictionaries?
    # If not then explicitly set missing keys to have value None.
    all_edge_kw = list(set(
        sum([list(d.keys()) for d in edge_kw.values()], [])))
    # Make new dict with no missing values:
    edge_kw_full = {}
    for k, d in edge_kw.items():
        edge_kw_full[k] = d
        missing_keys = [k for k in all_edge_kw if k not in d.keys()]
        for m in missing_keys:
            edge_kw_full[k][m] = None
    edge_kw = edge_kw_full

    # Place results in here:
    d = {}
    # Gather all edges and line formatting using the edge dict values
    # and keys to make sure each list is gathered in the same order.
    # Use sum(lists to join, []) with the extra empty list to make sure
    # we end up with a flat list.
    d['edges'] = sum(list(edges_dict.values()), [])
    # Lists of the same format x times for edge list of length x:
    for kw in all_edge_kw:
        d[kw] = sum([[edge_kw[k][kw]] * len(edges_dict[k])
                     for k in edges_dict.keys()], [])
    return d


def plot_comparative_dag_networkx(
        G_disc: networkx.classes.digraph.DiGraph,
        edges_dict: dict,
        pos: dict,
        edge_format_dict: dict={},
        ax: matplotlib.axes.Axes=None,
        title: str='',
        savename: str='',
        show: bool=True
        ):
    """
    Plot a DAG using networkx with fancier arrow formatting.
    
    Inputs
    ------
    G_disc             - networkx.classes.digraph.DiGraph. Graph with
                         all nodes present.
    edges_dict         - dict. Each key is a category of edge, each
                         value is a list of tuples, one tuple per
                         edge (start node, end node).
    pos                - dict. x, y coordinates for node centre for
                         each feature.
    all_features       - list. All of the features to be shown on the
                         x and y axes of the plot.
    edge_format_dict   - dict. Formatting for the lines for each
                         category of edges.
    ax                 - matplotlib.axes.Axes. Axis object. If not
                         given, a new fig and ax is created.
    title              - str. Title for this axis.
    savename           - str. Saves the plot to this location.
    show               - bool. Whether to show or close fig.
    """
    # --- Setup for plot ---
    # Pick out coordinates of most extreme nodes for axis limits:
    node_x_min = min([p[0] for p in pos.values()])
    node_x_max = max([p[0] for p in pos.values()])
    node_y_min = min([p[1] for p in pos.values()])
    node_y_max = max([p[1] for p in pos.values()])
    
    # Set up appearance of effect arrows.
    # Default appearance:
    edge_kw = {
        'correct': dict(color='k', lw=1, label='Correct', linestyle=None),
        'incorrect': dict(color='red', lw=1, label='Incorrect',
                          linestyle=None),
        'missing': dict(color='grey', lw=1, label='Missing', linestyle=':'),
    }
    # Update with user inputs:
    for k, line_dict in edge_format_dict.items():
        if k in edge_kw.keys():
            edge_kw[k] = line_dict | edge_kw[k]
        else:
            edge_kw[k] = line_dict
    # If no formatting is given for an edge category, leave blank:
    missing_keys = [k for k in edges_dict.keys() if k not in edge_kw.keys()]
    for k in missing_keys:
        edge_kw[k] = dict()

    # Set up contents for a legend for these lines:
    legend_elements = [make_legend_entry_for_line(d)
                       for d in edge_kw.values()]

    # For each edge in turn, set up the formatting.
    edge_plot_dict = make_formatting_for_edges(edges_dict, edge_kw)

    # --- Begin plotting! ---
    # Use input axis if given, or create a new one.
    if ax is None:
        fig, ax = plt.subplots()
    else:
        pass
    # Create graph:
    nx.draw_networkx(
        G_disc,
        pos,
        ax=ax,
        with_labels=True,
        arrows=True,
        edgelist=edge_plot_dict['edges'],
        edge_color=edge_plot_dict['color'],
        style=edge_plot_dict['linestyle'],
        node_size=2200,
        font_size=11,
        arrowsize=18,
    )
    
    # Modify axis limits for breathing room around labels:
    ax.set_xlim(node_x_min-1, node_x_max+1)
    # ax.set_ylim(node_y_min-1, node_y_max+1)
    # Remove border around ax:
    for s in ['top', 'bottom', 'left', 'right']:
        ax.spines[s].set_visible(False)
    # Legend for arrow formatting:
    ax.legend(handles=legend_elements, bbox_to_anchor=(1, 1),
              loc='upper left')

    if len(title) > 0:
        ax.set_title(title)
    else:
        pass

    # Save or display the graph as requested:
    if len(savename) > 0:
        plt.savefig(savename, bbox_inches='tight')
    else:
        pass
    if show:
        plt.show()
    else:
        plt.close()


# ##################################
# ##### Plotting edge matrices #####
# ##################################
def plot_matrix_edge_comparison(
        edges_dict: dict,
        all_features: list,
        dict_marker_format: dict={},
        ax: matplotlib.axes.Axes=None,
        title: str='',
        savename: str='',
        show: bool=True
    ):
    """
    Draw scatter on grid to show where features are linked by edges.

    Features on the y-axis cause features on the x-axis.
    
    If edges_dict has multiple keys then the edge categories will
    have different marker formatting.
    The keys of edges_dict and dict_marker_format should match.
    If no formatting options are given for an edge category, then
    defaults are used.

    Inputs
    ------
    edges_dict         - dict. Each key is a category of edge, each
                         value is a list of tuples, one tuple per
                         edge (start node, end node). Node names must
                         match entries in all_features.
    all_features       - list. All of the features to be shown on the
                         x and y axes of the plot.
    dict_marker_format - dict. Formatting for the markers for each
                         category of edges.
    ax                 - matplotlib.axes.Axes. Axis object. If not
                         given, a new fig and ax is created.
    title              - str. Title for this axis.
    savename           - str. Saves the plot to this location.
    show               - bool. Whether to show or close fig.
    """
    # Default marker format dict:
    # zorder ensures that markers are drawn on top of grid.
    d = {
        'correct': dict(color='g', marker='o', s=70,
                        label='Correct', zorder=5),
        'incorrect': dict(color='r', marker='X', s=100,
                          label='Incorrect', zorder=5),
        'missing': dict(color='grey', marker='s', s=70,
                        label='Missing', zorder=5),
    }
    # Update with user inputs:
    for k, marker_dict in dict_marker_format.items():
        if k in d.keys():
            d[k] = marker_dict | d[k]
        else:
            d[k] = marker_dict
    # If no formatting is given for an edge category, leave blank:
    missing_keys = [k for k in edges_dict.keys() if k not in d.keys()]
    for k in missing_keys:
        d[k] = dict()

    # Use input axis if given, or create a new one.
    if ax is None:
        fig, ax = plt.subplots()
    else:
        pass

    # For each group of edges in turn, find coordinates from the
    # list of features and then plot the markers.
    for edge_type, edge_list in edges_dict.items():
        # Find coordinates based on feature names:
        cols = [all_features.index(edge[1]) for edge in edge_list]
        rows = [all_features.index(edge[0]) for edge in edge_list]
        # Draw with formatting from marker dict.
        ax.scatter(cols, rows, **d[edge_type])

    # Set up tick labels:
    n_feat = len(all_features)
    ax.set_xticks(range(n_feat))
    ax.set_xticklabels(all_features, rotation=80, ha='right')
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(all_features)
    ax.set_xlim(-0.5, n_feat-0.5)
    ax.set_ylim(-0.5, n_feat-0.5)
    
    ax.legend(bbox_to_anchor=(-0.1, -0.1), loc='upper right')
    ax.grid('on')
    ax.set_aspect('equal')
    if len(title) > 0:
        ax.set_title(title)
    else:
        pass

    # Save or display the graph as requested:
    if len(savename) > 0:
        plt.savefig(savename, bbox_inches='tight')
    else:
        pass
    if show:
        plt.show()
    else:
        plt.close()