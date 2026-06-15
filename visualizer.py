"""
CityMind - Terminal Visualizer (visualizer.py)
==============================================
Prints the city grid and statistics to the console using ANSI colors.
No external libraries required.
"""

from city_model import LocationType, CityGraph

_C = {
    LocationType.RESIDENTIAL:     '\033[92m',
    LocationType.HOSPITAL:        '\033[91m',
    LocationType.SCHOOL:          '\033[94m',
    LocationType.INDUSTRIAL:      '\033[93m',
    LocationType.POWER_PLANT:     '\033[95m',
    LocationType.AMBULANCE_DEPOT: '\033[96m',
    LocationType.EMPTY:           '\033[90m',
}
_RESET = '\033[0m'
_SYM = {
    LocationType.RESIDENTIAL:     'RES',
    LocationType.HOSPITAL:        'HOS',
    LocationType.SCHOOL:          'SCH',
    LocationType.INDUSTRIAL:      'IND',
    LocationType.POWER_PLANT:     'PWR',
    LocationType.AMBULANCE_DEPOT: 'AMB',
    LocationType.EMPTY:           '...',
}


def print_grid(graph, title="City Grid"):
    """Print a colored ASCII grid of the city layout."""
    cols = graph.cols
    print("\n" + "=" * 62)
    print("  " + title)
    print("=" * 62)
    print("     ", end="")
    for c in range(cols):
        print(" {:3d}".format(c), end="")
    print()
    print("     " + "----" * cols)
    for r in range(graph.rows):
        print("  {:2d} |".format(r), end="")
        for c in range(cols):
            nid = graph.node_id(r, c)
            node = graph.get_node(nid)
            clr = _C.get(node.location_type, _RESET)
            sym = _SYM.get(node.location_type, '???')
            print("{}{}{}".format(clr, sym, _RESET), end=" ")
        print("|")
    print("     " + "----" * cols)
    print("\n  Legend:")
    for lt, sym in _SYM.items():
        if lt == LocationType.EMPTY:
            continue
        count = len(graph.get_nodes_of_type(lt))
        print("    {}{}{} {:18s} x{}".format(_C[lt], sym, _RESET, lt.name, count))


def print_statistics(graph, optimizer=None):
    """Print grid statistics and optional road network info."""
    total = graph.rows * graph.cols
    print("\n" + "=" * 62)
    print("  Grid Statistics  ({}x{} = {} cells)".format(graph.rows, graph.cols, total))
    print("=" * 62)
    for lt in LocationType:
        if lt == LocationType.EMPTY:
            continue
        count = len(graph.get_nodes_of_type(lt))
        pct = count / total * 100
        print("  {:18s}  {:3d}  ({:4.1f}%)".format(lt.name, count, pct))
    if optimizer:
        print("\n  Road Network:")
        print("    Total Roads Built: {}".format(len(optimizer.built_roads)))
        print("    Total Road Cost:   {:.1f}".format(optimizer.calculate_total_cost()))
        print("    Safety Route 1:    {} edges".format(len(optimizer.path1_edges)))
        print("    Safety Route 2:    {} edges".format(len(optimizer.path2_edges)))
