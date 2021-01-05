from utils import options
from utils.smtlib import *

NAME = 'core'
MUTATORS = [
    'constants', 'erase-children', 'merge-children', 'replace-by-variable',
    'sort-children', 'substitute-children', 'top-level-binary-reduction'
]


class Constants:
    """Replaces any node by a constant."""
    def mutations(self, node):
        """Return :code:`get_constants(get_return_type(node))`."""
        res = get_constants(get_return_type(node))
        if node in res:
            return []
        return res

    def __str__(self):
        return 'substitute by a constant'


class EraseChildren:
    """Erases a single child of the given node."""
    def filter(self, node):
        return True  # not is_leaf(node)

    def mutations(self, node):
        return [None]
        res = []
        for i in range(len(node)):
            tmp = list(node)
            del tmp[i]
            res.append(tuple(tmp))
        return res

    def __str__(self):
        return 'erase child'


class MergeWithChildren:
    """Merges a node with one of its children.

    This is possible for n-ary operators like :code:`and` or :code:`+`.
    """
    def filter(self, node):
        return is_nary(node)

    def mutations(self, node):
        res = []
        for cid, child in enumerate(node):
            if has_name(child) and get_name(node) == get_name(child):
                res.append(node[:cid] + node[cid][1:] + node[cid + 1:])
        return res

    def __str__(self):
        return 'merge with child'


class ReplaceByVariable:
    """Replaces a node by a variable."""
    def filter(self, node):
        return not is_constant(node)

    def mutations(self, node):
        ret_type = get_return_type(node)
        if ret_type is None:
            return []
        variables = get_variables_with_type(ret_type)
        if is_leaf(node):
            if options.args().replace_by_variable_mode == 'inc':
                return [v for v in variables if v > node]
            return [v for v in variables if v < node]
        return [v for v in variables if node_count(v) < node_count(node)]

    def __str__(self):
        return 'substitute by existing variable'


class SortChildren:
    """Sorts the children of a node."""
    def filter(self, node):
        return not is_leaf(node)

    def mutations(self, node):
        """Return :code:`sorted(node, key = node_count)`."""
        s = sorted(node, key=node_count)
        if s != node:
            return [s]
        return []

    def __str__(self):
        return 'sort children'


class SubstituteChildren:
    """Substitutes a node with one of its children."""
    def filter(self, node):
        return not is_leaf(node) and not is_operator(node, 'let')

    def mutations(self, node):
        return node[1:]

    def __str__(self):
        return 'substitute with child'


class TopLevelBinaryReduction:
    """Performs binary reduction on the top level node.

    Essentially mimics line based reduction.
    """
    def global_mutations(self, linput, ginput):
        if linput != ginput:
            return []
        # generate all sublists as generated by binary-search in bfs order
        # let den be the denominator of the list length (the tree level)
        # let num be the numerator within the current tree level
        res = []
        den = 2
        while den < len(ginput):
            for num in range(0, den):
                start = int(num / den * len(ginput))
                end = int((num + 1) / den * len(ginput))
                res.append(ginput[:start] + ginput[end:])
            den *= 2
        return res

    def __str__(self):
        return 'binary reduction'


def collect_mutator_options(argparser):
    options.add_mutator_argument(argparser, NAME, True, 'core mutators')
    options.add_mutator_argument(argparser, 'constants', True,
                                 'replace by theory constants')
    options.add_mutator_argument(argparser, 'erase-children', True,
                                 'erase individual children of nodes')
    options.add_mutator_argument(argparser, 'merge-children', True,
                                 'merge children into nodes')
    options.add_mutator_argument(argparser, 'replace-by-variable', True,
                                 'replace with existing variable')
    argparser.add_argument(
        '--replace-by-variable-mode',
        choices=['inc', 'dec'],
        default='inc',
        help='replace with existing variables that are larger or smaller')
    options.add_mutator_argument(argparser, 'sort-children', True,
                                 'sort children of nodes')
    options.add_mutator_argument(argparser, 'substitute-children', True,
                                 'substitute nodes with their children')
    options.add_mutator_argument(argparser, 'top-level-binary-reduction', True,
                                 'use top level binary reduction')


def collect_mutators(args):
    res = []
    if args.mutator_core:
        #if args.mutator_top_level_binary_reduction:
        #    res.append(TopLevelBinaryReduction())
        if args.mutator_erase_children:
            res.append(EraseChildren())
        #if args.mutator_constants:
        #    res.append(Constants())
        #if args.mutator_merge_children:
        #    res.append(MergeWithChildren())
        #if args.mutator_replace_by_variable:
        #    res.append(ReplaceByVariable())
        #if args.mutator_sort_children:
        #    res.append(SortChildren())
        #if args.mutator_substitute_children:
        #    res.append(SubstituteChildren())
    return res
