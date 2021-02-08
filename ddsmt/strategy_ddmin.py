#
# ddSMT: A delta debugger for SMT benchmarks in SMT-Lib v2 format.
#
# This file is part of ddSMT.
#
# Copyright (C) 2013-2021 by the authors listed in the AUTHORS file.
#
# ddSMT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ddSMT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ddSMT.  If not, see <https://www.gnu.org/licenses/>.

import multiprocessing
import multiprocessing.pool
import collections
import logging
import pickle
import sys
import time
import traceback

from . import checker
from . import mutators
from . import nodeio
from . import nodes
from . import options
from . import smtlib
from .mutator_utils import Simplification, apply_simp

Task = collections.namedtuple('Task', ['id', 'exprs', 'simplifications'])

Result = collections.namedtuple(
    'Result', ['task_id', 'success', 'reduced', 'exprs', 'tests'])


def _partition(exprs, gran):
    """Partition ``exprs`` into a list of subsets of size ``gran``."""
    return [exprs[s:s + gran] for s in range(0, len(exprs), gran)]


class TaskGenerator:
    """Filter ``exprs`` based on ``mutator`` and generates tasks."""
    def __init__(self, exprs, gran, mutator, max_depth=None):
        self.exprs = exprs
        self.mutator = mutator
        self.max_depth = max_depth
        self.index = 0
        self.stopped = False

        # Filter nodes and partition into subsets of size ``gran``.
        filter_func = getattr(mutator, 'filter', lambda x: True)
        filtered = list(nodes.filter_nodes(exprs, filter_func, max_depth))
        self.num_filtered = len(filtered)
        self.gran = len(filtered) if gran is None else gran
        self.subsets = _partition(filtered, self.gran) if self.gran else []

        njobs = options.args().jobs
        if njobs > 1 and len(self.subsets) > 2 * njobs:
            self.pickled_exprs = pickle.dumps(exprs)
        else:
            self.pickled_exprs = None

    def __iter__(self):
        return self

    def __next__(self):
        """Generate next task."""
        while not self.stopped and self.index < len(self.subsets):
            task_id = self.index
            self.index += 1

            # Filter nodes in subset in order to ensure that the mutator still
            # applies after updating ``self.exprs`` via ``self.update``.
            subset = self.subsets[task_id]
            subset = [n for n in subset if self.mutator.filter(n)]
            if not subset:
                continue

            simps = self.__get_substs(subset)

            if not simps:
                continue

            logging.debug(f'TaskGen: Generate next {task_id}')
            if self.pickled_exprs:
                return Task(task_id, self.pickled_exprs, pickle.dumps(simps))
            return Task(task_id, self.exprs, simps)
        raise StopIteration

    def __get_substs(self, subset):
        """Generate substitutions for ``subset`` based on ``self.mutator``."""
        # Granularity 1: Try all mutations separately.
        if len(subset) == 1:
            node = subset[0]
            if hasattr(self.mutator, 'mutations'):
                mutations = self.mutator.mutations(node)
            elif hasattr(self.mutator, 'global_mutations'):
                mutations = self.mutator.global_mutations(node, self.exprs)
            else:
                return None
            return list(mutations)

        # Granularity > 1: Pick first simplification for each node and group
        # them all into one.
        fresh_vars = []
        substs = dict()
        for node in subset:
            if hasattr(self.mutator, 'mutations'):
                mutations = self.mutator.mutations(node)
            elif hasattr(self.mutator, 'global_mutations'):
                mutations = self.mutator.global_mutations(node, self.exprs)
            else:
                continue

            try:
                simp = next(iter(mutations))
                assert isinstance(simp, Simplification)
                fresh_vars.extend(simp.fresh_vars)
                substs.update(simp.substs)
            except StopIteration:
                continue

        if not substs:
            return None
        return [Simplification(substs, fresh_vars)]

    def reset(self, index):
        """Reset ``self.index`` to ``index``."""
        logging.debug(f'TaskGen: Reset to {index}')
        self.index = index

    def update(self, exprs):
        """Update ``self.exprs`` with new ``exprs``."""
        self.exprs = exprs
        if self.pickled_exprs is not None:
            self.pickled_exprs = pickle.dumps(exprs)

    def stop(self):
        """Stop generating new taks."""
        self.stopped = True

    def start(self):
        """Continue generating new taks."""
        self.stopped = False


def ddmin_passes():
    """Return list of ddmin passes.

    Mutators that apply global_mutations only are disabled since this is
    not supported by ddmin.
    """

    # Passes applied to top-level nodes (DFS max-depth 1)
    stage1_names = ['EraseNode', 'CheckSatAssuming']

    stage1 = mutators.get_initialized_mutator('EraseNode', {'ident': 'assert'})
    stage1.extend(mutators.get_mutators(stage1_names))

    # Use mutators that promise maximum reduction first
    stage2_names = [
        'Constants',
        'ReplaceByChild',
        'IntroduceFreshVariable',
        'LetElimination',
        'LetSubstitution',
        'ReplaceByVariable',
        'MergeWithChildren',
    ]

    exclude = [
        'BinaryReduction',
    ]
    exclude.extend(stage1_names)
    exclude.extend(stage2_names)

    # Add remaining mutators
    for theory in mutators.get_all_mutators().values():
        stage2_names.extend(x for x in theory[1] if x not in exclude)

    return [stage1, mutators.get_mutators(stage2_names)]


def _simp(exprs, simplifications):
    """Apply ``simplifications`` to ``exprs`` and return simplified
    formulas."""
    for simp in simplifications:
        mexprs = apply_simp(exprs, simp)
        if mexprs is not None:
            yield mexprs


__cached_exprs = None
__cached_exprs_hash = None
__abort_flag = None


def _worker(task):
    """Process given ``task``.

    If _worker runs in a separate process ``task.exprs`` and
    ``task.simplifications`` are pickled and need to be unpickled before
    performing the substitutions and checks.
    """
    global __cached_exprs
    global __cached_exprs_hash
    global __abort_flag

    try:
        if __abort_flag and __abort_flag.is_set():
            logging.debug(f'Worker: Abort task {task.id}')
            return Result(task.id, False, 0, [], 0)

        if isinstance(task.exprs, bytes):
            hashval = hash(task.exprs)
            if __cached_exprs_hash != hashval:
                __cached_exprs = pickle.loads(task.exprs)
                __cached_exprs_hash = hashval

            exprs = __cached_exprs
            substs = pickle.loads(task.simplifications)
        else:
            exprs = task.exprs
            substs = task.simplifications

        ntests = 0
        for mexprs in _simp(exprs, substs):
            ntests += 1
            if checker.check_exprs(mexprs):
                nreduced = nodes.count_exprs(exprs) - nodes.count_exprs(mexprs)
                return Result(task.id, True, nreduced, mexprs, ntests)
        return Result(task.id, False, 0, [], ntests)
    except Exception as e:
        logging.info(f'{type(e)} in ddmin worker: {e}')
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_tb(exc_traceback, limit=10, file=sys.stderr)


__last_msg = ""


def _print_progress(msg, update=True):
    """Print progress ``msg``.

    If ``update`` is True the previous line will be overwritten.
    """
    global __last_msg

    if options.args().verbosity < 1:
        return

    if sys.stderr.isatty():
        print(' ' * len(__last_msg), end='\r', flush=True, file=sys.stderr)
    if update and sys.stderr.isatty():
        __last_msg = f'[ddSMT INFO] {msg}'
        print(__last_msg, end='\r', flush=True, file=sys.stderr)
    else:
        logging.info(msg)
        __last_msg = msg


def _check_seq(taskgen, nexprs, stats):
    """Sequentially process tasks generated by ``taskgen``."""

    outfile = options.args().outfile

    for task in taskgen:
        result = _worker(task)
        stats['tests'] += result.tests

        if result.success:
            stats['tests_success'] += 1
            stats['reduced'] += result.reduced
            taskgen.update(result.exprs)
            nodeio.write_smtlib_to_file(outfile, taskgen.exprs)
            smtlib.collect_information(taskgen.exprs)

        _print_progress(
            f"{taskgen.mutator}: " \
            f"nodes: {taskgen.num_filtered}, " \
            f"gran: {taskgen.gran}, " \
            f"subset {task.id} of {len(taskgen.subsets)}, " \
            f"exprs: {nexprs - stats['reduced']}/{nexprs}",
            options.args().verbosity == 1)

    return taskgen.exprs


def _check_par(taskgen, nexprs, stats):
    """Process tasks generated by ``taskgen`` with multiple processes.

    As soon as one process performs a successful check with task N the
    main process stops the task generator and notifies all worker
    processes to stop ASAP. ``taskgen.exprs`` will be updated with the
    reduced expressions and ``taskgen`` is reset to start with task N+1.
    """
    global __abort_flag

    outfile = options.args().outfile

    __abort_flag = multiprocessing.Manager().Event()

    start_index = 0
    #with multiprocessing.Pool(options.args().jobs) as pool:
    with multiprocessing.pool.ThreadPool(options.args().jobs) as pool:
        while start_index >= 0:
            start_index = -1
            skip = False
            for result in pool.imap_unordered(_worker, taskgen):
                stats['tests'] += result.tests

                if result.success and not skip:
                    __abort_flag.set()
                    logging.debug(f'Main: Set abort flag')
                    taskgen.stop()
                    taskgen.update(result.exprs)
                    nodeio.write_smtlib_to_file(outfile, taskgen.exprs)
                    smtlib.collect_information(taskgen.exprs)
                    stats['tests_success'] += 1
                    stats['reduced'] += result.reduced
                    start_index = result.task_id + 1
                    skip = True
                    logging.debug(
                        f'Successful test with subset {result.task_id}: ' \
                        f'{result.reduced}')
                elif result.success and skip:
                    logging.debug(
                        f'Skip test with subset {result.task_id}: ' \
                        f'{result.reduced}')

                _print_progress(
                    f"{taskgen.mutator}: " \
                    f"nodes: {taskgen.num_filtered}, " \
                    f"gran: {taskgen.gran}, " \
                    f"subset {result.task_id} of {len(taskgen.subsets)}, " \
                    f"exprs: {nexprs - stats['reduced']}/{nexprs}",
                    options.args().verbosity == 1)

            if __abort_flag.is_set():
                __abort_flag.clear()
                taskgen.reset(start_index)
                taskgen.start()
                logging.debug(f'Restart tests starting from {start_index}')

    return taskgen.exprs


def _apply_mutator(mutator, exprs, max_depth=None):
    """Apply ``mutator`` with strategy ddmin on input ``exprs``.

    ``max_depth`` limits the DFS traversal when filtering nodes in
    ``exprs``.
    """

    start_time = time.time()
    nexprs = nodes.count_exprs(exprs)
    stats = {'tests': 0, 'tests_success': 0, 'reduced': 0}
    taskgen = TaskGenerator(exprs, None, mutator, max_depth)
    gran = taskgen.gran
    while gran > 0:
        check_func = _check_par if taskgen.pickled_exprs else _check_seq
        exprs = check_func(taskgen, nexprs, stats)
        exprs = nodes.reduplicate(exprs)
        gran = gran // 2
        taskgen = TaskGenerator(exprs, gran, mutator, max_depth)

    if stats['tests'] > 0 or options.args().verbosity >= 2:
        _print_progress(f"{mutator}: diff {-stats['reduced']:+} exprs, " \
                        f"{stats['tests']} tests ({stats['tests_success']}), " \
                        f"{time.time() - start_time:.1f}s", False)

    return exprs, stats['tests'], stats['reduced']


def reduce(exprs):
    """Reduce given ``exprs`` until fixed-point with ddmin strategy."""

    smtlib.collect_information(exprs)

    passes = ddmin_passes()

    ntests_total = 0
    while True:
        nreduced_round = 0

        # Apply top-level passes until fixed-point.
        for mut in passes[0]:
            while True:
                exprs, ntests, nreduced = _apply_mutator(mut, exprs, 1)
                ntests_total += ntests
                nreduced_round += nreduced

                if nreduced == 0:
                    break

        for mut in passes[1]:
            exprs, ntests, nreduced = _apply_mutator(mut, exprs)
            ntests_total += ntests
            nreduced_round += nreduced

        if nreduced_round == 0:
            break

    return exprs, ntests_total
