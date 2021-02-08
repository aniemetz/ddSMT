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

import collections
import logging
import multiprocessing
import multiprocessing.pool
import pickle
import sys
import time
import traceback

from . import checker
from . import mutators
from . import nodeio
from . import nodes
from . import options
from . import progress
from . import smtlib
from .mutator_utils import Simplification, apply_simp


def get_passes():
    """Returns a list of passes, each pass being a list of mutators.

    The list is ordered so that earlier passes are more promising for a
    quick reduction.
    """
    prelude = [
        (mutators.get_initialized_mutator('BinaryReduction',
                                          {'ident': 'assert'}), {
                                              'max_depth': 1
                                          }),
        mutators.get_mutators(['BinaryReduction']),
        mutators.get_mutators([
            'EraseNode',
            'Constants',
            'IntroduceFreshVariable',
            'LetElimination',
            'LetSubstitution',
            'ReplaceByChild',
            'BinaryReduction',
            'CheckSatAssuming',
        ]),
    ]

    late = [  # Usually only have cosmetic impact
        'SimplifyQuotedSymbols',
        'SimplifySymbolNames',
        'StringSimplifyConstant',
    ]
    main = []
    for _, theory in mutators.get_all_mutators().items():
        for mname in theory[1]:
            if mname not in late:
                main.append(mname)

    return prelude + [
        mutators.get_mutators(main),
        mutators.get_mutators(late + main),
    ]


def get_pass(passes, id):
    """Return the current pass, add empty parameter dict if none is given
    explicitly."""
    res = passes[id]
    if isinstance(res, tuple):
        return res
    return res, {}


# nodeid: id of the mutated node in bfs order. Only used for progress indication
# name: name of the mutator
# exprs: the current input
# simp: the substitution to be checked
# runtime: time needed to check this task
Task = collections.namedtuple('Task',
                              ['nodeid', 'name', 'exprs', 'simp', 'runtime'])


class Producer:
    """Manages the generation of candidates that shall be checked.

    Performs a walk through the current input and applies the
    ``mutators`` to every node. Supports skipping the first ``skip``
    nodes. As soon as ``abort_flag`` is triggered, stops generation as
    soon as possible.
    """
    def __init__(self, mutators, abort_flag, original):
        self.__node_count = 0
        self.__mutators = mutators
        self.__abort = abort_flag
        self.__original = original
        #self.__pickled = pickle.dumps(original)

    def __mutate_node(self, count, linput):
        """Apply all mutators to the given node.

        Returns a list of all possible mutations as ``Task`` objects.
        """
        for m in self.__mutators:
            if self.__abort.is_set():
                break
            try:
                if hasattr(m, 'filter') and not m.filter(linput):
                    continue
                if hasattr(m, 'mutations'):
                    for x in m.mutations(linput):
                        if self.__abort.is_set():
                            break
                        assert isinstance(x, Simplification)
                        yield Task(count, str(m), self.__original, x, None)
                if hasattr(m, 'global_mutations'):
                    for x in m.global_mutations(linput, self.__original):
                        if self.__abort.is_set():
                            break
                        assert isinstance(x, Simplification)
                        yield Task(count, f'(global) {m}', self.__original, x, None)
            except Exception as e:
                logging.info(f'{type(e)} in application of {m}: {e}')
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_tb(exc_traceback, limit=10, file=sys.stderr)

    def generate(self, skip, params):
        """A generator that produces all possible mutations as ``Task`` from
        the given original."""
        count = 0
        for node in nodes.bfs(self.__original, params.get('max_depth', None)):
            count += 1
            if skip < count:
                yield from self.__mutate_node(count, node)
            if self.__abort.is_set():
                break


class Consumer:
    """Calls the ``checker`` on individual tasks to figure out whether they are
    valid simplifications.

    Uses the ``abort_flag`` to stop as soon as a valid simplification
    has been found.
    """
    def __init__(self, abort_flag):
        self.__abort = abort_flag

    def check(self, task):
        abortres = (False,
                    Task(task.nodeid, task.name, None, None, None))
        if self.__abort.is_set():
            return abortres
        try:
            start = time.time()
            simp = task.simp
            assert isinstance(simp, Simplification)
            if self.__abort.is_set():
                return abortres
            exprs = apply_simp(task.exprs, simp)

            if self.__abort.is_set():
                return abortres
            res = checker.check_exprs(exprs)
            runtime = time.time() - start
            if self.__abort.is_set():
                return abortres
            if res:
                return (True,
                        Task(task.nodeid, task.name, exprs, None, runtime))
            return (False,
                    Task(task.nodeid, task.name, None, None, runtime))
        except Exception as e:
            logging.info(f'{type(e)} in check of {task.name}: {e}')
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_tb(exc_traceback, limit=10, file=sys.stderr)
        return abortres


class MutatorStats:
    """Gather information about the performance of the individual mutators."""
    def __init__(self):
        self.data = {}
        self.__enabled = logging.getLogger().isEnabledFor(logging.INFO)

    def add(self, success, task, original):
        """Add results from one check."""
        if task.runtime is None or not self.__enabled:
            return
        d = self.data.setdefault(task.name, {
            'tests': 0,
            'success': 0,
            'diff': 0,
            'runtime': 0
        })
        d['tests'] += 1
        if success:
            d['success'] += 1
            d['diff'] += nodes.count_exprs(
                task.exprs) - nodes.count_exprs(original)
        d['runtime'] += task.runtime

    def print(self):
        """Print cumulative results."""
        if self.__enabled:
            for name, data in sorted(self.data.items()):
                logging.info(
                    f'{name}: diff {data["diff"]:+} expressions, {data["tests"]} tests ({data["success"]}), {data["runtime"]:.1f}s'
                )


def reduce(exprs):
    """Reduces the input given in ``exprs`` as good as possible in a fixed-
    point loop."""
    passes = get_passes()
    passid = 0

    nchecks = 0
    nreduce = 0
    stats = MutatorStats()

    # use one pool for the whole reduction
    #with multiprocessing.Pool(options.args().jobs) as pool:
    with multiprocessing.pool.ThreadPool(options.args().jobs) as pool:
        # abort flag is passed to both producer and consumer
        # important: the pool will break if the abort_flag is destroyed early
        abort_flag = multiprocessing.Manager().Event()
        # iterate over all passes
        for passid in range(len(passes)):
            cur_passes, params = get_pass(passes, passid)
            logging.chat(f'pass {passid + 1} / {len(passes)}')
            skip = 0
            fresh_run = True
            # repeat until no further reduction is found
            while True:
                reduction = False
                start = time.time()
                smtlib.collect_information(exprs)
                cnt = nodes.count_nodes(exprs)
                progress.start(cnt)
                progress.update(min(cnt, skip))
                abort_flag.clear()
                prod = Producer(cur_passes, abort_flag, exprs)
                cons = Consumer(abort_flag)
                for result in pool.imap_unordered(cons.check,
                                                  prod.generate(skip, params)):
                    nchecks += 1
                    success, task = result
                    if abort_flag.is_set():
                        # skip remaining results if we had a success
                        skip = min(skip, task.nodeid - 1)
                        continue
                    progress.update(task.nodeid)
                    stats.add(success, task, exprs)
                    if success:
                        # trigger abort, then process the result
                        abort_flag.set()
                        progress.finish()
                        nreduce += 1
                        runtime = time.time() - start
                        logging.chat(
                            f'#{nreduce}: {task.name} ({runtime:.2f}s, {nodes.count_nodes(task.exprs)} expressions)'
                        )
                        reduction = True
                        exprs = nodes.reduplicate(task.exprs)
                        skip = task.nodeid - 1
                        fresh_run = False
                        nodeio.write_smtlib_to_file(options.args().outfile,
                                                    exprs)
                if not reduction:
                    progress.finish()
                    logging.info('No further simplification found')
                    if fresh_run:
                        # this was a fresh run, continue with next pass
                        break
                    logging.info('Starting over')
                    skip = 0
                    fresh_run = True

    stats.print()

    return exprs, nchecks
