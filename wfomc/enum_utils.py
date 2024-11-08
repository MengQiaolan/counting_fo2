from __future__ import annotations
from collections import defaultdict

from logzero import logger
from wfomc.fol.sc2 import SC2, to_sc2
from wfomc.fol.utils import new_predicate, convert_counting_formula

from wfomc.network.constraint import CardinalityConstraint
from wfomc.fol.syntax import *
from wfomc.problems import WFOMCProblem
from wfomc.fol.syntax import SKOLEM_PRED_NAME
from wfomc.utils.third_typing import RingElement, Rational

from wfomc.context import WFOMCContext
from wfomc.cell_graph.cell_graph import CellGraph, Cell, OptimizedCellGraphWithPC
from wfomc.network.constraint import PartitionConstraint

from functools import reduce
from wfomc.fol.syntax import AtomicFormula

from collections import defaultdict
from itertools import product

from contexttimer import Timer

from itertools import combinations

from wfomc.utils import multinomial, Rational

from wfomc.cell_graph.utils import conditional_on

from wfomc.cell_graph.cell_graph import build_cell_graphs
from wfomc.network.constraint import PartitionConstraint
from wfomc.utils import MultinomialCoefficients, multinomial_less_than, RingElement, Rational
from wfomc.fol.syntax import Const, Pred, QFFormula

from wfomc.fol.syntax import AtomicFormula, Const, Pred, QFFormula, a, b, c


# def get_cells(preds: tuple[Pred], formula: QFFormula):
#     gnd_formula_cc: QFFormula = ground_on_tuple(formula, c)
#     cells = []
#     code = {}
#     for model in gnd_formula_cc.models():
#         for lit in model:
#             code[lit.pred] = lit.positive
#         cells.append(Cell(tuple(code[p] for p in preds), preds))
#     return cells

Pred_A = Pred("@A", 1)
Pred_P = Pred("@P", 1)
Pred_D = Pred("@D", 2)

ENUM_R_PRED_NAME = '@R'
ENUM_Z_PRED_NAME = '@Z'
ENUM_P_PRED_NAME = '@P'
ENUM_X_PRED_NAME = '@X'
ENUM_T_PRED_NAME = '@T'

def build_two_tables(formula: QFFormula, cells: list[Cell]) -> tuple[set[frozenset[AtomicFormula]], dict]:
    models = dict()
    gnd_formula: QFFormula = ground_on_tuple(formula, a, b) & ground_on_tuple(formula, b, a)
    gnd_formula = gnd_formula.substitute({a: X, b: Y})
    gnd_lits = gnd_formula.atoms()
    gnd_lits = gnd_lits.union(frozenset(map(lambda x: ~x, gnd_lits)))
    for model in gnd_formula.models():
        model = frozenset({atom for atom in model if len(atom.args) ==2 and atom.args[0] != atom.args[1]})
        models[model] = 1

    tables = dict()
    for i, cell in enumerate(cells):
        models_1 = conditional_on(models, gnd_lits, cell.get_evidences(a))
        for j, other_cell in enumerate(cells):
            if i > j:
                tables[(cell, other_cell)] = tables[(other_cell, cell)]
                continue
            models_2 = conditional_on(models_1, gnd_lits, other_cell.get_evidences(b))
            tables[(cell, other_cell)] = set(models_2.keys())
    return set(models.keys()), tables

def ground_on_tuple(formula: QFFormula, c1: Const, c2: Const = None) -> QFFormula:
        variables = formula.vars()
        if len(variables) > 2:
            raise RuntimeError(
                "Can only ground out FO2"
            )
        if len(variables) == 1:
            constants = [c1]
        else:
            if c2 is not None:
                constants = [c1, c2]
            else:
                constants = [c1, c1]
        substitution = dict(zip(variables, constants))
        gnd_formula = formula.substitute(substitution)
        return gnd_formula

def remove_aux_atoms(atoms: set[AtomicFormula]) -> set[AtomicFormula]:
    return set(filter(lambda atom: not atom.pred.name.startswith('@'), atoms))

def fast_wfomc_with_pc(opt_cell_graph_pc: OptimizedCellGraphWithPC, 
                       partition_constraint: PartitionConstraint) -> RingElement:
    # formula: QFFormula, domain_size: int,
    #                      get_weight: Callable[[Pred], tuple[RingElement, RingElement]],
    #                      partition_constraint: PartitionConstraint) -> RingElement:
   
    cliques = opt_cell_graph_pc.cliques
    nonind = opt_cell_graph_pc.nonind
    nonind_map = opt_cell_graph_pc.nonind_map

    pred_partitions: list[list[int]] = list(num for _, num in partition_constraint.partition)
    # partition to cliques
    partition_cliques: dict[int, list[int]] = opt_cell_graph_pc.partition_cliques

    res = Rational(0, 1)
    with Timer() as t:
        for configs in product(
            *(list(multinomial_less_than(len(partition_cliques[idx]), constrained_num)) for
                idx, constrained_num in enumerate(pred_partitions))
        ):
            coef = Rational(1, 1)
            remainings = list()
            # config for the cliques
            overall_config = list(0 for _ in range(len(cliques)))
            # {clique_idx: [number of elements of pred1, pred2, ..., predk]}
            clique_configs = defaultdict(list)
            for idx, (constrained_num, config) in enumerate(zip(pred_partitions, configs)):
                remainings.append(constrained_num - sum(config))
                mu = tuple(config) + (constrained_num - sum(config), )
                coef = coef * MultinomialCoefficients.coef(mu)
                for num, clique_idx in zip(config, partition_cliques[idx]):
                    overall_config[clique_idx] = overall_config[clique_idx] + num
                    clique_configs[clique_idx].append(num)

            body = opt_cell_graph_pc.get_i1_weight(
                remainings, overall_config
            )

            for i, clique1 in enumerate(cliques):
                for j, clique2 in enumerate(cliques):
                    if i in nonind and j in nonind:
                        if i < j:
                            body = body * opt_cell_graph_pc.get_two_table_weight(
                                (clique1[0], clique2[0])
                            ) ** (overall_config[nonind_map[i]] *
                                    overall_config[nonind_map[j]])

            for l in nonind:
                body = body * opt_cell_graph_pc.get_J_term(
                    l, tuple(clique_configs[nonind_map[l]])
                )
            res = res + coef * body
    return res

def get_init_configs_old(cell_graph: CellGraph, m: int, 
                     A_prefix_len: int, contain_P_idx: list, 
                     A_idx_to_Zpred: list[set[Pred]], negA_idx_to_elimZ_in_A: list[set[Pred]],
                     domain_size: int) -> dict[int, list[tuple[int, ...]]]:
        # the first A_prefix_len cells have A(x) and the rest have ~A(x) (we hack this in CellGraph)
        cells = cell_graph.get_cells()
        u = len(cells)
        
        # sat_dict[(cell_idx, R_idx)] is a set of cell indices 
        # that can satisfy the R_idx-th predicate of the cell_idx-th cell
        sat_dict: dict[tuple[int, int], set[int]] = dict()
        for i in range(u):
            for j in range(m):
                sat_dict[(i, j)] = set()
        
        # fill sat_dict
        for i, cell_i in enumerate(cells):
            for pred in cell_i.preds:
                if pred.name.startswith('@R') and cell_i.is_positive(pred):
                    sat_dict[(i, int(pred.name[2:]))].add(i)
            for j, cell_j in enumerate(cells):
                if j > i:
                    continue
                for rel, weight in cell_graph.get_two_tables((cell_i, cell_j)):
                    if weight > 0:
                        for atom in rel:
                            pred_name = atom.pred.name
                            if pred_name.startswith('@R') and atom.positive:
                                if atom.args[0] == a:
                                    sat_dict[(i, int(pred_name[2:]))].add(j)
                                else:
                                    sat_dict[(j, int(pred_name[2:]))].add(i)
        
        remaining = u - A_prefix_len - len(contain_P_idx)
        suffixes: list[tuple[int, ...]] = []
        for k in range(2, domain_size):
            if k > remaining:
                break
            for indices in combinations(range(remaining), k):
                l = [0] * remaining
                for index in indices:
                    l[index] = 1
                # there is at least one element whose evidence contain '@P(X)'
                if k != 0 and sum([l[i-A_prefix_len] for i in contain_P_idx]) == 0:
                    continue
                suffixes.append(tuple(l))
        
        res = []
        for true_loc in range(A_prefix_len):
            zpred_in_A = A_idx_to_Zpred[true_loc]
            prefix = (0,) * true_loc + (1,) + (0,) * (A_prefix_len - true_loc - 1)
            for true_loc in range(len(contain_P_idx)):
                elimZ_by_suffix = set(z for i in range(len(suffix)) if suffix[i] != 0 for z in negA_idx_to_elimZ_in_A[i])
            
            for suffix in suffixes:
                elimZ_by_suffix = set(z for i in range(len(suffix)) if suffix[i] != 0 for z in negA_idx_to_elimZ_in_A[i])
                if len(zpred_in_A & elimZ_by_suffix) != 0:
                    continue
                init_config = prefix + suffix
                cell_set = set([index for index, value in enumerate(init_config) if value != 0])
                flag = True # record if the init_config is possible sat
                for i in range(len(init_config)):
                    if init_config[i] == 0:
                        continue
                    for r in range(m):
                        need_check = False
                        for pred in cells[i].preds:
                            if pred.name.startswith('@Z') and int(pred.name[2:]) == r:
                                if cells[i].is_positive(pred):
                                    need_check = True
                        if need_check:
                            # if there is no cell in cell_set that can satisfy predicate r of i-th cell
                            if len(sat_dict[(i, r)] & cell_set) == 0:
                                flag = False
                                break
                    if not flag:
                        break
                if flag:
                    res.append(init_config)
        return res
    
    
def get_init_configs(cell_graph: CellGraph, m: int, 
                     A_idx: list[int], P_idx: list[int], domain_size: int) -> dict[int, list[tuple[int, ...]]]:
    
    cells = cell_graph.get_cells()
    u = len(cells)
    
    # sat_dict[(cell_idx, R_idx)] is a set of cell indices 
    # that can satisfy the R_idx-th predicate of the cell_idx-th cell
    sat_dict: dict[tuple[int, int], set[int]] = dict()
    for i in range(u):
        for j in range(m):
            sat_dict[(i, j)] = set()
    
    # fill sat_dict
    for i, cell_i in enumerate(cells):
        for pred in cell_i.preds:
            if pred.name.startswith('@R') and cell_i.is_positive(pred):
                sat_dict[(i, int(pred.name[2:]))].add(i)
        for j, cell_j in enumerate(cells):
            if j > i:
                continue
            for rel, weight in cell_graph.get_two_tables((cell_i, cell_j)):
                if weight > 0:
                    for atom in rel:
                        pred_name = atom.pred.name
                        if pred_name.startswith('@R') and atom.positive:
                            if atom.args[0] == a:
                                sat_dict[(i, int(pred_name[2:]))].add(j)
                            else:
                                sat_dict[(j, int(pred_name[2:]))].add(i)
    
    A_prefix_len = len(A_idx)
    remaining = u - A_prefix_len
    suffixes: list[tuple[int, ...]] = []
    for k in range(1, domain_size):
        if k > remaining:
            break
        for indices in combinations(range(remaining), k):
            l = [0] * remaining
            for index in indices:
                l[index] = 1
            # there is at least one element whose evidence contain '@P(X)'
            if sum([l[i-A_prefix_len] for i in P_idx]) == 0:
                continue
            suffixes.append(tuple(l))
    
    res = []
    for true_loc in range(A_prefix_len):
        prefix = (0,) * true_loc + (1,) + (0,) * (A_prefix_len - true_loc - 1)
        
        for suffix in suffixes:
            init_config = prefix + suffix
            cell_set = set([index for index, value in enumerate(init_config) if value != 0])
            flag = True # record if the init_config is possible sat
            for i in range(len(init_config)):
                if init_config[i] == 0:
                    continue
                for r in range(m):
                    need_check = False
                    for pred in cells[i].preds:
                        if pred.name.startswith('@Z') and int(pred.name[2:]) == r:
                            if cells[i].is_positive(pred):
                                need_check = True
                    if need_check:
                        # if there is no cell in cell_set that can satisfy predicate r of i-th cell
                        if len(sat_dict[(i, r)] & cell_set) == 0:
                            flag = False
                            break
                if not flag:
                    break
            if flag:
                res.append(init_config)
    return res