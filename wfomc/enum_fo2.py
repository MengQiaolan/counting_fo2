from __future__ import annotations

import os
import argparse
import logging
import logzero
import functools
from logzero import logger
from contexttimer import Timer
from itertools import combinations

from wfomc.parser import parse_input
from wfomc.fol.syntax import AtomicFormula, Const, Pred, X, Y
from wfomc.utils import MultinomialCoefficients, multinomial
from wfomc.cell_graph.components import Cell
from wfomc.cell_graph.cell_graph import CellGraph, OptimizedCellGraphWithPC
from wfomc.network.constraint import PartitionConstraint

from wfomc.enum_context import EnumContext
from wfomc.enum_utils import remove_aux_atoms, Pred_A, fast_wfomc_with_pc, get_init_configs


META_CCS: set[tuple[int]] = set()
STATIS_META_CCS: dict[tuple[int], int] = dict()
DOMAIN_SIZE: int = 0
MC: int = 0
SAT_COUNT: int = 0
DELTA: int = 0

Domain_to_Cell: dict[Const, Cell] = {}
Rel_Dict: dict[tuple[Cell, Cell], set[frozenset[AtomicFormula]]] = {}

Rpred_to_Zpred: dict[Pred, Pred] = {}
Zpred_to_Rpred: dict[Pred, Pred] = {}

Pi_to_Rel: dict[Pred, frozenset[AtomicFormula]] = {}
Rel_to_Pi: dict[frozenset[AtomicFormula], Pred] = {}
Pi_to_elimZ: dict[Pred, set[Pred]] = {}
Pi_to_elimZ_in_A: dict[Pred, set[Pred]] = {}

Evi_to_Xpred: dict[frozenset[AtomicFormula], Pred] = {}
Xpred_to_Evi: dict[Pred, frozenset[AtomicFormula]] = {}

Xpred_to_NewCellIndices: dict[Pred, set[int]] = {}
Xpred_to_NewCellIndices_List: list[set[int]] = {}
            
@functools.lru_cache(maxsize=None)
def calculate_meta_cc(config: tuple, hold: tuple):
    global META_CCS, STATIS_META_CCS
    if sum(config) > DOMAIN_SIZE:
        return
    cc = tuple([sum([config[idx] for idx in Xpred_to_NewCellIndices_List[i]]) 
                        for i in range(len(Xpred_to_NewCellIndices_List))])
    if any(all((a >= b and b > 0) or (a == b and b == 0) for a, b in zip(cc, meta_cc)) for meta_cc in META_CCS):
        return
    if sat_config(config):
        STATIS_META_CCS[cc] = 0
        META_CCS.add(cc)
        return
    if sum(config) == DOMAIN_SIZE:
        return
    
    hold = list(hold)
    # hold[i]==True means that we don't need to increase the i-th element
    # to get a new increamented config,
    # since it has been increased to the upper bound 
    # or the new incremented config can be derived from a meta_cfg.
    # each hold[i] can only change from False to True.
    for i in range(len(config)):
        if hold[i]:
            # we don't consider to increase the i-th element if hold[i] is True
            continue
        # the four cases that hold[i] should be True
        if config[i] == DELTA:
            # 1) the i-th element has been increased to the upper bound
            hold[i] = True
            continue
        l = list(config)
        l[i] += 1
        inc_config = tuple(l)
        inc_cc = tuple([sum([inc_config[idx] for idx in Xpred_to_NewCellIndices_List[i]]) 
                        for i in range(len(Xpred_to_NewCellIndices_List))])
        if inc_cc in META_CCS:
            # 2) the new incremented config has been a meta_cfg
            hold[i] = True
            continue
        if any(all((a >= b and b > 0) or (a == b and b == 0) for a, b in zip(inc_cc, meta_cc))
                    for meta_cc in META_CCS):
            # 3) the new incremented config can be derived from a meta_cfg
            hold[i] = True
            continue
        if (not hold[i]) and sat_config(inc_config): # call sat function as little as possible
            STATIS_META_CCS[inc_cc] = 0
            META_CCS.add(inc_cc)
            # 4) the new incremented config is a meta_cfg
            hold[i] = True

    for i in range(len(config)):
        # TODO: only consider the 1-types that can provade possible 2-tables
        if not hold[i]:
            l = list(config)
            l[i] += 1
            calculate_meta_cc(tuple(l), tuple(hold))

CurCells: list[Cell] = []
Cur_Cell_to_Tpred: dict[Cell, Pred] = {}
Cur_Cell_Graph: CellGraph = None

@functools.lru_cache(maxsize=None)
def sat_config(config: tuple) -> bool:
    global SAT_COUNT
    SAT_COUNT += 1
    '''
    Check if the config is satisfiable
    Note: the config is based on the sentence without
    1-type predicates (@T) and skolem predicates (@skolem)
    '''
    # the config is the cardinality constraint of 1-type predicates.
    # we can see it as partition constraint in the correspoding skolemized sentence.
    pcl: list = []
    for cell, num in zip(CurCells, config):
        # a partition constraint restricts the number of Tpred (@T)
        # @T is equivalent to the 1-types in the original sentence (without skolemization)
        pcl.append((Cur_Cell_to_Tpred[cell], num))
    pc: PartitionConstraint = PartitionConstraint(pcl)
    return fast_wfomc_with_pc(Cur_Cell_Graph, pc) > 0

@functools.lru_cache(maxsize=None)
def sat_cc(cc: tuple[int]) -> bool:
    global STATIS_META_CCS, META_CCS
    for meta_cc in META_CCS:
        if all((a >= b and b > 0) or (a == b and b == 0) for a, b in zip(cc, meta_cc)):
            STATIS_META_CCS[meta_cc] += 1
            return True
    return False

def update_PZ_evidence(evidence_dict: dict[Const, Pred], 
                      cc_xpred: dict[Pred, int],
                      relation: frozenset[AtomicFormula],
                      first_e: Const, second_e: Const):
    '''
    update P and Z evidence when considering a new relation (2-table)
    '''
    cc_xpred[evidence_dict[first_e]] -= 1
    cc_xpred[evidence_dict[second_e]] -= 1
    for atom in relation:
        # consider the influence of positive R literals to correspoding block (Z)
        if atom.pred.name.startswith('@R') and atom.positive:
            z_pred = Rpred_to_Zpred[atom.pred]
            firt_arg = first_e if atom.args[0] == X else second_e
            if z_pred(X) in Xpred_to_Evi[evidence_dict[firt_arg]]:
                # update x pred when deleting block (z)
                evidence_dict[firt_arg] = Evi_to_Xpred[frozenset(
                    {atom for atom in Xpred_to_Evi[evidence_dict[firt_arg]] if atom != z_pred(X)})]
    # update x pred when the second element add P unary evidence according to the relation
    evidence_dict[second_e] = Evi_to_Xpred[frozenset(Xpred_to_Evi[evidence_dict[second_e]]|{Rel_to_Pi[relation](X)})]
    # update cc after updating evidence
    cc_xpred[evidence_dict[first_e]] += 1
    cc_xpred[evidence_dict[second_e]] += 1

def clean_P_evidence(evidence_dict: dict[Const, Pred], cc_xpred: dict[Pred, int]):
    '''
    clean the P evidence of all elements before sampling a ego sturcture
    '''
    for e, xpred in evidence_dict.items():
        cc_xpred[xpred] -= 1
        evidence_dict[e] = Evi_to_Xpred[frozenset(
            {atom for atom in Xpred_to_Evi[xpred] 
                    if not atom.pred.name.startswith('@P')})]
        cc_xpred[evidence_dict[e]] += 1

def update_A_evidence(ego_element: Const, evidence_dict: dict[Const, Pred], cc_xpred: dict[Pred, int]):
    '''
    update A evidence after determining the ego element
    '''
    cc_xpred[evidence_dict[ego_element]] -= 1
    evidence_dict[ego_element] = Evi_to_Xpred[frozenset({Pred_A(X)}|
                {atom for atom in Xpred_to_Evi[evidence_dict[ego_element]] if atom != ~Pred_A(X)})]
    cc_xpred[evidence_dict[ego_element]] += 1

def domain_recursion(domain: set[Const], 
                     evidence_dict: dict[Const, Pred], 
                     cc_xpred: dict[Pred, int],
                     cur_model: set[AtomicFormula] = set(),):
    if domain.__len__() == 0:
        global MC
        MC += 1
        # print(MC, remove_aux_atoms(cur_model))
        return

    ego_element = domain.pop()
    # print('    '*(3-len(domain))+f'current element: {ego_element}')
    clean_P_evidence(evidence_dict, cc_xpred)
    update_A_evidence(ego_element, evidence_dict, cc_xpred)
    ego_structure_sampling(ego_element=ego_element, domain_todo=domain, domain_done=set(), 
                           evidence_dict=evidence_dict, cc_xpred=cc_xpred,
                           cur_model=cur_model)

def ego_structure_sampling(ego_element: Const, 
                           domain_todo: set[Const], domain_done: set[Const],
                           evidence_dict: dict[Const, Pred], cc_xpred: dict[Pred, int],
                           cur_model: set[AtomicFormula]):
    if domain_todo.__len__() == 0:
        cc_xpred[evidence_dict[ego_element]] -= 1
        del evidence_dict[ego_element]
        domain_recursion(domain_done, evidence_dict, cc_xpred, cur_model)
        return
    cur_element = domain_todo.pop()
    domain_done.add(cur_element)
    for rel in Rel_Dict[(Domain_to_Cell[ego_element], Domain_to_Cell[cur_element])]:
        # we need to keep the original evidence set for the next iteration (next relation)
        # as same as the cc_xpred, domain_todo, domain_done
        new_evidence_dict = evidence_dict.copy()
        new_cc_xpred = cc_xpred.copy()
        # update evidence about P and Z according to the selected relation
        update_PZ_evidence(new_evidence_dict, new_cc_xpred, rel, ego_element, cur_element)
        # print('  '+'    '*(3-len(domain_done)-len(domain_todo))+f'###    ({ego_element}, {cur_element}) => {set(rel)} => {new_evidence_dict}, {new_cc_xpred}')
        # use tuple instead of dict (non-hashable) to use lru_cache
        if sat_cc(tuple([new_cc_xpred[key] for key in sorted(new_cc_xpred.keys(), key=lambda x: x.name)])):
            # print('  '+'    '*(3-len(domain_done)-len(domain_todo))+f'    ({ego_element}, {cur_element}) => {set(rel)} => {new_evidence_dict}, {new_cc_xpred}')
            ego_structure_sampling(
                ego_element, domain_todo.copy(), domain_done.copy(),
                new_evidence_dict, new_cc_xpred,
                cur_model|{atom.substitute({X: ego_element, Y: cur_element}) for atom in rel})

def generate_config_class(domain_size:int, len_config:int, delta:int):
    '''
    Generate all possible tuple (k_0, k_sup) where
    k_0 is the number of cells that have no element and 
    k_sup is the number of cells that have delta elements.
    '''
    for k_zero in range(len_config):
        if (len_config - k_zero) * (delta - 1) >= domain_size:
            yield (k_zero, 0)
        for k_sup in range(1, len_config + 1 - k_zero):
            if k_sup * delta + (len_config - k_zero - k_sup) > domain_size:
                break
            yield (k_zero, k_sup)

def generate_base_configs(len_config, k_zero, k_sup, delta):
    '''
    A base config is a config consisting of (0, 1, delta)
    Generate all possible base configurations based on the tuple (k_0, k_sup)
    '''
    def backtrack(cur_tuple: tuple, zero_left, sup_left):
        if cur_tuple.__len__() == len_config:
            yield cur_tuple
            return
        if len_config - cur_tuple.__len__() >= zero_left + sup_left:
            if zero_left > 0:
                yield from backtrack(cur_tuple + (0,), zero_left-1, sup_left)
            if sup_left > 0:
                yield from backtrack(cur_tuple + (delta,), zero_left, sup_left-1)
            if len_config - cur_tuple.__len__() -1 >= zero_left + sup_left:
                yield from backtrack(cur_tuple + (1,), zero_left, sup_left)
    yield from backtrack(tuple(), k_zero, k_sup)

def calculate_template_config(base_config: tuple, flag: bool, tpl_configs: set[tuple], delta):
    '''
    A template config is a sat config consisting of (0, x, delta) where 1<=x<delta
    Calculate the template configurations based on the current config.
    '''
    if sum(base_config) > DOMAIN_SIZE:
        return
    # flag==True means that the current config is based on a template config
    if not flag:
        flag = sat_config(base_config)
    if flag:
        if sum(base_config) == DOMAIN_SIZE:
            tpl_configs.add(base_config)
            return
        if sum(base_config) < DOMAIN_SIZE and any(ni == delta for ni in base_config):
            tpl_configs.add(base_config)
        else:
            pass
    
    for i in range(len(base_config)):
        if base_config[i] == 0 or base_config[i] == delta :
            continue
        if base_config[i]+1 == delta:
            continue
        inc = list(base_config)
        inc[i] += 1
        calculate_template_config(tuple(inc), flag, tpl_configs, delta)
        
def distribute_cells(domain, config):
    """
    Distribute cells to domain elements according to the configuration.
    """
    def backtrack(remaining_elements, current_distribution):
        if len(current_distribution) == len(config):
            if all(len(d) == cnt for d, cnt in zip(current_distribution, config)):
                yield current_distribution
            return

        count = config[len(current_distribution)]
        for comb in combinations(remaining_elements, count):
            next_remaining = remaining_elements - set(comb)
            yield from backtrack(next_remaining, current_distribution + [list(comb)])

    yield from backtrack(domain, [])

def generate_sat_configs(domain_size, template_configs, delta):
    """
    Generate all satisfiable configurations based on the template configurations.
    """
    for template_config in template_configs:
        indeces = list()
        remaining = domain_size - sum(template_config)
        if remaining > 0:
            for idx, ni in enumerate(template_config):
                if ni == delta: # TODO
                    indeces.append(idx)
            if len(indeces) == 0:
                raise RuntimeError('The template config is not valid')
            for extra_partition in multinomial(len(indeces), remaining):
                config = list(template_config)
                for idx, extra in zip(indeces, extra_partition):
                    config[idx] += extra
                yield tuple(config)
        else:
            yield template_config

def parse_args():
    parser = argparse.ArgumentParser(
        description='Enumerate models of a given sentence',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', type=str, required=True, help='wfomcs file')
    parser.add_argument('--output_dir', '-o', type=str, default='./check-points')
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # import sys
    # sys.setrecursionlimit(int(1e6))
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.debug:
        logzero.loglevel(logging.DEBUG)
    else:
        logzero.loglevel(logging.INFO)
    logzero.logfile('{}/log.txt'.format(args.output_dir), mode='w')

    problem = parse_input(args.input)
    context: EnumContext = EnumContext(problem)
    DELTA = context.delta
    DOMAIN_SIZE = len(context.domain)
    MultinomialCoefficients.setup(DOMAIN_SIZE)
    
    original_cells: list[Cell] = context.original_cells
    
    Zpred_to_Rpred = context.zpred_to_rpred
    Rpred_to_Zpred = context.rpred_to_zpred

    Pi_to_Rel = context.Pi_to_Rel
    Rel_to_Pi = context.Rel_to_Pi
    Rel_Dict = context.rel_dict
    Pi_to_elimZ = context.Pi_to_elimZ
    Pi_to_elimZ_in_A = context.Pi_to_elimZ_in_A
    uni_formula_with_AP = context.uni_formula_AP
    
    x_preds = context.x_preds
    evi_formulas = context.evi_formulas
    xpreds_with_P = context.xpreds_with_P
    Evi_to_Xpred = context.Evi_to_Xpred
    Xpred_to_Evi = context.Xpred_to_Evi
    uni_formula_with_APZX = context.uni_formula_with_APZX
    
    # logger.setLevel(logging.CRITICAL)
    # logger.setLevel(logging.INFO)
    
    uni_APZX_cell_graph = context.uni_APZX_cell_graph
    uni_APZX_cells: list[Cell] = uni_APZX_cell_graph.cells 
    
    for cell in uni_APZX_cells:
        print(cell)
                        
    Xpred_to_NewCellIndices, Xpred_to_NewCellIndices_List = \
        context.map_Xpred_to_Uni_PZX_Cell()
    
    uni_APZX_cell_to_Tpred = context.uni_APZX_cell_to_Tpred
    skolem_formula_with_APZXT = context.skolem_formula_with_APZXT
    
    with Timer() as t:
        # global vars for the func 'sat_config'
        sat_config.cache_clear()
        CurCells = uni_APZX_cells
        Cur_Cell_to_Tpred = uni_APZX_cell_to_Tpred
        Cur_Cell_Graph = OptimizedCellGraphWithPC(skolem_formula_with_APZXT, context.get_weight, DOMAIN_SIZE, 
                                                PartitionConstraint([(tau, 0) for tau in uni_APZX_cell_to_Tpred.values()]))
        # we only consider the case that:
        # 1) there is only one element whose evidence contain '@A(X)'
        A_idx = [idx for idx, cell in enumerate(uni_APZX_cells) if cell.is_positive(Pred_A)]
        # 2) there is at least one element whose evidence contain '@P(X)'
        P_idx = [i for x in xpreds_with_P for i in Xpred_to_NewCellIndices[x] ]
        # 3) we can assert that some evi types containing '@A(X)' is impossible
        # when there are some P evidence in other elements
        A_idx_to_Zpred: list[set[Pred]] = []
        negA_idx_to_elimZ_in_A: list[set[Pred]] = []
        for x in x_preds:
            if Pred_A(X) not in Xpred_to_Evi[x]:
                break
            zs = set(z_atom.pred for z_atom in Xpred_to_Evi[x] if z_atom.pred.name.startswith('@Z'))
            for _ in Xpred_to_NewCellIndices[x]:
                A_idx_to_Zpred.append(zs)
        for x in x_preds:
            if Pred_A(X) in Xpred_to_Evi[x]:
                continue
            zs = set()
            for p_atom in Xpred_to_Evi[x]:
                if p_atom.pred.name.startswith('@P') and p_atom.positive:
                    zs.update(Pi_to_elimZ_in_A[p_atom.pred])
            for _ in Xpred_to_NewCellIndices[x]:
                negA_idx_to_elimZ_in_A.append(zs)
        
        print(A_idx_to_Zpred)
        print(negA_idx_to_elimZ_in_A)
        
        # find all possible initial configurations that satisfy above constraints
        init_configs = get_init_configs(uni_APZX_cell_graph, 
                                        len(context.original_ext_formulas), 
                                        len(A_idx), P_idx, 
                                        A_idx_to_Zpred, negA_idx_to_elimZ_in_A,
                                        DOMAIN_SIZE)
        for init_config in init_configs:
            init_holds = [True if (j in A_idx or init_config[j] == 0) else False 
                            for j in range(len(uni_APZX_cells))]
            # a meta config is based on the sentence uni_formula_with_PZX (only @R, @A, @P, and @X)
            # not the sentence with 1-type predicates (@T) and skolem predicates (@skolem)
            calculate_meta_cc(tuple(init_config), tuple(init_holds))
        
        print(SAT_COUNT)
    print('time: %s', t.elapsed)
    # print(META_CCS)
    for cc in META_CCS:
        print(cc)
    exit()
    
    
    with Timer() as t:
    # global variables for 'sat_config' function
        sat_config.cache_clear()
        CurCells = original_cells
        Cur_Cell_to_Tpred = context.oricell_to_tau
        Cur_Cell_Graph = OptimizedCellGraphWithPC(context.skolem_tau_formula, context.get_weight, DOMAIN_SIZE, 
                                                    PartitionConstraint([(context.oricell_to_tau[cell], 0) for cell in original_cells]))
        # enumerate satisfiable configs
        ori_len_config = len(context.original_cells)
        ori_delta = context.delta
        for cfg_class in generate_config_class(DOMAIN_SIZE, ori_len_config, ori_delta):
            for base_config in generate_base_configs(ori_len_config, cfg_class[0], cfg_class[1], ori_delta):
                tpl_cfgs = set()
                calculate_template_config(base_config, False, tpl_cfgs, ori_delta)
                for config in generate_sat_configs(DOMAIN_SIZE, tpl_cfgs, ori_delta):
                    logger.info('The configuration: %s', config)
                    # init cardinality constraint for each X pred
                    cc_xpred: dict[Pred, int] = {x: 0 for x in x_preds}
                    for cell, num in zip(original_cells, config):
                        cc_xpred[Evi_to_Xpred[context.init_evi_dict[cell]]] += num
                    # assign 1-types for all elements
                    for idx, domain_partitions in enumerate(distribute_cells(context.domain, config)):
                        logger.info('The distribution: %s', domain_partitions)
                        # init evidence set (1-type, block type and negative A)
                        evidence_dict: dict[Const, Pred] = {}
                        for cell, elements in zip(original_cells, domain_partitions):
                            print(cell, " = ",elements)
                            for element in elements:
                                Domain_to_Cell[element] = cell
                                evidence_dict[element] = Evi_to_Xpred[context.init_evi_dict[cell]]
                        logger.info('The init evidence: \n%s', evidence_dict)
                        domain_recursion(context.domain.copy(), evidence_dict, cc_xpred.copy())
    logger.info('The number of models: %s', MC)
    print('time: %s', t.elapsed)
    # sorted_keys = sorted(STATIS_META_CCS, key=lambda k: (STATIS_META_CCS[k], sum(k)), reverse=True)
    sorted_keys = sorted(STATIS_META_CCS, key=lambda k: (sum(k)), reverse=False)
    for k in sorted_keys:
        print(k, STATIS_META_CCS[k], "        (", sum(k),  sum([k[int(x.name[2:])] for x in xpreds_with_P]), ")")
    print()
    for cell in uni_APZX_cells:
        print(cell)
    print()
    for evi_formula in evi_formulas:
        print(evi_formula)
    print()
    for rel_formula in context.rel_formulas:
        print(rel_formula)
    print()