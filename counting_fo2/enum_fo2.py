from __future__ import annotations
from collections import defaultdict

from functools import reduce
import math
import os
import argparse
import logging
import logzero
from itertools import combinations, product

from logzero import logger
from typing import Callable
from contexttimer import Timer
from counting_fo2.parser import parse_input
from counting_fo2.wfomc import Algo
from counting_fo2.context import EnumContext, EnumContext2, FinalEnumContext
from counting_fo2.cell_graph.utils import conditional_on

from counting_fo2.utils import multinomial, Rational
from counting_fo2.fol.sc2 import to_sc2, SC2
from counting_fo2.fol.utils import new_predicate, exactly_one_qf, remove_aux_atoms
from counting_fo2.cell_graph.components import Cell
from counting_fo2.fol.syntax import AtomicFormula, Const, Pred, QFFormula, a, b, c, \
    ENUM_P_PRED_NAME, ENUM_X_PRED_NAME, X, Y, top, bot, Implication, QuantifiedFormula
    
from counting_fo2.meta_config import get_template_config
from counting_fo2.problems import WFOMCSProblem

from counting_fo2.context.wfoms_context import WFOMSContext
from counting_fo2.cell_graph.cell_graph import CellGraph

ORIGINAL_DELTA: int = 0
NEW_UNI_FORMULA: QFFormula
NEW_META_CONFIGS: set[tuple[int]] = set()

MC: int = 0

Domain_to_Cell: dict[Const, Cell] = {}
Rel_Dict: dict[tuple[Cell, Cell], set[frozenset[AtomicFormula]]] = {}
Pred_A = Pred("@A", 1)
Pred_Pi_Set: set[Pred] = set()

Rpred_to_Zpred: dict[Pred, Pred] = {}
Pi_to_Rel: dict[Pred, frozenset[AtomicFormula]] = {}
Rel_to_Pi: dict[frozenset[AtomicFormula], Pred] = {}
Evi_to_Xpred: dict[frozenset[AtomicFormula], Pred] = {}
Xpred_to_Evi: dict[Pred, frozenset[AtomicFormula]] = {}
Xpred_to_NewCellIndices: dict[Pred, set[int]] = {}

def sat_cc(cc: dict[Pred, int]) -> bool:
    # TODO: we can add the meta config according to xi earlier
    for meta_config in NEW_META_CONFIGS:
        flag = True
        for xpred, c in cc.items():
            base = sum([meta_config[idx] for idx in Xpred_to_NewCellIndices[xpred]])
            if base > 0 and c < base:
                flag = False
                break
            if base == 0 and c != 0:
                flag = False
                break
        if flag:
            return True
    return False

def update_PZ_evidence(evidence_dict: dict[Const, Pred], 
                      cc_xpred: dict[Pred, int],
                      relation: frozenset[AtomicFormula],
                      first_e: Const, second_e: Const):
    cc_xpred[evidence_dict[first_e]] -= 1
    cc_xpred[evidence_dict[second_e]] -= 1
    # update x pred when adding pi
    evidence_dict[second_e] = Evi_to_Xpred[frozenset(Xpred_to_Evi[evidence_dict[second_e]]|{Rel_to_Pi[relation](X)})]
    for atom in relation:
        if atom.pred.name.startswith('@R') and atom.positive:
            z_pred = Rpred_to_Zpred[atom.pred]
            firt_arg = first_e if atom.args[0] == X else second_e
            if z_pred(X) in Xpred_to_Evi[evidence_dict[firt_arg]]:
                # update x pred when deleting block (z)
                evidence_dict[firt_arg] = Evi_to_Xpred[frozenset(
                    {atom for atom in Xpred_to_Evi[evidence_dict[firt_arg]] if atom != z_pred(X)})]
    # update cc when considering the new relation
    cc_xpred[evidence_dict[first_e]] += 1
    cc_xpred[evidence_dict[second_e]] += 1
    
def clean_P_evidence(evidence_dict: dict[Const, Pred], cc_xpred: dict[Pred, int]):
    for e, xpred in evidence_dict.items():
        cc_xpred[xpred] -= 1
        evidence_dict[e] = Evi_to_Xpred[frozenset(
            {atom for atom in Xpred_to_Evi[xpred] 
                    if not atom.pred.name.startswith('@P')})]
        cc_xpred[evidence_dict[e]] += 1
        
def update_A_evidence(ego_element: Const, evidence_dict: dict[Const, Pred], cc_xpred: dict[Pred, int]):
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
        print(MC, remove_aux_atoms(cur_model))
        return

    ego_element = domain.pop()
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
        update_PZ_evidence(new_evidence_dict, new_cc_xpred, 
                           rel, ego_element, cur_element)
        # if we move the transform of CC to function SAT, 
        # we need to iterate all elements in the domain to get the CC (complexity: O(n))
        if sat_cc(new_cc_xpred):
            ego_structure_sampling(
                ego_element, domain_todo.copy(), domain_done.copy(),
                new_evidence_dict, new_cc_xpred,
                cur_model|{atom.substitute({X: ego_element, Y: cur_element}) for atom in rel})

def build_two_tables(formula: QFFormula, cells: list[Cell]):
    models = dict()
    gnd_formula: QFFormula = ground_on_tuple(formula, a, b) & ground_on_tuple(formula, b, a)
    gnd_formula = gnd_formula.substitute({a: X, b: Y})
    gnd_lits = gnd_formula.atoms()
    gnd_lits = gnd_lits.union(frozenset(map(lambda x: ~x, gnd_lits)))
    for model in gnd_formula.models():
        model = frozenset({atom for atom in model if atom.args[0] != atom.args[1]})
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

def add_relation_formulas(formula: QFFormula):
    global Pred_Pi_Set
    global Pi_to_Rel
    global Rel_to_Pi
    global Rel_Dict
    relations, Rel_Dict = build_two_tables(context.uni_formula, original_cells)
    logger.info('The new formula for relations:')
    p_preds: set[Pred] = set()
    for _, rel in enumerate(relations):
        twotable = top
        for lit in rel:
            twotable = twotable & lit
        new_pred_P = new_predicate(1, ENUM_P_PRED_NAME)
        Pred_Pi_Set.add(new_pred_P)
        Pi_to_Rel[new_pred_P] = rel
        Rel_to_Pi[rel] = new_pred_P
        p_preds.add(new_pred_P)
        rel_formula = Implication(Pred_A(X) & new_pred_P(Y), twotable)
        logger.info(' %s', rel_formula)
        formula = formula & rel_formula
    logger.info('The exactly one formula of Pi:')
    exactly_one_pi = exactly_one_qf(p_preds)
    return formula & exactly_one_pi
    
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
    
def generate_configs(domain_size, template_configs):
    """
    Generate all satisfiable configurations based on the template configurations.
    """
    for template_config in template_configs:
        indeces = list()
        remaining = domain_size - sum(template_config)
        logger.info('Base config: %s', template_config)
        if remaining > 0:
            for idx, ni in enumerate(template_config):
                if ni == ORIGINAL_DELTA:
                    indeces.append(idx)
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

    with Timer() as t:
        problem = parse_input(args.input)
    logger.info('Parse input: %ss', t)

    context: EnumContext = EnumContext(problem)
    domain_size = len(context.domain)
    ORIGINAL_UNI_FORMULA = context.uni_formula
    ORIGINAL_EXI_FORMULAS = context.ext_formulas
    ORIGINAL_DELTA = max(2*len(ORIGINAL_EXI_FORMULAS)+1, 
                         len(ORIGINAL_EXI_FORMULAS)*(len(ORIGINAL_EXI_FORMULAS)+1))
    
    # This is used to enumerate satisfiable configurations
    # and get the init evidence of each element
    temp_problem = WFOMCSProblem(SC2(uni_formula=ORIGINAL_UNI_FORMULA, ext_formulas=ORIGINAL_EXI_FORMULAS), 
                                 context.domain, context.weights)
    temp_context: WFOMSContext = WFOMSContext(temp_problem)
    temp_skolem_cell_graph: CellGraph = CellGraph(temp_context.formula, temp_context.get_weight)
    temp_uni_cell_graph: CellGraph = CellGraph(temp_context.uni_formula, temp_context.get_weight)
    template_configs = get_template_config(temp_context, temp_skolem_cell_graph, temp_uni_cell_graph, domain_size)
    # NOTE: we need to keep the order of the cells with the config
    original_cells: list[Cell] = [cell.drop_preds(prefixes=['@aux']) for cell in temp_uni_cell_graph.cells]
    
    # introduce block predicates
    # NOTE: the transformated sentence is not equal to the original one if without evidence
    context.transform_block()
    z_preds = context.tseitin_preds
    Zpred_to_Rpred = context.z2r
    for z, r in Zpred_to_Rpred.items():
        Rpred_to_Zpred[r] = z
    
    # init evidence set including 1-type, block type and the predicate A
    init_evi_dict: dict[Cell, frozenset[AtomicFormula]] = {}
    for cell in original_cells:
        evi = set(cell.get_evidences(X)|{~Pred_A(X)})
        evi.update({z(X) for z in z_preds 
                            if not cell.is_positive(Zpred_to_Rpred[z])})
        init_evi_dict[cell] = frozenset(evi)
    
    
    
    # TODO: move this to the section that meta-configuration is determined
    # and pi cases may be smaller
    uni_formula_with_P = add_relation_formulas(context.uni_formula)
    logger.info('The formula after adding relation formulas: \n%s', uni_formula_with_P)
    p_lit_comb = [{p(X)} for p in Pred_Pi_Set] + [set()]
    # here we need to consider all possible combinations of Zi predicates
    z_lit_comb = [set(i) for r in range(len(z_preds) + 1) 
                            for i in combinations([pred(X) 
                                for pred in z_preds], r)]
    
    # X preds and evi formulas
    cell_to_evis: dict[Cell, list[set[AtomicFormula]]] = {}
    evi_to_cell: dict[set[AtomicFormula], Cell] = {}
    cell_to_xpreds: dict[Cell, list[Pred]] = {}
    cell_to_evi_formulas: dict[Cell, list[QFFormula]] = {}
    x_preds: set[Pred] = set()
    
    for cell in original_cells:
        cell_cases: set[AtomicFormula] = set(cell.get_evidences(X))
        cell_block_combs = [cell_cases|z for z in z_lit_comb]
        add_A = [{Pred_A(X)}|s for s in cell_block_combs]
        # For an element e, each Pi(e) is determined by A(e) and Cell(e).
        # So we do not need to consider the case of Pi(e) when A(e) is true
        add_negA = [{~Pred_A(X)}|s for s in cell_block_combs]
        add_negA_P = [set.union(*tup) for tup in product(add_negA, p_lit_comb)]
        cell_to_evis[cell] = add_A + add_negA_P
        for atom_set in cell_to_evis[cell]:
            evi_to_cell[frozenset(atom_set)] = cell
    
    for cell in original_cells:
        cell_to_evi_formulas[cell] = []
        cell_to_xpreds[cell] = []
        for atom_set in cell_to_evis[cell]:
            evidence_type = top
            for atom in atom_set:
                evidence_type = evidence_type & atom
            new_x_pred = new_predicate(1, ENUM_X_PRED_NAME)
            Xpred_to_Evi[new_x_pred] = frozenset(atom_set)
            Evi_to_Xpred[frozenset(atom_set)] = new_x_pred
            x_preds.add(new_x_pred)
            new_atom = new_x_pred(X)
            cell_to_xpreds[cell].append(new_x_pred)
            cell_to_evi_formulas[cell].append(new_atom.implies(evidence_type))
    
    uni_formula_with_XP = uni_formula_with_P
    for cell in original_cells:
        for evi_formula in cell_to_evi_formulas[cell]:
            logger.info(' %s', evi_formula)
            uni_formula_with_XP = uni_formula_with_XP & evi_formula
    uni_formula_with_XP = uni_formula_with_XP & exactly_one_qf(x_preds)
    uni_formula_with_XP = to_sc2(uni_formula_with_XP).uni_formula
    NEW_UNI_FORMULA = uni_formula_with_XP
    
    
    original_level = logger.getEffectiveLevel()
    logger.setLevel(logging.CRITICAL)
    try:
        # meta configuration of the current formula       
        new_sc2 = SC2(uni_formula=NEW_UNI_FORMULA, ext_formulas=ORIGINAL_EXI_FORMULAS.copy())
        new_weightings: dict[Pred, tuple] = dict()
        # for pred in NEW_UNI_FORMULA.preds():
        #     new_weightings[pred] = (Rational(1, 1), Rational(1, 1))
        new_problem = WFOMCSProblem(new_sc2, context.domain, new_weightings)
        new_context: EnumContext2 = EnumContext2(new_problem)
        new_skolen_cell_graph: CellGraph = CellGraph(new_context.formula, new_context.get_weight)
        new_uni_cell_graph: CellGraph = CellGraph(NEW_UNI_FORMULA, new_context.get_weight)
    finally:
        logger.setLevel(original_level)
    
    new_cells = new_uni_cell_graph.cells
    Xpred_to_NewCellIndices = {x: set() for x in x_preds}
    for cell in new_cells:
        for x in x_preds:
            if cell.is_positive(x):
                Xpred_to_NewCellIndices[x].add(new_cells.index(cell))
                break
    
    NEW_META_CONFIGS = set()
    for i in range(1, domain_size+1):
        NEW_META_CONFIGS.update(get_template_config(new_context, 
                                            new_skolen_cell_graph, new_uni_cell_graph, i))
    
    # enumerate satisfiable configs
    for config in generate_configs(domain_size, template_configs):
        logger.info('The configuration: %s', config)
        # init cardinality constraint for each X pred
        cc_xpred: dict[Pred, int] = {x: 0 for x in x_preds}
        for cell, num in zip(original_cells, config):
            cc_xpred[Evi_to_Xpred[init_evi_dict[cell]]] += num
        # assign 1-types for all elements
        for idx, domain_partitions in enumerate(distribute_cells(context.domain, config)):
            logger.info('The distribution: \n%s', domain_partitions)
            # init evidence set (1-type, block type and negative A)
            evidence_dict: dict[Const, Pred] = {}
            for cell, elements in zip(original_cells, domain_partitions):
                for element in elements:
                    Domain_to_Cell[element] = cell
                    evidence_dict[element] = Evi_to_Xpred[init_evi_dict[cell]]
            logger.info('The init evidence: \n%s', evidence_dict)
            domain_recursion(context.domain.copy(), evidence_dict, cc_xpred.copy())
    logger.info('The number of models: %s', MC)