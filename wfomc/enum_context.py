from __future__ import annotations
from logzero import logger
from itertools import combinations
from copy import deepcopy

from wfomc.fol.syntax import *
from wfomc.problems import WFOMCProblem
from wfomc.fol.sc2 import SC2, to_sc2
from wfomc.fol.utils import new_predicate, exactly_one_qf
from wfomc.enum_utils import ENUM_R_PRED_NAME, ENUM_Z_PRED_NAME, ENUM_T_PRED_NAME, \
    ENUM_P_PRED_NAME, ENUM_X_PRED_NAME, SKOLEM_PRED_NAME, build_two_tables, \
        Pred_A, Pred_D, Pred_P
from wfomc.utils.third_typing import RingElement, Rational
from wfomc.cell_graph.cell_graph import CellGraph, Cell, OptimizedCellGraph

class EnumContext(object):
    """
    Context for enumeration
    """
    def __init__(self, problem: WFOMCProblem):
        self.domain: set[Const] = problem.domain
        self.sentence: SC2 = problem.sentence
        self.weights: dict[Pred, tuple[Rational, Rational]] = problem.weights
        
        # build self.original_uni_formula and self.original_ext_formulas
        self.original_uni_formula: QFFormula = self.sentence.uni_formula
        self.original_ext_formulas: list[QuantifiedFormula] = []
        self._scott()
        
        self._m = len(self.original_ext_formulas)
        self.delta = max(2*self._m+1, self._m*(self._m+1)) if self._m > 0 else 1
        
        self.original_cell_graph: CellGraph = CellGraph(self.original_uni_formula, self.get_weight)
        self.original_cells: list[Cell] = self.original_cell_graph.cells
        self.oricell_to_tau: dict[Cell, Pred] = {cell: new_predicate(1, ENUM_T_PRED_NAME) for cell in self.original_cells}
        
        
        
        # ===================== Introduce @D predicates (relation) =====================
        
        self.uni_formula_D: QFFormula = self.original_uni_formula
        self.uni_formula_D = self.uni_formula_D & AtomicFormula(Pred_D, (X,X), True) 
        
        
        # ===================== Introduce @Z predicates (block type) =====================
        
        # build self.z_preds, self.zpred_to_rpred, self.rpred_to_zpred and self.ext_formulas_with_Z
        self.z_preds: set[Pred] = set()
        self.zpred_to_rpred: dict[Pred, Pred] = {}
        self.rpred_to_zpred: dict[Pred, Pred] = {}
        self.ext_formulas_Z: list[QuantifiedFormula] = []
        self.ext_formulas_ZD: list[QuantifiedFormula] = []
        self._introduce_block()
        
        # init evidence set including 1-type, block type and the predicate A
        self.init_evi_dict: dict[Cell, frozenset[AtomicFormula]] = {}
        for cell in self.original_cells:
            evi = set(cell.get_evidences(X)|{~Pred_A(X)}|{~Pred_P(X)})
            evi.update({z(X) for z in self.z_preds 
                                if not cell.is_positive(self.zpred_to_rpred[z])})
            self.init_evi_dict[cell] = frozenset(evi)
        
        # ===================== Skolemize and Introduce @T predicates =====================
        # =================== (neccessary to calculate template configs) ==================
        
        # build self.original_skolem_formula
        self.original_skolem_formula: QFFormula = self.original_uni_formula
        self._skolemize()
        
        # build self.skolem_tau_formula and self.skolem_tau_cell_graph
        self.skolem_tau_formula: QFFormula = self.original_skolem_formula
        self.skolem_tau_cell_graph: CellGraph
        self._add_tau_preds()
        
        
        
        # ===================== Introduce @P predicates (relation) =====================
        # ============== (neccessary to convert binary-evi to unary-evi) ===============
        
        self.relations: set[frozenset[AtomicFormula]] = None
        self.rel_dict: dict[tuple[Cell, Cell], set[frozenset[AtomicFormula]]] = {}
        self.relations, self.rel_dict = build_two_tables(self.original_uni_formula, self.original_cells)
        
        # build balabala...
        # self.p_preds: set[Pred] = set()
        # self.Pi_to_Rel: dict[Pred, frozenset[AtomicFormula]] = {}
        # self.Rel_to_Pi: dict[frozenset[AtomicFormula], Pred] = {}
        # self.Pi_to_elimZ: dict[Pred, set[Pred]] = {}
        # self.Pi_to_elimZ_in_A: dict[Pred, set[Pred]] = {}
        # self.rel_formulas: list[QFFormula] = []
        self.uni_formula_AP: QFFormula = self.uni_formula_D
        # self._introduce_rel_formulas()
        self._introduce_rel_formula()
        
        # ===================== Introduce @X predicates (evidence) =====================
        # ======================= (neccessary to evidence type) ========================
        
        # build balabala...
        self.x_preds: list[Pred] = list()
        self.evi_formulas: list[QFFormula] = []
        self.xpreds_with_P: set[Pred] = set()
        self.Evi_to_Xpred: dict[frozenset[AtomicFormula], Pred] = {}
        self.Xpred_to_Evi: dict[Pred, frozenset[AtomicFormula]] = {}
        self.uni_formula_APZX: QFFormula = self.uni_formula_AP
        self._introduce_evi_formulas()
        
        
        self.uni_APZX_cell_graph: CellGraph = CellGraph(self.uni_formula_APZX, self.get_weight)
        self.uni_APZX_cells: list[Cell] = self.uni_APZX_cell_graph.cells 
        
        
        # =================== Skolemize and Introduce @T predicates ===================
        # =================== (neccessary to calculate meta configs) ==================
        
        # build balabala...
        self.uni_APZX_cell_to_Tpred: dict[Cell, Pred] = \
                                    {cell: new_predicate(1, ENUM_T_PRED_NAME) 
                                            for cell in self.uni_APZX_cells}
        self.skolem_formula_APZXT = self.uni_formula_APZX
        self._skolem_with_T()



    def contain_existential_quantifier(self) -> bool:
        return self.sentence.contain_existential_quantifier()

    def get_weight(self, pred: Pred) -> tuple[RingElement, RingElement]:
        default = Rational(1, 1)
        if pred in self.weights:
            return self.weights[pred]
        return (default, default)

    def _scott(self):
        while(not isinstance(self.original_uni_formula, QFFormula)):
            self.original_uni_formula = self.original_uni_formula.quantified_formula

        for formula in self.sentence.ext_formulas:
            quantified_formula = formula.quantified_formula.quantified_formula
            new_pred = new_predicate(2, ENUM_R_PRED_NAME)
            new_atom = new_pred(X,Y)
            formula.quantified_formula.quantified_formula = new_atom
            self.original_ext_formulas.append(formula)
            self.original_uni_formula = self.original_uni_formula.__and__(new_atom.equivalent(quantified_formula))

        logger.info('The universal formula: \n%s', self.original_uni_formula)
        logger.info('The existential formulas: \n%s', self.original_ext_formulas)

    def _introduce_block(self):
        ext_formulas = self.original_ext_formulas
        for ext_formula in ext_formulas:
            new_pred = new_predicate(1, ENUM_Z_PRED_NAME)
            self.z_preds.add(new_pred)
            self.zpred_to_rpred[new_pred] = ext_formula.quantified_formula.quantified_formula.pred
            new_atom = new_pred(X)
            quantified_formula = ext_formula.quantified_formula.quantified_formula
            ext_formula.quantified_formula.quantified_formula = new_atom.equivalent(quantified_formula)
            self.ext_formulas_Z.append(ext_formula)
            
            ext_formula_D = deepcopy(ext_formula)
            ext_formula_D.quantified_formula.quantified_formula = new_atom.equivalent(quantified_formula & Pred_D(X,Y))
            self.ext_formulas_ZD.append(ext_formula_D)
            
        # NOTE: the transformated sentence is not equal to the original one if without Z evidence
        logger.info('The existential formulas after introducing blocks: \n%s', self.ext_formulas_Z)
        logger.info('The map from tseitin predicates Zi to existential predicates: \n%s', self.zpred_to_rpred)
        for z, r in self.zpred_to_rpred.items():
            self.rpred_to_zpred[r] = z
        return 
    
    def _skolemize_one_formula(self, formula: QuantifiedFormula) -> QFFormula:
        quantified_formula = formula.quantified_formula
        quantifier_num = 1
        while(not isinstance(quantified_formula, QFFormula)):
            quantified_formula = quantified_formula.quantified_formula
            quantifier_num += 1

        if quantifier_num == 2:
            skolem_pred = new_predicate(1, SKOLEM_PRED_NAME)
            skolem_atom = skolem_pred(X)
        elif quantifier_num == 1:
            skolem_pred = new_predicate(0, SKOLEM_PRED_NAME)
            skolem_atom = skolem_pred()
        self.weights[skolem_pred] = (Rational(1, 1), Rational(-1, 1))
        return (skolem_atom | ~ quantified_formula)
    
    def _skolemize(self):
        while(not isinstance(self.original_skolem_formula, QFFormula)):
            self.original_skolem_formula = self.original_skolem_formula.quantified_formula
        
        for ext_formula in self.original_ext_formulas:
            self.original_skolem_formula = self.original_skolem_formula \
                                    & self._skolemize_one_formula(ext_formula)
    
    def _add_tau_preds(self):
        for cell, tau in self.oricell_to_tau.items():
            new_atom = tau(X)
            new_formula = top
            for atom in cell.get_evidences(X):
                    new_formula = new_formula & atom
            new_formula = new_formula.equivalent(new_atom)
            self.skolem_tau_formula = self.skolem_tau_formula & new_formula
        self.skolem_tau_cell_graph = CellGraph(self.skolem_tau_formula, self.get_weight)
    
    def _introduce_rel_formula(self):
        rel_formula = Implication(Pred_A(X) & Pred_P(Y), ~Pred_D(X,Y) & ~Pred_D(Y,X))
        rel_formula = rel_formula & Implication(~(Pred_A(X) & Pred_P(Y) | Pred_A(Y) & Pred_P(X)), Pred_D(X,Y) & Pred_D(Y,X))
        logger.info('The new formula for predicates A and P: %s', rel_formula)
        self.uni_formula_AP = self.uni_formula_AP & to_sc2(rel_formula).uni_formula
        logger.info('The uni formula after adding A&P formulas: \n%s', self.uni_formula_AP)
        
    
    def _introduce_evi_formulas(self):
        # here we need to consider all possible combinations of Zi predicates
        z_lit_comb = [set(i) for r in range(len(self.z_preds) + 1) 
                                for i in combinations([pred(X) 
                                    for pred in self.z_preds], r)]
        # exactly one P predicate
        # p_lit_comb = [set()] + [{p(X)} for p in self.p_preds]

        # X preds and evi formulas
        all_evi: list[set[AtomicFormula]] = []
        evi_formulas: list[QFFormula] = []
        for cell in self.original_cells:
            cell_atoms: set[AtomicFormula] = set(cell.get_evidences(X))
            # we do not need to consider all comb of tau and Z
            # some Z are not neccessary when there are some tau
            # cell_block_combs = [cell_atoms|z for z in z_lit_comb]
            cell_block_combs = []
            for zs in z_lit_comb:
                rs = set(self.zpred_to_rpred[z.pred](X,X) for z in zs)
                if len(rs) != 0 and len(rs & cell_atoms) != 0:
                    logger.info('Impossible evidence type: %s', cell_atoms | zs)
                else:
                    cell_block_combs.append(cell_atoms | zs)
            
            add_A = [{Pred_A(X), ~Pred_P(X)}|s for s in cell_block_combs]
            # For an element e, each Pi(e) is determined by A(e) and Cell(e).
            # So we do not need to consider the case of Pi(e) when A(e) is true
            add_negA = [{~Pred_A(X)}|s for s in cell_block_combs]
            # we do not need to consider all comb of Z and P
            # some Z are not neccessary when there are some P 
            # add_negA_P = [set.union(*tup) for tup in product(add_negA, p_lit_comb)]
            add_negA_P = [{Pred_P(X)}|s for s in add_negA]
            add_negA_negP = [{~Pred_P(X)}|s for s in add_negA]
            all_evi = all_evi + add_A + add_negA_P + add_negA_negP
        
        
        def sort(evi):
            if Pred_A(X) in evi:
                return (1,0)
            for atom in evi:
                if atom.pred == Pred_P and atom.positive:
                    return (0,1)
            return (0,0)
        
        
        all_evi.sort(key=sort, reverse=True)

        for atom_set in all_evi:
            new_x_pred = new_predicate(1, ENUM_X_PRED_NAME)
            self.Xpred_to_Evi[new_x_pred] = frozenset(atom_set)
            self.Evi_to_Xpred[frozenset(atom_set)] = new_x_pred
            self.x_preds.append(new_x_pred)
            if Pred_P(X) in atom_set:
                self.xpreds_with_P.add(new_x_pred)
            new_atom = new_x_pred(X)
            
            evidence_type = top
            for atom in atom_set:
                evidence_type = evidence_type & atom
            evi_formulas.append(new_atom.implies(evidence_type))
            
        for evi_formula in evi_formulas:
            logger.info(' %s', evi_formula)
            self.uni_formula_APZX = self.uni_formula_APZX & evi_formula
        self.uni_formula_APZX = self.uni_formula_APZX & exactly_one_qf(self.x_preds)
        # self.uni_formula_APZX = to_sc2(self.uni_formula_APZX).uni_formula
        
    def map_Xpred_to_Uni_PZX_Cell(self):
        # when processing SAT with CC, we need Xpred_to_NewCellIndices
        # to get the corresponding cell index in a meta config for each @X pred
        Xpred_to_UniPZX_CellIdx = {x: set() for x in self.x_preds}
        for cell in self.uni_APZX_cells:
            for x in self.x_preds:
                if cell.is_positive(x):
                    Xpred_to_UniPZX_CellIdx[x].add(self.uni_APZX_cells.index(cell))
                    break
        Xpred_to_UniPZX_CellIdx_List = [Xpred_to_UniPZX_CellIdx[k] for k in self.x_preds]
        return Xpred_to_UniPZX_CellIdx, Xpred_to_UniPZX_CellIdx_List
    
    def _skolem_with_T(self):
        for cell, tau in self.uni_APZX_cell_to_Tpred.items():
            new_atom = tau(X)
            new_formula = top
            for atom in cell.get_evidences(X):
                    new_formula = new_formula & atom
            new_formula = new_formula.equivalent(new_atom)
            self.skolem_formula_APZXT = self.skolem_formula_APZXT & new_formula
            
        for ext_formula in self.ext_formulas_ZD:
            self.skolem_formula_APZXT = self.skolem_formula_APZXT \
                                    & self._skolemize_one_formula(ext_formula)
            