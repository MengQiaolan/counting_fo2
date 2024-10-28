from __future__ import annotations
from logzero import logger
from counting_fo2.fol.sc2 import SC2, to_sc2
from counting_fo2.fol.utils import new_predicate, convert_counting_formula

from counting_fo2.network.constraint import CardinalityConstraint
from counting_fo2.fol.syntax import *
from counting_fo2.problems import WFOMCSProblem
from counting_fo2.fol.syntax import AUXILIARY_PRED_NAME, SKOLEM_PRED_NAME, ENUM_R_PRED_NAME, ENUM_Z_PRED_NAME
from counting_fo2.utils.third_typing import RingElement, Rational


class EnumContext(object):
    """
    Context for WFOMC algorithm
    """

    def __init__(self, problem: WFOMCSProblem):
        self.domain: set[Const] = problem.domain
        self.sentence: SC2 = problem.sentence
        self.weights: dict[Pred, tuple[Rational, Rational]] = problem.weights
        self.cardinality_constraint: CardinalityConstraint = problem.cardinality_constraint
        # self.repeat_factor = 1

        logger.info('input sentence: \n%s', self.sentence)
        logger.info('domain: \n%s', self.domain)
        logger.info('weights:')
        for pred, w in self.weights.items():
            logger.info('%s: %s', pred, w)
        logger.info('cardinality constraint: %s', self.cardinality_constraint)

        self.uni_formula: QFFormula
        self.ext_formulas: list[QuantifiedFormula] = []
        self.tseitin_preds: set[Pred] = set()
        self.z2r: dict[Pred, Pred] = {}
        
        self._build()
        # logger.info('Skolemized formula for WFOMC: \n%s', self.uni_formula)
        # logger.info('weights for WFOMC: \n%s', self.weights)

    def contain_cardinality_constraint(self) -> bool:
        return self.cardinality_constraint is not None and \
            not self.cardinality_constraint.empty()

    def contain_existential_quantifier(self) -> bool:
        return self.sentence.contain_existential_quantifier()

    def get_weight(self, pred: Pred) -> tuple[RingElement, RingElement]:
        default = Rational(1, 1)
        if pred in self.weights:
            return self.weights[pred]
        return (default, default)

    # def decode_result(self, res: RingElement):
    #     if not self.contain_cardinality_constraint():
    #         return res / self.repeat_factor
    #     res = self.cardinality_constraint.decode_poly(res)
    #     return res / self.repeat_factor

    def _build(self):
        self.uni_formula = self.sentence.uni_formula
        while(not isinstance(self.uni_formula, QFFormula)):
            self.uni_formula = self.uni_formula.quantified_formula
        
        for formula in self.sentence.ext_formulas:
            quantified_formula = formula.quantified_formula.quantified_formula
            new_pred = new_predicate(2, ENUM_R_PRED_NAME)
            new_atom = new_pred(X,Y)
            formula.quantified_formula.quantified_formula = new_atom
            self.ext_formulas.append(formula)
            self.uni_formula = self.uni_formula & Equivalence(new_atom, quantified_formula)
        
        self.uni_formula = to_sc2(self.uni_formula).uni_formula
        
        logger.info('The universal formula after building: \n%s', self.uni_formula)
        logger.info('The existential formulas after building: \n%s', self.ext_formulas)

    def transform_block(self):
        for formula in self.ext_formulas:
            new_pred = new_predicate(1, ENUM_Z_PRED_NAME)
            self.tseitin_preds.add(new_pred)
            self.z2r[new_pred] = formula.quantified_formula.quantified_formula.pred
            new_atom = new_pred(X)
            quantified_formula = formula.quantified_formula.quantified_formula
            formula.quantified_formula.quantified_formula = to_sc2(Equivalence(new_atom, quantified_formula)).uni_formula
            logger.info('%s', formula)
        logger.info('The existential formulas after transformation of block type: \n%s', self.ext_formulas)
        logger.info('The map from tseitin predicates to existential predicates: \n%s', self.z2r)
        
        tseitin_evidence_set = set()
        for element in self.domain:
            for pred in self.tseitin_preds:
                tseitin_evidence_set.add(pred(element))
        logger.info('The (initial) evidence of tseitin predicates: \n%s', tseitin_evidence_set)

        return tseitin_evidence_set

class EnumContext2(object):

    def __init__(self, problem: WFOMCSProblem):
        self.sentence: SC2 = problem.sentence
        self.weights: dict[Pred, tuple[Rational, Rational]] = problem.weights

        self.uni_formula: QFFormula
        self.ext_formulas: list[QuantifiedFormula] = []
        
        self._build()
        
        self.formula = self.uni_formula

    def contain_existential_quantifier(self) -> bool:
        return self.sentence.contain_existential_quantifier()

    def get_weight(self, pred: Pred) -> tuple[RingElement, RingElement]:
        default = Rational(1, 1)
        if pred in self.weights:
            return self.weights[pred]
        return (default, default)

    def _build(self):
        self.uni_formula = self.sentence.uni_formula
        while(not isinstance(self.uni_formula, QFFormula)):
            self.uni_formula = self.uni_formula.quantified_formula
        
        self.ext_formulas = self.sentence.ext_formulas
        for ext_formula in self.ext_formulas:
            self._skolemize_one_formula(ext_formula)
        
        # self.uni_formula = to_sc2(self.uni_formula).uni_formula
        
    def _skolemize_one_formula(self, formula: QuantifiedFormula) -> QFFormula:
        """
        Only need to deal with \forall X \exists Y: f(X,Y) or \exists X: f(X,Y)
        """
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
        
        self.uni_formula = self.uni_formula & (skolem_atom | ~ quantified_formula)
        self.weights[skolem_pred] = (Rational(1, 1), Rational(-1, 1))
        
        
        

class FinalEnumContext(object):
    """
    Context for enumeration
    """
    def __init__(self, problem: WFOMCSProblem):
        self.domain: set[Const] = problem.domain
        self.sentence: SC2 = problem.sentence
        self.weights: dict[Pred, tuple[Rational, Rational]] = problem.weights
        
        self.original_uni_formula: QFFormula
        self.original_ext_formulas: list[QuantifiedFormula] = []
        self._scott()
        
        self._m = len(self.original_ext_formulas)
        self.original_delta = max(2*self._m+1, self._m*(self._m+1))
        
        self.z_preds: set[Pred] = set()
        self.z2r: dict[Pred, Pred] = {}
        self.uni_formula_with_Z: QFFormula
        self.ext_formulas_with_Z: list[QuantifiedFormula] = []
        
        logger.info('input sentence: \n%s', self.sentence)
        logger.info('domain: \n%s', self.domain)
        logger.info('weights:')
        for pred, w in self.weights.items():
            logger.info('%s: %s', pred, w)
        logger.info('cardinality constraint: %s', self.cardinality_constraint)

        self.uni_formula: QFFormula
        self.ext_formulas: list[QuantifiedFormula] = []
        
        
        # logger.info('Skolemized formula for WFOMC: \n%s', self.uni_formula)
        # logger.info('weights for WFOMC: \n%s', self.weights)

    def contain_existential_quantifier(self) -> bool:
        return self.sentence.contain_existential_quantifier()

    def get_weight(self, pred: Pred) -> tuple[RingElement, RingElement]:
        default = Rational(1, 1)
        if pred in self.weights:
            return self.weights[pred]
        return (default, default)

    def _scott(self):
        self.original_uni_formula = self.sentence.uni_formula
        while(not isinstance(self.original_uni_formula, QFFormula)):
            self.original_uni_formula = self.original_uni_formula.quantified_formula
        
        for formula in self.sentence.ext_formulas:
            quantified_formula = formula.quantified_formula.quantified_formula
            new_pred = new_predicate(2, ENUM_R_PRED_NAME)
            new_atom = new_pred(X,Y)
            formula.quantified_formula.quantified_formula = new_atom
            self.original_ext_formulas.append(formula)
            self.original_uni_formula = self.original_uni_formula & Equivalence(new_atom, quantified_formula)
        
        self.original_uni_formula = to_sc2(self.original_uni_formula).uni_formula
        
        logger.info('The universal formula: \n%s', self.original_uni_formula)
        logger.info('The existential formulas: \n%s', self.original_ext_formulas)

    def transform_block(self):
        for formula in self.original_ext_formulas:
            new_pred = new_predicate(1, ENUM_Z_PRED_NAME)
            self.z_preds.add(new_pred)
            self.z2r[new_pred] = formula.quantified_formula.quantified_formula.pred
            new_atom = new_pred(X)
            quantified_formula = formula.quantified_formula.quantified_formula
            formula.quantified_formula.quantified_formula = to_sc2(Equivalence(new_atom, quantified_formula)).uni_formula
            logger.info('%s', formula)
        logger.info('The existential formulas after introducing blocks: \n%s', self.ext_formulas)
        logger.info('The map from tseitin predicates Zi to existential predicates: \n%s', self.z2r)

        return 