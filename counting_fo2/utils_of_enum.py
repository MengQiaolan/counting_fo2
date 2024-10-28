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

def get_cells(preds: tuple[Pred], formula: QFFormula):
    gnd_formula_cc: QFFormula = ground_on_tuple(formula, c)
    cells = []
    code = {}
    for model in gnd_formula_cc.models():
        for lit in model:
            code[lit.pred] = lit.positive
        cells.append(Cell(tuple(code[p] for p in preds), preds))
    return cells