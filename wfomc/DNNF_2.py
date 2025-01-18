from __future__ import annotations

import os
import argparse
import logging
import logzero
import functools
from logzero import logger
from contexttimer import Timer
from itertools import combinations
from enum import Enum
import copy

import numpy as np
from cython_modules.matrix_utils import swap_rows_cols

from wfomc.config_sat import ConfigSAT
from wfomc.parser import parse_input
from wfomc.fol.syntax import AtomicFormula, Const, Pred, X, Y, top, QFFormula
from wfomc.utils import MultinomialCoefficients, multinomial
from wfomc.cell_graph.components import Cell

from enum_context import EnumContext
from wfomc.enum_utils import Pred_P, Pred_A, remove_aux_atoms


NODE_INDEX = 0
NODES = []

class NodeType(Enum):
    AND = 'and'
    OR = 'or'
    LEAF = 'leaf'
    
    def __str__(self):
        return self.value

class Node:
    def __init__(self, type: NodeType, value: int = None):
        self.type = type
        self.children = []
        if type == NodeType.LEAF:
            self.children.append(value)
        global NODES, NODE_INDEX
        self.index = NODE_INDEX
        NODE_INDEX += 1
        NODES.append(self)
        self.status: dict[Const, any] = {}
    
    def __str__(self):
        if self.type == NodeType.LEAF:
            return f'L {self.children[0]}'
        else:
            return f'{self.index} {self.type} {self.children}'

def create_node(type: NodeType, value: int = None):
    return Node(type, value)


STATISTICS_META_CONFIG: dict[tuple[int], int] = dict()
DOMAIN_SIZE: int = 0
MODEL_COUNT: int = 0
SAT_COUNT: int = 0
DELTA: int = 0


CACHE_OF_MODELS = None

EARLY_STOP: bool = True
MC_LIMIT: int = 1000000

Domain_to_Cell: dict[Const, Cell] = {}
Rel_Dict: dict[tuple[Cell, Cell], list[frozenset[AtomicFormula]]] = {}

Rpred_to_Zpred: dict[Pred, Pred] = {}
Zpred_to_Rpred: dict[Pred, Pred] = {}

Evi_to_Xpred: dict[frozenset[AtomicFormula], Pred] = {}
Xpred_to_Evi: dict[Pred, frozenset[AtomicFormula]] = {}

# NOTE(lucien): for checking the satifiability of configs
Config_SAT_pre: ConfigSAT = None

Config_SAT_unary: ConfigSAT = None

class ExitRecursion(Exception):
    pass

def witness_relation(ego_element: Const, witness: Const, block_idx: int) -> list[frozenset[AtomicFormula]]:
    result = []
    for rel in Rel_Dict[(Domain_to_Cell[ego_element], Domain_to_Cell[witness])]:
        for atom in rel:
            if atom.pred.name.startswith('@R') and atom.positive and \
                    int(atom.pred.name[2:]) == block_idx and atom.args[0] == X:
                result.append(rel)
                break
    return result

def check_sat(block_status: dict[Const, list[bool]], sat_matrix: dict[Const, list[list[Const]]]):
    for ego_element, blocks in block_status.items():
        if not any(blocks):
            continue
        for block_idx, block in enumerate(blocks):
            if not block:
                continue
            if len(sat_matrix[ego_element][block_idx]) == 0:
                return False
            # 先只考虑一个 1-type 的情况
            for witness in sat_matrix[ego_element][block_idx]:
                for rel in witness_relation(ego_element, witness, block_idx):
                    block_status_copy = copy.deepcopy(block_status)
                    sat_matrix_copy = copy.deepcopy(sat_matrix)
                    # update block status
                    for atom in rel:
                        if atom.pred.name.startswith('@R') and atom.positive:
                            if atom.args[0] == X:
                                block_status_copy[ego_element][int(atom.pred.name[2:])] = False
                            else:
                                block_status_copy[witness][int(atom.pred.name[2:])] = False
                    # update sat matrix
                    for witness_list in sat_matrix_copy[ego_element]:
                        if witness_list == None:
                            continue
                        witness_list.remove(witness)
                    for witness_list in sat_matrix_copy[witness]:
                        if witness_list == None:
                            continue
                        witness_list.remove(ego_element)

                    if check_sat(block_status_copy, sat_matrix_copy):
                        return True  
    sat_flag = True
    for blocks in block_status.values():
        if not any(blocks):
            continue
        sat_flag = False
        break 
    return sat_flag

def domain_recursion(domain: list[Const],
                     evidence_dict: dict[Const, Pred],
                     cc_xpred: dict[Pred, int],
                     cur_model, 
                     subroot_node: Node = None,
                     block_status: dict[Const, list[bool]] = None, 
                     sat_matrix: dict[Const, list[list[Const]]] = None):
    if domain.__len__() == 0:
        # exit(1)
        return

    ego_element = domain[0]
    domain = domain[1:]
    
    # if the element have no block
    if not any(block_status[ego_element]):
        domain_recursion(domain, evidence_dict, cc_xpred, cur_model, subroot_node, block_status, sat_matrix)
        return
    
    for block_idx, block in enumerate(block_status[ego_element]):
        if not block:
            continue
        # TODO: select a witness
        for witness in sat_matrix[ego_element][block_idx]:
            assert witness != None
            for rel in witness_relation(ego_element, witness, block_idx):
                
                if not check_sat(block_status, sat_matrix):
                    continue
                
                block_status_copy = copy.deepcopy(block_status)
                sat_matrix_copy = copy.deepcopy(sat_matrix)
                
                and_node = create_node(NodeType.AND)
                and_node.children.append(f'R{block_idx}({ego_element},{witness})')
                new_subroot_node = create_node(NodeType.OR)
                and_node.children.append(new_subroot_node.index)
                subroot_node.children.append(and_node.index)
                # update block status
                for atom in rel:
                    if atom.pred.name.startswith('@R') and atom.positive:
                        if atom.args[0] == X:
                            block_status_copy[ego_element][int(atom.pred.name[2:])] = False
                        else:
                            block_status_copy[witness][int(atom.pred.name[2:])] = False
                # update sat matrix
                for witness_list in sat_matrix_copy[ego_element]:
                    if witness_list == None:
                        continue
                    witness_list.remove(witness)
                for witness_list in sat_matrix_copy[witness]:
                    if witness_list == None:
                        continue
                    witness_list.remove(ego_element)
                domain_recursion(domain, evidence_dict, cc_xpred, cur_model, new_subroot_node, block_status_copy, sat_matrix_copy)

def generate_config_class(domain_size:int, len_config:int, delta:int):
    '''
    Generate all possible tuple (k_0, k_delta) where
    k_0 is the number of cells that have no element and
    k_delta is the number of cells that have delta elements.
    '''
    for k_zero in range(len_config):
        if (len_config - k_zero) * (delta - 1) >= domain_size:
            yield (k_zero, 0)
        for k_delta in range(1, len_config + 1 - k_zero):
            if k_delta * delta + (len_config - k_zero - k_delta) > domain_size:
                break
            if delta == 1 and k_delta + k_zero != len_config:
                continue
            yield (k_zero, k_delta)

def generate_base_configs(len_config, k_zero, k_delta, delta):
    '''
    A base config is a config consisting of {0, 1, delta}
    Generate all possible base configurations based on the tuple (k_0, k_delta)
    A base config may be unsat
    '''
    def backtrack(cur_tuple: tuple, zero_left, sup_left):
        if sum(cur_tuple) > DOMAIN_SIZE:
            return
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
    yield from backtrack(tuple(), k_zero, k_delta)

TEMPLATE_CONFIGS = set()
def generate_template_config(base_config: tuple, flag: bool, delta: int, domain_size: int):
    '''
    A template config is a sat config consisting of (0, x, delta) where 1<=x<delta
    generate the template configurations based on the current config.
    '''
    if sum(base_config) > domain_size:
        return
    # flag==True means that the current config is based on a template config
    if not flag:
        # TODO: use sat_cc instead of sat_config
        flag = Config_SAT_unary.check_config_by_pysat(base_config)
    if flag:
        global TEMPLATE_CONFIGS
        if sum(base_config) == domain_size:
            if base_config not in TEMPLATE_CONFIGS:
                TEMPLATE_CONFIGS.add(base_config)
                yield base_config
            return
        elif sum(base_config) < domain_size and any(ni == delta for ni in base_config):
            if base_config not in TEMPLATE_CONFIGS:
                TEMPLATE_CONFIGS.add(base_config)
                yield base_config
        else:
            pass

    for i in range(len(base_config)):
        if base_config[i] == 0 or base_config[i] == delta :
            continue
        if base_config[i]+1 == delta:
            continue
        inc = list(base_config)
        inc[i] += 1
        yield from generate_template_config(tuple(inc), flag, delta, domain_size)

def generate_sat_configs(domain_size, template_config, delta):
    """
    Generate all satisfiable configurations based on the template configuration.
    """
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

def cell_assignment(domain: set[Const], config: tuple[int]):
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

def get_domain_order(cells: list[Cell],
                    cell_correlation: dict[tuple[Cell, Cell], int],
                    config: tuple[int]):

    cell_to_index: dict[Cell, int] = dict()
    for index, cell in enumerate(cells):
        cell_to_index[cell] = index

    cell_importance: dict[Cell, int] = dict()
    for i, cell_i in enumerate(cells):
        cell_importance[cell_i] = 0
        if config[i] == 0:
            continue
        for j, cell_j in enumerate(cells):
            if i == j:
                cell_importance[cell_i] += cell_correlation[(cell_i, cell_j)] * (config[j] - 1)
            else:
                cell_importance[cell_i] += cell_correlation[(cell_i, cell_j)] * config[j]

    # sorted_cells = sorted(cell_importance, key=lambda x: cell_importance[x], reverse=True)
    max_cell = max(cell_importance, key=cell_importance.get)

    config = list(config)
    domain_order = [cell_to_index[max_cell]]
    config[cell_to_index[max_cell]] -= 1
    while len(domain_order) != DOMAIN_SIZE:
        max_cell = None
        max_correlation = (-1,)*len(domain_order)
        for cell in cells:
            if config[cell_to_index[cell]] == 0:
                continue
            cur_correlation = ()
            for e in domain_order:
                cur_correlation = (cell_correlation[(cell, cells[e])], ) + cur_correlation
            if cur_correlation > max_correlation:
                max_cell = cell
                max_correlation = cur_correlation
        domain_order.append(cell_to_index[max_cell])
        config[cell_to_index[max_cell]] -= 1

    domain_order.reverse()
    return domain_order

def get_domain_list(domain_order, domain_partition):
    domain_list = []
    domain_partition_copy = []
    for lst in domain_partition:
        domain_partition_copy.append(lst.copy())
    for i in domain_order:
        domain_list.append(domain_partition_copy[i].pop())
    return domain_list

def parse_args():
    parser = argparse.ArgumentParser(
        description='Enumerate models of a given sentence',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', type=str, required=True, help='wfomcs file')
    parser.add_argument('--domain_size', '-n', type=int, help='domain_size')
    parser.add_argument('--output_dir', '-o', type=str, default='./check-points')
    parser.add_argument('--log_level', '-log', type=str, default='INFO')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    import sys
    sys.setrecursionlimit(int(1e6))
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.log_level == 'D':
        logzero.loglevel(logging.DEBUG)
    elif args.log_level == 'C':
        logzero.loglevel(logging.CRITICAL)
    else:
        logzero.loglevel(logging.INFO)

    # logger.setLevel(logging.CRITICAL)
    # logger.setLevel(logging.INFO)

    logzero.logfile('{}/log.txt'.format(args.output_dir), mode='w')

    problem = parse_input(args.input)
    if args.domain_size:
        problem.domain = {Const(f'e{i}') for i in range(args.domain_size)}
        
    context: EnumContext = EnumContext(problem)

    DELTA = context.delta
    DOMAIN_SIZE = len(context.domain)
    MultinomialCoefficients.setup(DOMAIN_SIZE)

    original_cells: list[Cell] = context.original_cells
    original_original_cell_correlation: dict[tuple[Cell, Cell], int] = context.original_cell_correlation

    # Zpred_to_Rpred = context.zpred_to_rpred
    Rpred_to_Zpred = context.rpred_to_zpred

    Rel_Dict = context.rel_dict

    x_preds = context.x_preds
    xpreds_with_P = context.xpreds_with_P
    
    Evi_to_Xpred = context.Evi_to_Xpred
    Xpred_to_Evi = context.Xpred_to_Evi

    auxiliary_uni_formula_cell_graph = context.auxiliary_uni_formula_cell_graph
    auxiliary_cells: list[Cell] = auxiliary_uni_formula_cell_graph.cells
    
    Config_SAT_unary = ConfigSAT(context.original_uni_formula,
                                 context.original_ext_formulas,
                                 context.original_cells,
                                 DELTA, False)



    with Timer() as t_enumaration:
        try:
            # enumerate satisfiable configs
            ori_len_config = len(context.original_cells)
            ori_delta = context.delta
            for cfg_class in generate_config_class(DOMAIN_SIZE, ori_len_config, ori_delta):
                for base_config in generate_base_configs(ori_len_config, cfg_class[0], cfg_class[1], ori_delta):
                    for tpl_config in generate_template_config(base_config, False, ori_delta, DOMAIN_SIZE):
                        for config in generate_sat_configs(DOMAIN_SIZE, tpl_config, ori_delta):
                            logger.info('The configuration: %s', config)
                            domain_order = get_domain_order(original_cells,
                                                          original_original_cell_correlation,
                                                          config)
                            # init cardinality constraint for each X pred
                            cc_xpred: dict[Pred, int] = {x: 0 for x in x_preds}
                            for cell, num in zip(original_cells, config):
                                cc_xpred[Evi_to_Xpred[context.init_evi_dict[cell]]] += num
                            CACHE_OF_MODELS = []
                            firt_partition:list = None
                            # assign 1-types for all elements
                            for domain_partition in cell_assignment(context.domain, config):
                                if firt_partition != None:
                                    substitute_dict:dict = {}
                                    for first_par, cur_par in zip(firt_partition, domain_partition):
                                        set_1 = set(first_par)
                                        set_2 = set(cur_par)
                                        s1 = set_1 - set_2
                                        s2 = set_2 - set_1
                                        while len(s1) != 0:
                                            substitute_dict[s1.pop()] = s2.pop()

                                    for model in CACHE_OF_MODELS:
                                        new_model = model.copy()
                                        for k, v in substitute_dict.items():
                                            # continue
                                            # row_temp = new_model[k, :].copy()
                                            # new_model[k, :] = new_model[v, :]
                                            # new_model[v, :] = row_temp

                                            # col_temp = new_model[:, k].copy()
                                            # new_model[:, k] = new_model[:, v]
                                            # new_model[:, v] = col_temp
                                            swap_rows_cols(new_model, k, v)
                                        MODEL_COUNT += 1
                                        if EARLY_STOP and MODEL_COUNT > MC_LIMIT:
                                            raise ExitRecursion
                                    # MC += len(MODEL_SUBSET)
                                    continue
                                firt_partition = domain_partition
                                logger.debug('The distribution:')
                                # init evidence set (1-type, block type and negative A)
                                evidence_dict: dict[Const, Pred] = {}
                                for cell, elements in zip(original_cells, domain_partition):
                                    # logger.info(cell, " = ",elements)
                                    for element in elements:
                                        Domain_to_Cell[element] = cell
                                        evidence_dict[element] = Evi_to_Xpred[context.init_evi_dict[cell]]
                                logger.debug('The init evidence: \n%s', evidence_dict)
                                domain_list = get_domain_list(domain_order, domain_partition)
                                
                                block_status: dict[Const, list[bool]] = {}
                                for element in domain_list:
                                    block_status[element] = [True for _ in range(context._m)]
                                    element_cell = Domain_to_Cell[element]
                                    for pred in element_cell.preds:
                                        if pred.name.startswith('@R') and element_cell.is_positive(pred):
                                            pred_idx = int(pred.name[2:])
                                            block_status[element][pred_idx] = False
                                
                                cur_edges: list[list[Const]] = [[False for _ in range(DOMAIN_SIZE)] for _ in range(DOMAIN_SIZE)]
                                sat_matrix: dict[Const, list[list[Const]]] = {}
                                for element in domain_list:
                                    sat_matrix[element] = []
                                    for idx_ext, block in enumerate(block_status[element]):
                                        if not block:
                                            sat_matrix[element].append(None)
                                            continue
                                        e_list = []
                                        for element_2 in domain_list:
                                            if element == element_2:
                                                continue
                                            for _, rel in enumerate(Rel_Dict[(Domain_to_Cell[element], Domain_to_Cell[element_2])]):
                                                for atom in rel:
                                                    if atom.pred.name.startswith('@R') and atom.positive and \
                                                            int(atom.pred.name[2:]) == idx_ext and atom.args[0] == X:
                                                        e_list.append(element_2)
                                        assert len(e_list) != 0
                                        sat_matrix[element].append(e_list)
                                
                                root_node = create_node(NodeType.OR)
                                
                                domain_recursion(domain_list, evidence_dict, cc_xpred.copy(), 
                                                 np.full((DOMAIN_SIZE, DOMAIN_SIZE), -1, dtype=np.int8), root_node, block_status, sat_matrix)
                                
        except ExitRecursion:
            logger.info('Early stop, current MC: %s', MODEL_COUNT)
            pass

    enumeration_time = t_enumaration.elapsed
    logger.info('The number of models: %s', MODEL_COUNT)
    logger.info('time: %s', enumeration_time)

    if MODEL_COUNT != 0:
        avg_time = enumeration_time / MODEL_COUNT
    else:
        avg_time = -1
    res = f'{DOMAIN_SIZE}, {enumeration_time}, {MODEL_COUNT}, 0, {round(avg_time, 7)}\n'
    print(res)

    # for node in NODES:
    #     print(node)
     
    print()   
    
    # 删去
    
    to_remove = []
    for idx_ext, node in enumerate(NODES):
        if node.type == NodeType.LEAF:
            continue
        if len(node.children) != 0:
            continue
        to_remove.append(idx_ext)
        redundant_node_index = node.index
        for node_2 in NODES:
            if redundant_node_index in node_2.children:
                node_2.children.remove(redundant_node_index)
    
    for idx_ext, node in enumerate(NODES):
        if node.type == NodeType.LEAF:
            continue
        if len(node.children) != 1:
            continue
        to_remove.append(idx_ext)
        redundant_node_index = node.index
        unique_value = node.children[0]
        for node_2 in NODES:
            if redundant_node_index in node_2.children:
                node_2.children[node_2.children.index(redundant_node_index)] = unique_value
    
    new_nodes = []
    circuit_size = 0
    for idx_ext, node in enumerate(NODES):
        if idx_ext in to_remove:
            continue
        new_nodes.append(node)
        circuit_size += len(node.children)
        # print(node)
    print('Circuit size:', circuit_size)
    print('Gate count:', NODE_INDEX)


    import networkx as nx
    import matplotlib.pyplot as plt
    from networkx.drawing.nx_agraph import graphviz_layout
    G = nx.DiGraph()
    leaf_counter = 0

    for node in new_nodes:
        node_id = str(node.index)
        node_type = node.type
        children = node.children
        
        # 为节点添加属性
        G.add_node(node_id, label=node_type)
        
        # 遍历子节点列表
        for child in children:
            # 如果子节点是整数，表示是另一个图中的节点
            if isinstance(child, int):
                child = str(child)
                G.add_edge(node_id, child)
            # 如果是字符串，则认为是叶子节点
            elif isinstance(child, str):
                # 用 leaf_counter 来生成一个唯一的标识
                leaf_id = f"leaf_{leaf_counter}"
                leaf_counter += 1
                G.add_node(leaf_id, label=child)
                G.add_edge(node_id, leaf_id)

    # 获取节点标签，用于绘图显示
    labels = nx.get_node_attributes(G, 'label')
    for k,node_type in labels.items():
        if node_type == NodeType.LEAF:
            labels[k] = f'{v}_{k}'
        elif node_type == NodeType.AND:
            labels[k] = '⋀'
        elif node_type == NodeType.OR:
            labels[k] = '⋁'
        else:
            pass

    # # 这里用 spring_layout 布局，你也可以选择其他布局（例如：nx.shell_layout(G)）
    # pos = nx.spring_layout(G)
    # # 绘制图形
    # plt.figure(figsize=(8, 6))
    # nx.draw(G, pos, with_labels=True, labels=labels, node_color='lightblue', 
    #         arrows=True, arrowstyle='-|>', arrowsize=12, node_size=1500, font_size=10)
    # plt.title("有向无环图示例")
    # plt.show()
    
    # 使用 graphviz 的 dot 布局
    pos = graphviz_layout(G, prog='dot', root='0')  # 假设根节点 ID 是 '0'

    # 绘制图形
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos,
            with_labels=True, 
            labels=labels, 
            node_color='lightblue', 
            arrows=True, 
            arrowstyle='-|>', 
            arrowsize=12, 
            node_size=1500, 
            font_size=10,
            verticalalignment='bottom')
    plt.show()
    
    # dot_filename = "/Users/mengqiaolan/wfoms/counting_fo2/wfomc/dnnf_n=3.dot"
    # nx.nx_agraph.write_dot(G, dot_filename)
    # print(f"DAG 图已保存为 DOT 文件: {dot_filename}")
    
    