from .llm_micro_action import LLMMicroActionSet

import pandas as pd
import json


def get_budget(dataset, task, budget, path='../results/tables'):
    csv_path = f"{path}/{dataset}/{task}/final/0.csv"
    df = pd.read_csv(csv_path)
    max_idx = df['idx'].max() + 1
    budget = int(max_idx * budget)
    return budget

  
def get_micro_action_result(
    action: dict,
    llm_micro_action_set: LLMMicroActionSet,
    current_edge_set: list
):

    if action['action'] == 'add_fk_pk_edge':
        new_edge_set, graph_idx, error_msg = llm_micro_action_set.add_fk_pk_edge(
            current_edge_set=current_edge_set,
            from_table_name=action['parameters']['from_table_name'],
            from_col_name=action['parameters']['from_col_name'],
            to_table_name=action['parameters']['to_table_name']
        )
    elif action['action'] == 'remove_fk_pk_edge':
        new_edge_set, graph_idx, error_msg = llm_micro_action_set.remove_fk_pk_edge(
            current_edge_set=current_edge_set,
            from_table_name=action['parameters']['from_table_name'],
            from_col_name=action['parameters']['from_col_name'],
            to_table_name=action['parameters']['to_table_name']
        ) 

    elif action['action'] == 'convert_row_to_edge':
        new_edge_set, graph_idx, error_msg = llm_micro_action_set.convert_row_to_edge(
            current_edge_set=current_edge_set,
            table_1_name=action['parameters']['table_1_name'],
            table_2_name=action['parameters']['table_2_name'],
            edge_table_name=action['parameters']['edge_table_name']
        )
    elif action['action'] == 'convert_edge_to_row':
        new_edge_set, graph_idx, error_msg = llm_micro_action_set.convert_edge_to_row(
            current_edge_set=current_edge_set,
            table_1_name=action['parameters']['table_1_name'],
            table_2_name=action['parameters']['table_2_name'],
            edge_table_name=action['parameters']['edge_table_name']
        )

    else:
        # raise ValueError(f"Action {action['action']} is unsupported.")
        return None, None, None
    
    return new_edge_set, graph_idx, error_msg
        
def check_all_actions(
    actions: list [dict],
    llm_micro_action_set: LLMMicroActionSet,
    current_edge_set: tuple,
    perf_pred_dataset: dict
):  
    new_edge_sets, graph_idxs, error_msgs, scores = [], [], [], []
    for action in actions : 
        action['parameters'] = action['parameters'][0] if type(action['parameters']) == list else action['parameters']
        new_edge_set, graph_idx, error_msg = get_micro_action_result(
            action=action,
            llm_micro_action_set=llm_micro_action_set,
            current_edge_set=current_edge_set
        )
        if new_edge_set is None:
            continue
        # update results
        new_edge_sets.append(new_edge_set)
        graph_idxs.append(graph_idx)
        error_msgs.append(error_msg)
        score = perf_pred_dataset.get(graph_idx).y.item() if graph_idx != -1 else 0
        scores.append(score)

    if all(graph_idx == -1 for graph_idx in graph_idxs):
        return actions[0], new_edge_sets[0], graph_idxs[0], error_msgs[0]
    else:
        idx = scores.index(max(scores))
        return actions[idx], new_edge_sets[idx], graph_idxs[idx], error_msgs[idx]


            
def get_available_edges(full_edges: list):
    r2e_edges = []
    f2p_edges = []
    for edge in full_edges:
        if edge[1].startswith("r2e"):
            r2e_edges.append(edge)
        elif edge[1].startswith("f2p"):
            f2p_edges.append(edge)

    print(f"R2E edges: {r2e_edges}")
    print(f"F2P edges: {f2p_edges}")
    return r2e_edges, f2p_edges

def check_none_action(parsed_response_text):
    if not parsed_response_text:
        return True
    if len(parsed_response_text) == 0  : 
        return True
    
    parsed_response_text = [parsed_response_text] if type(parsed_response_text) == dict else parsed_response_text

    if 'action' not in parsed_response_text[0]:
        return True 
    elif not parsed_response_text[0]['action']:
        return True
    elif parsed_response_text[0]['action'].lower() == "none":
        return True
    return False


def get_changed_edge(before_edge_set, after_edge_set, full_edges):

    changed_edge_idx = [i for i, (a, b) in enumerate(zip(before_edge_set, after_edge_set)) if a != b]

    return full_edges[changed_edge_idx[0]]

def get_edge_info(full_edges, current_edge_set, llm_micro_action_set):

    add_f2p_edge = llm_micro_action_set.get_possible_add_fk_pk_edge(current_edge_set)
    add_f2p_edge = [get_changed_edge(current_edge_set, new_edge_set, full_edges) for new_edge_set in add_f2p_edge] if len(add_f2p_edge) > 0 else []
    
    remove_f2p_edge = llm_micro_action_set.get_possible_remove_fk_pk_edge(current_edge_set)
    remove_f2p_edge = [get_changed_edge(current_edge_set, new_edge_set, full_edges) for new_edge_set in remove_f2p_edge] if len(remove_f2p_edge) > 0 else []
    
    convert_row2edge = llm_micro_action_set.get_possible_convert_row_to_edge(current_edge_set)
    convert_row2edge = [get_changed_edge(current_edge_set, new_edge_set, full_edges) for new_edge_set in convert_row2edge] if len(convert_row2edge) > 0 else []
    
    convert_edge2row = llm_micro_action_set.get_possible_convert_edge_to_row(current_edge_set)
    convert_edge2row = [get_changed_edge(current_edge_set, new_edge_set, full_edges) for new_edge_set in convert_edge2row] if len(convert_edge2row) > 0 else []

    add_row2edge_edge = llm_micro_action_set.get_possible_add_row2edge_edge(current_edge_set)
    add_row2edge_edge = [get_changed_edge(current_edge_set, new_edge_set, full_edges) for new_edge_set in add_row2edge_edge] if len(add_row2edge_edge) > 0 else []

    remove_row2edge_edge = llm_micro_action_set.get_possible_remove_row2edge_edge(current_edge_set)
    remove_row2edge_edge = [get_changed_edge(current_edge_set, new_edge_set, full_edges) for new_edge_set in remove_row2edge_edge] if len(remove_row2edge_edge) > 0 else []
    

    add_f2p_edge_info = "Note: ONLY the following set of fk_pk_edge can be added:" if len(add_f2p_edge) > 0 else "Note: There are **NO** fk_pk_edge that can be added in current schema."
    remove_f2p_edge_info = "Note: ONLY the following set of fk_pk_edge can be removed:" if len(remove_f2p_edge) > 0 else "Note: There are **NO** fk_pk_edge that can be removed in current schema."
    convert_row2edge_info = "Note: ONLY the following set of edges can be converted from row to edge:" if len(convert_row2edge) > 0 else "Note: There are **NO** edges that can be converted from row to edge in current schema."
    convert_edge2row_info = "Note: ONLY the following set of edges can be converted from edge to row:" if len(convert_edge2row) > 0 else "Note: There are **NO** edges that can be converted from edge to row in current schema."
    add_row2edge_edge_info = "Note: ONLY the following set of edges can be added:" if len(add_row2edge_edge) > 0 else "Note: There are **NO** row2edge edges that can be added in current schema."
    remove_row2edge_edge_info = "Note: ONLY the following set of edges can be removed:" if len(remove_row2edge_edge) > 0 else "Note: There are **NO** row2edge edges that can be removed in current schema."

    for edge in add_f2p_edge:
        edge_info = {"from_table_name": edge[0], "from_col_name": edge[1].replace('f2p_',''),  "to_table_name": edge[2]}
        add_f2p_edge_info += f'\n{json.dumps(edge_info)}'
    for edge in remove_f2p_edge:
        edge_info = {"from_table_name": edge[0], "from_col_name": edge[1].replace('f2p_',''),  "to_table_name": edge[2]}
        remove_f2p_edge_info += f'\n{json.dumps(edge_info)}'
    for edge in convert_row2edge:
        edge_info = {"table_1_name": edge[0],  "table_2_name": edge[2], "edge_table_name": edge[1].replace('r2e_','')}
        convert_row2edge_info += f'\n{json.dumps(edge_info)}'
    for edge in convert_edge2row:
        edge_info = {"table_1_name": edge[0],  "table_2_name": edge[2], "edge_table_name": edge[1].replace('r2e_','')}
        convert_edge2row_info += f'\n{json.dumps(edge_info)}'
    for edge in add_row2edge_edge:
        edge_info = {"table_1_name": edge[0],  "table_2_name": edge[2], "edge_table_name": edge[1].replace('r2e_','')}
        add_row2edge_edge_info += f'\n{json.dumps(edge_info)}'
    for edge in remove_row2edge_edge:
        edge_info = {"table_1_name": edge[0],  "table_2_name": edge[2], "edge_table_name": edge[1].replace('r2e_','')}
        remove_row2edge_edge_info += f'\n{json.dumps(edge_info)}'
    
    return {"add_fk_pk_edge": add_f2p_edge_info, 
            "remove_fk_pk_edge": remove_f2p_edge_info, 
            "convert_row_to_edge": convert_row2edge_info, 
            "convert_edge_to_row": convert_edge2row_info,
            "add_row2edge_edge": add_row2edge_edge_info,
            "remove_row2edge_edge": remove_row2edge_edge_info}


def conduct_multiple_actions(
    actions: list [dict],
    llm_micro_action_set: LLMMicroActionSet,
    current_edge_set: tuple
):  
    valid_actions = []
    invalid_actions = []
    edge_set = current_edge_set
    graph_idx = -1
    error_msg = ""
    for action in actions : 
        action['parameters'] = action['parameters'][0] if type(action['parameters']) == list else action['parameters']
        new_edge_set, new_graph_idx, new_error_msg = get_micro_action_result(
            action=action,
            llm_micro_action_set=llm_micro_action_set,
            current_edge_set=current_edge_set
        )
        # if the action is not valid, skip it
        if new_edge_set is None:
            continue
        # if the action is invalid, add it to the invalid actions
        if new_graph_idx == -1:
            error_msg += f"Action: {action['action']} \nParameters: {json.dumps(action['parameters'])} \nError: {new_error_msg}\n"
            invalid_actions.append(action)
            continue
        else:
            valid_actions.append(action)
            edge_set = new_edge_set 
            graph_idx = new_graph_idx
            # print(f"New edge set: {edge_set}")
            # print(f"New graph idx: {graph_idx}")
    # print(f"Valid actions: {len(valid_actions)} Invalid actions: {len(invalid_actions)}")
    # print(f"Error msg: {error_msg}")
    return valid_actions, invalid_actions, edge_set, graph_idx, error_msg
    
def update_edge_set(
    current_edge_set, 
    new_edge_set, 
    best_edge_set,
    update_best,
):
    # update best edge set
    current_edge_set = new_edge_set
    # print(f"Edge set changed from {current_edge_set} to {new_edge_set}")
    if update_best:
        best_edge_set = new_edge_set
        print(f"\033[93mBest edge set updated to {best_edge_set}\033[0m")
    return current_edge_set, best_edge_set

def update_score(
    current_score, 
    new_score, 
    best_score,
    score_result,
    update_best,
):  
    # if graph_idx = -1, do not update the score ( i.e., new_score = current_score )
    if new_score == current_score:
        score_result.append(best_score)
        return current_score, current_score, best_score, score_result

    past_score = current_score
    current_score = new_score
    if update_best:
        best_score = current_score
        print(f"\033[93mBest score updated to {best_score:.4f}\033[0m")
    score_result.append(best_score)
    return past_score, current_score, best_score, score_result

def update_action(
    parsed_all_actions, # ALL actions
    valid_actions, # ONLY Valid actions
    action_result, # history actions
    valid_action_result, # history valid actions
    best_valid_action_result,
    last_action_num,
    update_best,
):  
    action_result.extend(parsed_all_actions)
    valid_action_result.extend(valid_actions)
    
    if update_best:
        best_valid_action_result = valid_action_result.copy()
        print(f"\033[93mBest valid action result updated to {len(best_valid_action_result)} actions \033[0m")
    else:
        pass # use the best_valid_action_result

    if len(valid_actions) > 0:
        last_action_num = len(valid_actions)
    else:
        pass # use the last_action_num
    return action_result, valid_action_result, best_valid_action_result, last_action_num


def remove_invalid_history_actions(
    action_result,
    invalid_actions,
):  
    print(f"Total history actions: {len(action_result)}")
    print(f"InValid history actions: {len(invalid_actions)}")
    for action in action_result:
        if action  in invalid_actions:
            action_result.remove(action)

    
    print(f"Total history actions: {len(action_result)}")
    return action_result

def get_history_actions(
    valid_action_result,
    # parsed_all_actions,
    max_history_actions,
):
    # valid_action_result.extend(parsed_all_actions)
    latest_actions = valid_action_result[-max_history_actions:]
    history_actions = json.dumps(latest_actions, indent=2).strip() if len(latest_actions) > 0 else ""
    return history_actions
