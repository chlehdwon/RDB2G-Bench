import torch
import numpy as np

def integrate_edge_tf(batch, edge_tf_dict):
    r2e_edge_types = []
    rev_r2e_edge_types = []

    for edge_type in batch.edge_types:
        src, rel, dst = edge_type
        if rel.startswith('r2e'):
            r2e_edge_types.append(edge_type)
        elif rel.startswith('rev_r2e'):
            rev_r2e_edge_types.append(edge_type)

    for edge_type in edge_tf_dict:
        del batch[edge_type]

    table_to_edge_types = {}
    for edge_type in r2e_edge_types:
        src, rel, dst = edge_type
        table_name = rel[4:]
        if table_name not in table_to_edge_types:
            table_to_edge_types[table_name] = {'r2e': [], 'rev_r2e': []}
        table_to_edge_types[table_name]['r2e'].append(edge_type)
    
    for edge_type in rev_r2e_edge_types:
        src, rel, dst = edge_type
        table_name = rel[8:]
        if table_name not in table_to_edge_types:
            table_to_edge_types[table_name] = {'r2e': [], 'rev_r2e': []}
        table_to_edge_types[table_name]['rev_r2e'].append(edge_type)

    for table_name, edge_types in table_to_edge_types.items():
        if table_name not in edge_tf_dict:
            continue

        all_mapped_ids = set()
        r2e_mapped_ids_dict = {}
        rev_r2e_mapped_ids_dict = {}
        
        for edge_type in edge_types['r2e']:
            if 'mapped_node_ids' in batch[edge_type] and len(batch[edge_type]['mapped_node_ids']) > 0:
                mapped_ids = batch[edge_type]['mapped_node_ids'].cpu().numpy()
                all_mapped_ids.update(mapped_ids)
                r2e_mapped_ids_dict[edge_type] = mapped_ids
        
        for edge_type in edge_types['rev_r2e']:
            if 'mapped_node_ids' in batch[edge_type] and len(batch[edge_type]['mapped_node_ids']) > 0:
                mapped_ids = batch[edge_type]['mapped_node_ids'].cpu().numpy()
                all_mapped_ids.update(mapped_ids)
                rev_r2e_mapped_ids_dict[edge_type] = mapped_ids
        
        if not all_mapped_ids:
            continue

        all_mapped_ids = sorted(list(all_mapped_ids))
        mapped_tensor = torch.tensor(all_mapped_ids, dtype=torch.long)

        batch[table_name]['tf'] = edge_tf_dict[table_name][mapped_tensor]

        max_id = max(all_mapped_ids) + 1
        id_to_idx_array = np.full(max_id, -1)
        for idx, id_val in enumerate(all_mapped_ids):
            id_to_idx_array[id_val] = idx
        
        for edge_type in edge_types['r2e']:
            if edge_type in r2e_mapped_ids_dict:
                old_mapped_ids = r2e_mapped_ids_dict[edge_type]
                new_indices = torch.tensor(id_to_idx_array[old_mapped_ids], dtype=torch.long)
                batch[edge_type]['mapped_node_ids'] = new_indices
        
        for edge_type in edge_types['rev_r2e']:
            if edge_type in rev_r2e_mapped_ids_dict:
                old_mapped_ids = rev_r2e_mapped_ids_dict[edge_type]
                new_indices = torch.tensor(id_to_idx_array[old_mapped_ids], dtype=torch.long)
                batch[edge_type]['mapped_node_ids'] = new_indices

    return batch

def divide_node_edge_dict(batch, x_dict):
    table_to_edge_types = {}
    for edge_type in batch.edge_types:
        src, rel, dst = edge_type
        if rel.startswith('r2e'):
            table_name = rel[4:]
            if table_name not in table_to_edge_types:
                table_to_edge_types[table_name] = {'r2e': [], 'rev_r2e': []}
            table_to_edge_types[table_name]['r2e'].append(edge_type)
        elif rel.startswith('rev_r2e'):
            table_name = rel[8:]
            if table_name not in table_to_edge_types:
                table_to_edge_types[table_name] = {'r2e': [], 'rev_r2e': []}
            table_to_edge_types[table_name]['rev_r2e'].append(edge_type)
    edge_dict = {}
    tables_to_remove = set()
    for table_name, edge_types in table_to_edge_types.items():
        if table_name in x_dict:
            table_features = x_dict[table_name]
            for edge_type in edge_types['r2e'] + edge_types['rev_r2e']:
                assert hasattr(batch[edge_type], 'mapped_node_ids')
                mapped_ids = batch[edge_type]['mapped_node_ids']
                edge_dict[edge_type] = table_features[mapped_ids]
            tables_to_remove.add(table_name)

    for table_name in tables_to_remove:
        if table_name in x_dict:
            del x_dict[table_name]

    return x_dict, edge_dict