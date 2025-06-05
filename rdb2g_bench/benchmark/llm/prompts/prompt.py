import json

def get_score_feedback(initial_score, past_score, current_score, higher_is_better, last_action_num):
    score_feedback = f"In history actions, after the last {last_action_num} actions, the score has changed from {past_score:.4f} to {current_score:.4f}.\n"
    score_feedback += (
        "Since a **higher** score is better, " if higher_is_better else "Since a **lower** score is better, "
    )
    if (current_score > past_score and higher_is_better) or (current_score < past_score and not higher_is_better):
        score_feedback += f"the performance has **improved**."
        # if score has improved but still lower than initial score
        if (initial_score == current_score) or (initial_score > current_score and higher_is_better) or (initial_score < current_score and not higher_is_better):
            score_feedback += f"\nHowever, the initial score was {initial_score:.4f}, Please try other actions to improve the performance."
    else:
        score_feedback += "the performance has **decreased**. Please consider either reversing the previous action or exploring alternative actions to improve the schema."
    return score_feedback


def get_action_info(actions= [ 'add_fk_pk_edge','remove_fk_pk_edge',  'convert_edge_to_row',  'convert_row_to_edge' ], #'add_row2edge_edge', 'remove_row2edge_edge', 
                    edge_info={"add_fk_pk_edge": "", "remove_fk_pk_edge": "", "convert_row_to_edge": "", "convert_edge_to_row": ""}): #, 'add_row2edge_edge', 'remove_row2edge_edge'
    with open("./prompts/action.json", "r") as f:
        action_info = json.load(f)
    action = ""
    for i, action_name in enumerate(actions):
        if edge_info[action_name] != "":
            action += f"{action_info[action_name]}\n{edge_info[action_name]}\n\n"
    return action

def augmentation_prompt(dataset_name, task_name,  edge_info, history_actions="", error_msg="", past_score=0, current_score=0, higher_is_better=True, initial_score=0, budget=10, initial_attempt=True, last_action_num=1):
    with open("./prompts/data_stats.json", "r") as f:
        stats = json.load(f)
        stats = stats[dataset_name]
    with open("./prompts/schema.json", "r") as f:
        schema = json.load(f)
        schema = schema[dataset_name]
    with open("./prompts/task.json", "r") as f:
        task = json.load(f)
        task = task[task_name]

    action_info = get_action_info(edge_info=edge_info)
    score_feedback = ""


    if error_msg != "":
        error_msg = f"Warning: The following actions will cause errors: \n{error_msg}"


    if initial_attempt:
        return f"""You are expected to construct graph schema based on the original inputs.
You will be given an original schema represented in the dictionary format:
<data>
1. dataset_name: name of the dataset
2. tables: meta data for list of tables, each one will present following attributes
1. name: table name
2. columns: list of columns, each column will have following attributes
    1. name: column name
    2. dtype: column type, can be either text, categorical, float, primary_key, foreign_key, or multi_category. primary_key and foreign_key are two special types of categorical columns, which presents a structural relationship with other tables. Multi_category means this column is of list type, and each cell main contains a list of categorical values. After a column is set as primary_key or foreign_key, it should not be changed to other types.
    3. link_to (optional): if this column is a foreign key, point to which primary key from which table
3. statistics of the table: statistics of the column value of tables. These statistics can be used to help you determine the characteristics of the columns. 
</data>

Here are the documents of the actions:
{action_info}
{error_msg}

Now, you need to:

1. Actively think about which actions (from the list below) should be conducted to improve the schema.
2. Output all actions you can think of from the above list to make the schema better, and output your selections in the following format:
If multiple actions are needed, please list **all** of them.

<selection>
[
  {{
    "explanation": <explanation for the selection>,
    "action": <selected action>,
    "parameters": <parameters for the action>
  }},
  {{
    "explanation": <explanation for the selection>,
    "action": <selected action>,
    "parameters": <parameters for the action>
  }},
  {{
    "explanation": <explanation for the selection>,
    "action": <selected action>,
    "parameters": <parameters for the action>
  }},
  ...
]
</selection>

<input>
<dataset_stats>
{stats}
</dataset_stats>
<task>
{task} 
</task>
<schema>
{schema}
</schema>
</input>


Return your output in the json format inside <selection></selection>.""" 
    
    else:
        # Add history actions and score feedback only if there are history actions
        if history_actions.strip() != "":
            history_actions = f"History Actions: \n{history_actions}"
            score_feedback = get_score_feedback(initial_score, past_score, current_score, higher_is_better, error_msg, last_action_num)
        return f"""You are expected to construct graph schema based on the original inputs.
You will be given an original schema represented in the dictionary format:
<data>
1. dataset_name: name of the dataset
2. tables: meta data for list of tables, each one will present following attributes
   1. name: table name
   2. columns: list of columns, each column will have following attributes
      1. name: column name
      2. dtype: column type, can be either text, categorical, float, primary_key, foreign_key, or multi_category. primary_key and foreign_key are two special types of categorical columns, which presents a structural relationship with other tables. Multi_category means this column is of list type, and each cell main contains a list of categorical values. After a column is set as primary_key or foreign_key, it should not be changed to other types.
      3. link_to (optional): if this column is a foreign key, point to which primary key from which table
3. statistics of the table: statistics of the column value of tables. These statistics can be used to help you determine the characteristics of the columns. 
</data>

Here are the documents of the actions:
{action_info}
{error_msg}

Now, you need to 
1. Actively think about whether any one of the 4 actions should be conducted
2. Output all actions you can think of from the above list to make the schema better, and output your selections in the following format:
If multiple actions are needed, please list **all** of them.

<selection>
[
  {{
    "explanation": <explanation for the selection>,
    "action": <selected action>,
    "parameters": <parameters for the action>
  }},
  {{
    "explanation": <explanation for the selection>,
    "action": <selected action>,
    "parameters": <parameters for the action>
  }},
  {{
    "explanation": <explanation for the selection>,
    "action": <selected action>,
    "parameters": <parameters for the action>
  }},
  ...
]
</selection>

3. If you think there's no more action, you can output 
<selection>
None
</selection>

{history_actions}

<input>
<dataset_stats>
{stats}
</dataset_stats>
<task>
{task} 
</task>
<schema>
{schema}
</schema>
</input>

{score_feedback}

Note that the current schema may **not be optimal**, so other actions may yield better results.
Please **only** halt the program with `None` if you believe no further actions are worth trying. 
You can try {budget} more times to improve the performance.
Return your output in the json format inside <selection></selection>.""" 

