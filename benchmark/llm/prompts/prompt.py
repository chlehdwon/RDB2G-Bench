import json

def type_inference_prompt(db_info_prompt):
    prompt = f"""Now you will be given a list of tables and columns, each one with the following format:
Analysis for Table <name of the table>:
\tColumn: <name of the column 1>
\t\tMax: <max value of the column>
\t\tMin: <min value of the column>
\t\tMode: <mode value of the column>
\t\tSampled Values: <list of sampled values>, for example, ['value1', 'value2', 'value3']
\tColumn: <name of the column 2>
\t\tMax: <max value of the column>
\t\tMin: <min value of the column>
\t\tMode: <mode value of the column>
\t\tSampled Values: <list of sampled values>, for example, ['value1', 'value2', 'value3']
...
{db_info_prompt}

You should identify the data type of each column. The data types you can choose from are:
['float', 'category', 'datetime', 'text', 'multi_category']
float: The column is probably a float-type embedding tensor. There should be (nearly) no redundant values.
category: The column is probably a categorical column.
datetime: The column is probably a datetime column. Only full datetime values should be considered, some columns presenting only year or month or day should be better considerd as category.
text: The column is probably a text column. There should be a lot of unique values. Otherwise it will probably be a category column. Moreover, we should expect texts with natural semantics, otherwise it's probably a category column.
multi_category: The column is probably a multi-category column. Usually this means the column value is a list. 
It should be noted that if the column is probably an embedding type, then directly put it to the float type.
Then, you should output a discription of the column, for example:
"This column is probably representing the ID from 1 to n of users in the system, as it has a lot of unique values."
Output the results with the following format:
{{
    "<name of the table>": {{
        "<name of the column 1>": ("<data type of the column 1>", "<description of the column 1>"),
        "<name of the column 2>": ("<data type of the column 2>", "<description of the column 2>")
    }},
    ...
}}
In description, if you see two columns are very similar and may represent the same thing, you should mention it."""
    return prompt


def reflect_prompt():
    reflect = "Please double check to eliminate errors"
    return reflect


def get_action():
    prompt_all = """Here is the introduction of the function add_fk_pk_edge:
Description:
Add a foreign key (FK) constraint from a column in one table to the primary key (PK) of another table.
This establishes a directed relationship between two tables, allowing the database to enforce referential integrity.
Parameters:
    dbb: the database object
    from_table: the name of the table containing the foreign key
    from_col: the name of the column in from_table to become the foreign key
    to_table: the name of the table containing the primary key
    to_col: the name of the primary key column in to_table


Here is the introduction of the function remove_fk_pk_edge:
Description:
Remove an existing foreign key (FK) constraint from a column in a table.
This breaks the referential integrity between the two tables, allowing the foreign key column to contain values not present in the referenced primary key column.
Parameters:
    dbb: the database object
    from_table: the name of the table containing the foreign key
    from_col: the name of the column in from_table that is the foreign key

    
Here is the introduction of the function convert_row_to_edge:
Description:
Convert a table that represents entities (rows) into a table that represents relationships (edges) between entities.
This is typically done by removing the primary key and ensuring the table contains only foreign keys referencing other tables, thus modeling a relationship.
Parameters:
    dbb: the database object
    table_name: the name of the table to convert
    source_col: the name of the column to become a source foreign key
    target_col: the name of the column to become a target foreign key


Here is the introduction of the function convert_edge_to_row:
Description:
Convert a table that represents relationships (edges) into a table that represents entities (rows).
This is typically done by adding a new primary key column and possibly removing foreign key constraints, so the table is treated as a set of entities.
Parameters:
    dbb: the database object
    table_name: the name of the edge table to convert
    new_pk_col: the name of the new primary key column to add


Here is the introduction of the function add_row2edge_edge:
Description:
Create a new edge (relationship) between two row tables by introducing a new edge table.
The new edge table will contain foreign keys referencing the primary keys of the two row tables, representing a many-to-many relationship.
Parameters:
    dbb: the database object
    table1: the name of the first row table
    table2: the name of the second row table
    edge_table: the name of the new edge table to be created
    table1_fk: the name of the foreign key column referencing table1
    table2_fk: the name of the foreign key column referencing table2

Here is the introduction of the function remove_row2edge_edge:
Description:
Remove an edge (relationship) between two row tables by deleting the edge table or removing its foreign key constraints.
This operation eliminates the explicit relationship between the two tables.
Parameters:
    dbb: the database object
    edge_table: the name of the edge table to remove"""
    
    prompt = """Here is the introduction of the function remove_fk_pk_edge:
Description:
Remove an existing foreign key (FK) constraint from a column in a table.
This breaks the referential integrity between the two tables, allowing the foreign key column to contain values not present in the referenced primary key column.
Parameters:
    dbb: the database object
    from_table: the name of the table containing the foreign key
    from_col: the name of the column in from_table that is the foreign key

    
Here is the introduction of the function convert_row_to_edge:
Description:
Convert a table that represents entities (rows) into a table that represents relationships (edges) between entities.
This is typically done by removing the primary key and ensuring the table contains only foreign keys referencing other tables, thus modeling a relationship.
Parameters:
    dbb: the database object
    table_name: the name of the table to convert
    source_col: the name of the column to become a source foreign key
    target_col: the name of the column to become a target foreign key"""
    
    return prompt

def type_infer_prompt(dataset_name):
    with open("./prompt/type_infer_prompt.json", "r") as f:
        type_infer_prompt = json.load(f)
        type_infer_prompt = type_infer_prompt[dataset_name]
    return type_infer_prompt

def augmentation_prompt(dataset_name, task_name):
    with open("./prompt/data_stats.json", "r") as f:
        stats = json.load(f)
        stats = stats[dataset_name]
    with open("./prompt/schema.json", "r") as f:
        schema = json.load(f)
        schema = schema[dataset_name]
    with open("./prompt/task.json", "r") as f:
        task = json.load(f)
        task = task[task_name]

    actions = get_action()

    prompt = f"""
I'll remove the index numbers from the text and format it cleanly for you.

Imagine you are an expert graph data scientist, and now you are expected to construct graph schema based on the original inputs. You will be given an original schema represented in the dictionary format:

<data>
1. dataset_name: name of the dataset
2. tables: meta data for list of tables, each one will present following attributes
   1. name: table name
   2. source: source of the data, can either be a numpy .npz file or a parquet file
   3. columns: list of columns, each column will have following attributes
      1. name: column name
      2. dtype: column type, can be either text, categorical, float, primary_key, foreign_key, or multi_category. primary_key and foreign_key are two special types of categorical columns, which presents a structural relationship with other tables. Multi_category means this column is of list type, and each cell main contains a list of categorical values. After a column is set as primary_key or foreign_key, it should not be changed to other types.
      3. link_to (optional): if this column is a foreign key, point to which primary key from which table
3. statistics of the table: statistics of the column value of tables. These statistics can be used to help you determine the characteristics of the columns. For example, if one categorical column only contains one unique value, then creating a node type based on this column can result in a super node, which is not ideal for graph construction. You should also determine whether two columns represent the same thing based on these statistics.
</data>

Here are the documents of the actions:
{actions}

Now, you need to
1. Actively think about whether any one of the three actions should be conducted; If not, you can select "None" and then halt the program.
2. output all actions you can think of from the above list to perform, and output your selection in the following format. It should be noted that for those actions with sequential relation, you don't need to generate in one round.

<selection>
[{{'explanation': <explanation for the selection>, 'action': <first action>, 'parameters': <parameters for the first action> }},
 {{'explanation': <explanation for the selection>, 'action': <second action>, 'parameters': <parameters for the second action> }}, ...
]
</selection>

3. If not more action, output <selection>None</selection>

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
    return prompt




txt = (augmentation_prompt("rel-avito", "user-repeat"))
with open("prompt.txt", "w") as f:
    f.write(txt)