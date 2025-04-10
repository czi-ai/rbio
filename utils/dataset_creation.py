from perturbqa import load_de, load_dir, auc_per_gene
from datasets import Dataset, load_dataset

def generate_prompt_column(row, strict_binary=False, prompt_type='de_expression', cell_type='K562'):
    """
    Generates a prompt columns
    
    Args:
        dataset_name: pertq_dataset name
    Returns:
        train_dataset: training_dataset
        test_dataset: test_dataset
        X_train_keys: training keys for pertqa_dataset
        X_test_keys: testing keys for pertqa_dataset
    """
    print(f"I am using a binary answer: {strict_binary} for the task {prompt_type}")
    gene_A = row['pert']
    gene_B = row['gene']
    
    if prompt_type == 'gene_de_expression':
        row["prompt"] = f'Is a knockdown of {gene_A} in {cell_type} cells likely to result in differential expression of {gene_B}?'
    elif prompt_type == 'gene_dir_change':
        row["prompt"] = f'Is a knockdown of {gene_A} in {cell_type} cells likely to result in an increase of {gene_B}?'
    if strict_binary:
        row['prompt'] += 'Give only a "Yes" or "No" answer.'
    # else:
        # row['prompt'] += 'At the end of your answer, give a Yes or No. Please provide your entire reasoning trace'
    return row


def generate_dataset_from_pertqa(dataset_name, logger, num_rows = -1):
    """
    Generates a HuggingFace dataset from a pertqa dataset. Returns the top num_rows from the dataset
    
    Args:
        dataset_name: pertq_dataset name
    Returns:
        train_dataset: training_dataset
        test_dataset: test_dataset
        X_train_keys: training keys for pertqa_dataset
        X_test_keys: testing keys for pertqa_dataset
    """
    data_de = load_de(dataset_name)
    X_train = data_de["train"]
    X_test = data_de["test"]
    logger.info(f"Total number of observations: {len(X_train)}")
    
    X_train_keys = [(x["pert"], x["gene"]) for x in X_train][:num_rows]
    X_test_keys = [(x["pert"], x["gene"]) for x in X_test][:num_rows]
        
    data_dir = load_dir("k562")
    train_dataset = Dataset.from_list([{"pert": x['pert'], "gene": x['gene'], 'label' : x['label']} for x in X_train])
    test_dataset = Dataset.from_list([{"pert": x['pert'], "gene": x['gene'], 'label' : x['label']} for x in X_test])
    if num_rows != -1:
        train_dataset = train_dataset.select(range(num_rows))
        test_dataset = test_dataset.select(range(num_rows))
    return train_dataset, test_dataset, X_train_keys, X_test_keys


def generate_dataset_from_norman_query(task):
    """
    Generates a HuggingFace dataset from a query on the Norman datset
    
    Args:
        task: type of task within the Norman dataset
    Returns:
        test_dataset: test dataset
    """
    prompts = []
    perts = []
    if task == 'cell_cycle_position':
        pert_pairs = [(None, 'CDKN1A'), (None, 'CDKN1B'), ('CDKN1C', None), 
                  ('CDKN1C', 'CDKN1A'), ('CDKN1C', 'CDKN1B'), 
                  ('PLK4', None), (None, "STIL"), ("PLK4", "STIL"), 
                  ('CKS1B', None), ('KIF18B', None), (None, 'KIF2C'), ('KIF18B', 'KIF2C')]
        cell_cycle_positions = ['M', 'M-G1', 'G1-S', 'S', 'G2-M']
        for (gene_A, gene_B) in pert_pairs:
            if gene_A and gene_B:
                # Would you expect it to be arrested at a particular stage? 
                # If the answer is yes, then at what stage? make is multiple choice
                prompt = f"How would an overexpression of {gene_A} and {gene_B} affect the cell cycle? Would you expect the cell to become arrested at a particular stage? If the answer is yes, then at what stage?"
            elif gene_A:
                prompt = f"How would an overexpression of {gene_A} affect the cell cycle? Would you expect the cell to become arrested at a particular stage? If the answer is yes, then at what stage?"
            elif gene_B:
                prompt = f"How would an overexpression of {gene_B} affect the cell cycle? Would you expect the cell to become arrested at a particular stage? If the answer is yes, then at what stage?"
            prompt +=f'Choose one of the following cycles: {str(cell_cycle_positions)}?'
            prompts.append({"prompt" : prompt})
        print(prompts)
        test_dataset = Dataset.from_list(prompts)
                                         
    return test_dataset