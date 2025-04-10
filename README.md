# rbio

Repository for reasoning-related tasks. 

Usage:
```
python train_grpo.py --task=task_name --model-type=model_type --dataset=dataset --reward-type=reward_type
```

## LLMs
If **reward_type**==none, then the model runs model_type with no RL

For the **pertqa_K562** dataset *task_name* is one of **gene_de_expression** or **gene_dir_change**:
- **gene_de_expression**: generative differential expression, binary output: *Is a knockdown of {gene_A} in {cell_type} cells likely to result in an increase of {gene_B}?*
 ```
python train_grpo.py --dataset=pertqa_K562 --task=gene_de_expression
```
- **gene_dir_change**: generative direction of change, binary output: *Is a knockdown of {gene_A} in {cell_type} cells likely to result in differential expression of {gene_B}?*
```
python train_grpo.py --dataset=pertqa_K562 --task=gene_dir_change
```

If true, the flag **--strict-binary=True** will constrain the model to only give a binary Yes/No answer. If false, it will generate answers unconstrained.
```
python train_grpo.py --dataset=pertqa_K562 --task=gene_de_expression  --strict-binary=True
```
```
python train_grpo.py --dataset=pertqa_K562 --task=gene_dir_change  --strict-binary=True
```
For the *norman* dataset *task_name* is one of **cell_cycle_position**:
```
python train_grpo.py --dataset=norman --task=cell_cycle_position
```
This will run a script answering the following query: *How would an overexpression of CDKN1A affect the cell cycle? Would you expect the cell to become arrested at a particular stage? If the answer is yes, then at what stage?Choose one of the following cycles: ['M', 'M-G1', 'G1-S', 'S', 'G2-M']*

## LLMs + RL
If **reward_type**!=none, then the model runs model_type with RL and --reward-type

Right now, a simple reward of r_len is supported
```
wandb disabled
python train_grpo.py --task=gene_de_expression --sanity-check --strict-binary=True --reward-type=r_len
```

## Sanity Checking
If the flag --sanity-check=True, the model will only use a small amount of the training data
```
python train_grpo.py --task=gene_dir_change --sanity-check=True --strict-binary=True
```
For distributed training one can launch:

```
accelerate launch train_grpo.py --task=gene_dir_change --sanity-check=True --strict-binary=True
```
