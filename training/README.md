# Training
The training code contained in this repository is an example of how to post-train an LLM with the methodology introduced by Rbio. We also provide a Colab notebook with a similar example. Visit [Virtual Cell Platform](https://virtualcellmodels.cziscience.com).

## Required files
In order to run this code you will need to download the following files from [Google Drive](https://drive.google.com/drive/folders/1RSrnjTxikjrvZfKfOfUHti5Jwih-xLdw?usp=drive_link).
You will find
- mlp_model.pt
- k562-train-v0.3.0.csv
- esm_embedding_dictionary_filled.pkl
download all three files and place them within this directory. You will be able to execute training.py

## Notes
- It is strongly recommended to run this code with an A100 GPU. Alternatively it is possible to downgrade the LLM to Qwen2.5-0.5B-Instruct or other smaller models.
- This code is a different implementation released for illustration purposes only compared to the code that has been used to train the methods discussed in our paper. 
- If you are interested only in using the perturbation data we employ in this dataset, please refer to the original repository https://github.com/genentech/PerturbQA and cite the work from our colleagues at Genentech accordingly.