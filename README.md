

This repository contains the code for the paper [*Individually Fair Rankings*](https://openreview.net/forum?id=71zCSP_HuBN) by Bower et al. The paper appeared at ICLR 2021.

"fair_training_ranking.py": Contains the code to run SenSTIR. To run SenSTIR, use the "train_fair_nn" function. Look at the comments regarding its input.

Folders:

- "synthetic_data": This folder contains the code to reproduce the synthetic data experiments. Run the "synthetic.ipynb" notebook.

- "Fair-PGRank": This folder contains the code written by Ashudeep Singh and taken from https://github.com/ashudeep/Fair-PGRank that accompanies the "Policy Learning for Fairness in Ranking" paper by Ashudeep Singh and Thorsten Joachims.

- "German": This folder contains the code to reproduce the German credit experiments. 
	* First, run the "german_credit_preprocessing.ipynb notebook to download the data and preprocess it. 
	* To run the baseline, project baseline, and random baseline with the hyperparameters in our paper, run 'python german_baseline.py --seed number_of_seeds --n_units 0 --l2_reg 0' where number_of_seeds is the number of random train/test splits you want to use. 
	* To run the SenSTIR experiments with the hyperparameters in our paper, run 'python german.py --epsilon 1.0 --fair_reg fair_reg_val --seed seed_number --n_units 0 --l2_reg 0' where fair_reg_val (rho in the paper) varies (see the appendix) and seed_number is which of the random train/test splits to use. 

	* If you run all the experiments on 10 train/test splits with all the choices of fair_reg and lamb_val as in our paper, "plots.ipynb" will reproduce the plots in the paper.

- "MSLR": This folder contains the code to reproduce the Microsoft LTR experiments.
	* First, download the data https://www.microsoft.com/en-us/research/project/mslr/ (the MSLR-WEB10K data), and save it to a folder called "original_data".
	* Second, run the 'preprocess_MSLR.ipynb' notebook to preprocess the data.
	* To run the baseline, project baseline, and random baseline with the hyperparameters in our paper, run 'python MSLR_baseline.py --n_units 0 --l2_reg .001'
	* To run the SenSTIR experiments with the hyperparameters in our paper, run 'python MSLR.py --l2_reg .001 --adv_epoch 40 --adv_step .01 --adv_epoch_full 40 --adv_step_full .001 --epsilon .01 --fair_reg fair_reg_val --n_units 0' where fair_reg_val (rho in the paper) varies (see the appendix).
	* To run the fair-PG-rank experiments with the hyperparameters in our paper, run 'python PG_MSLR.py --lamb lamb_val --l2_reg .01 --n_units 0' where lamb_val (lambda in the fair-pg-rank paper refers to the fair regularization strength) varies (see the appendix).