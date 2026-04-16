## Project Overview

This repository contains the implementation for evaluating top quark tagging models within a constrained "small-data" regime (323,200 training jets). The study reproduces the architectural comparison between Deep Sets and Graph Neural Networks (GNNs) to evaluate their robustness under limited hardware resources. 

Quantitative results demonstrate a significant 7.7% performance gap: the GNN achieved 90.8% accuracy (AUC 0.966), while the Deep Sets model plateaued at 83.1% (AUC 0.901). This discrepancy, which is much wider than differences reported in large-scale analyses (DOI: 10.5281/zenodo.2603256) , highlights the superior inductive bias of GNNs for jet physics in limited-data environments.

## Physics Context

The primary objective of this project is to extract the top quark signal from Quantum Chromodynamics (QCD) background noise while comparing different AI architectures.

* **Signal**: Corresponds to hadronically decaying top quarks characterized by large Lorentz boosts and a distinctive three-prong substructure.


* **Background**: Consists of light-quark or gluon jets produced via QCD processes, exhibiting a single-prong or diffuse radiation pattern.


To simulate detector reconstruction, the models operate on low-level kinematic features: transverse momentum ($p_T$), pseudorapidity ($\eta$), and azimuthal angle ($\phi$).

## Dataset Overview

The data pipeline utilizes the Top Quark Tagging Reference Dataset (DOI: 10.5281/zenodo.2603256).
- Simulation Framework: Monte Carlo simulated proton-proton collision events at a center-of-mass energy of $\sqrt{s}$ = 14 TeV, generated using Pythia8 and Delphes for ATLAS detector simulation.
- Jet Clustering: Jets were clustered using the anti-kT​ algorithm with a radius parameter of R=0.8 using FastJet.
- Scale & Splits: The complete official dataset contains approximately 2 million events (1.2M training, 400k validation, and 400k testing). To accommodate hardware limits, this study operates on a restricted subset of 404,000 jets from the test.h5 file, utilizing the leading 30 constituent particles per jet rather than the full 200.


## Code Progression

The repository is structured as a progressive scientific experiment, evolving from synthetic baselines to real-world physics data:

* **`0_random_jets.ipynb`**: Synthesizes and visualizes basic kinematic features of jets. Establishes the geometric baseline and formatting requirements for signal (tight) versus background (wide) signatures.
* **`1_simple_condition_deep_sets.ipynb`**: Implements a baseline Deep Sets architecture utilizing *early fusion*. Global conditioning variables are connected with the individual particle level before processing, allowing the network to learn early context at the cost of memory.
* **`2_medium_condition_deep_sets.ipynb`**: Advances the Deep Sets model by applying *late fusion* (jet-level conditioning). This optimizes memory efficiency by appending conditions only to the globally aggregated "Jet Summary" representation.
* **`3_medium_condition_GNN.ipynb`**: Introduces Graph Neural Networks. Replaces isolated particle processing with local message passing to capture geometric relationships. Utilizes Dropout regularization to prevent model grokking and force the learning of generalized geometric rules over noise memorization.
* **`4_top_quark_tagging_MC_GNN.ipynb`**: Transitions to the official Monte Carlo Top Tagging Reference Dataset. Implements a ParticleNet-Lite dynamic GNN architecture utilizing EdgeConv operations to successfully recover the 3-prong topology of top quark decays.
* **`5_MC_deep_set.ipynb`**: Acts as the scientific control experiment. Mathematically blinds the AI to local geometry.


## Methodology

The study contrasts two primary paradigms:

* **Particle Flow Networks (Deep Sets)**: Treats jets as permutation-invariant unordered sets. It processes constituents independently through a Multilayer Perceptron (MLP) before executing a global summation to ensure the output is independent of particle ordering.

* **ParticleNet (GNN)**: Operates on jets as point clouds. It constructs dynamic $k$-nearest neighbor graphs to capture local spatial correlations via EdgeConv operations, dynamically updating the graph at each layer based on learned latent features.

## Prerequisites

The workflow relies on Python-based open-source libraries for data processing and deep learning inference:

* **Data Handling**: Pandas, NumPy, and Awkward Array (for variable-length particle collections).
* **Deep Learning**: PyTorch and PyTorch Lightning.
* **Evaluation**: Scikit-Learn (ROC/AUC calculations) and Matplotlib (Confusion matrices).

## Usage

The codebase is optimized for execution on GPU hardware (e.g., NVIDIA T4 via Google Colab). Ensure the HDF5 datasets are downloaded and correctly linked to your directory before running notebooks 4 and 5. Execute the notebooks sequentially to replicate the data generation, baseline testing, and final evaluations.

## Contact

If you have any questions or suggestions regarding the implementation or the numerical results of this study, feel free to reach out to me.

- Author: Mehmet Ehliz
- LinkedIn: https://www.linkedin.com/in/ehliz/
- Email: imi.ehliz@gmail.com

