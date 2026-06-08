# AngPPIS-DisPPIS-SecPPIS
README
**Overview**
This study focuses on residue-level protein-protein interaction site (PPIS) prediction, which is distinct from protein-level protein-protein interaction (PPI) prediction. Protein-level PPI prediction determines whether two proteins interact, whereas PPIS prediction aims to identify the specific amino acid residues that participate in protein-binding interfaces.
Three complementary structural-view models are provided:

* AngPPIS: uses residue-level angular matrix representations derived from protein three-dimensional structures.
* DisPPIS: uses atom-level distance-vector representations and statistical distance descriptors. Atom-level predictions are mapped back to residue-level probabilities for residue-level interpretation.
* SecPPIS: uses local amino-acid, secondary-structure, physicochemical, positional, and run-length features.

The models were evaluated using protein-level training, validation, and independent test subsets to reduce information leakage. Model selection and threshold determination were performed on the validation set, and final performance was assessed on independent test proteins across five random seeds.
Data Source and Processing

The initial protein-chain list and residue-level PPIS annotations were derived from benchmark data used in EnsemPPIS and related DeepPPISP-based studies. The corresponding protein structures were obtained from the RCSB PDB database, and secondary-structure information was generated using DSSP.

The original benchmark split was not directly reused. Instead, the benchmark annotations were used as the initial source, and the corresponding PDB structures were retrieved, processed, filtered, and aligned with residue-level labels. Chains with unresolved sequence-label inconsistencies or incompatible structural files were excluded. The final retained protein chains were repartitioned at the protein level into training, validation, and independent test subsets.

Because the three models require different input representations, model-specific feature files were generated:

* AngPPIS: residue-level angular matrices.
* DisPPIS: atom-level distance vectors and statistical descriptors.
* SecPPIS: residue-level sequence-structure feature vectors.

Large intermediate files, such as full .npy angular matrices, distance matrices, and trained model checkpoints, may be large and are therefore not all included in this repository. These files are available from the corresponding author upon reasonable request.

Protein-Level Split Files

To avoid data leakage, all datasets were partitioned at the protein level. All residues or atoms derived from the same protein chain were assigned exclusively to one subset.

For each model and random seed, the split metadata files record:

* protein or chain identifier;
* train, validation, or independent test assignment;
* number of samples;
* number of positive and negative labels;
* random seed.

These files are provided to document the exact protein-level partitions used in the manuscript.

Model Descriptions

AngPPIS

AngPPIS represents each residue using an angular matrix derived from residue-level spatial relationships. Each residue is represented by the average coordinate of its non-hydrogen atoms. Angular matrices are normalized and processed by a compact two-dimensional convolutional neural network together with statistical descriptors.

DisPPIS

DisPPIS represents each atom using a distance vector relative to all other atoms in the same protein chain. Distance values are clipped and normalized. Statistical descriptors of each distance profile are used as auxiliary features. The model predicts atom-level probabilities, which are then mapped to residue-level probabilities by taking the maximum predicted probability among atoms belonging to the same residue.

SecPPIS

SecPPIS uses engineered residue-level features based on amino acid identity, secondary-structure state, local amino acid composition, local secondary-structure composition, physicochemical residue-group composition, relative sequence position, and secondary-structure run-length information. These features are processed by a multilayer perceptron classifier.

Training and Evaluation

All models were trained using PyTorch. The evaluation procedure follows the manuscript:

1. Construct model-specific input features.
2. Generate protein-level training, validation, and independent test subsets.
3. Train each model only on the training set.
4. Select the best checkpoint using validation AUPRC.
5. Select the decision threshold on the validation set by maximizing MCC.
6. Apply the fixed model and threshold to the independent test set.
7. Repeat the process across five random seeds.
8. Report performance as mean ± standard deviation.

The main evaluation metrics include:

* AUROC;
* AUPRC;
* MCC;
* F1-score;
* balanced accuracy;
* precision;
* recall;
* accuracy.

Because PPIS prediction is an imbalanced binary classification task, AUPRC, MCC, F1-score, and balanced accuracy should be interpreted together with AUROC and accuracy.

Figure Reproduction

The figures/ or results/ folders contain scripts and source tables used to reproduce the main figures in the manuscript, including:

* independent-test performance summaries;
* validation AUPRC and threshold stability;
* confusion matrices;
* representative residue-level prediction maps.

The figure scripts read the processed result tables and generate the final plots used in the manuscript.

**How to Run**
# 1. Prepare or download PDB files
# 2. Generate model-specific features
# 3. Generate protein-level train/validation/test splits
# 4. Train the selected model
# 5. Evaluate the model on the independent test set
# 6. Generate summary tables and figures

**The Python versions used in this article are Python 3.11.5 and PyTorch2.1.2**

**Availability of Large Files**
Due to file-size limitations, large intermediate files are not all included in this repository. These may include:
* full PDB-derived distance matrices;
* angular matrix .npy files;
* trained .pth model checkpoints;
* large intermediate feature files.
These files are available from the corresponding author upon reasonable request.

**Citation**
If you use this repository, please cite the associated manuscript once it is available.

**Contact**
For any questions or further assistance, please contact Wenyan Wu at weny60006@gmail.com.
