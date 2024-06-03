# AngPPIS-DisPPIS-SecPPIS
README
**Overview**
![Fig](./abstract.jpg)
This repository contains the code for three innovative deep learning models: SecPPIS, DisPPIS, and AngPPIS. These models are designed to predict features related to proteins' secondary structures, spatial distances, and spatial angles, respectively. Each model's code is stored in its respective named folder.

**Data Loading**
The training and testing datasets are loaded using the DataLoader class. The training data loader (train_loader) is configured with a batch size (batch_size) that has been fine-tuned for each model, and the data is shuffled to ensure randomness during the training process. The testing data loader (test_loader) also uses the same batch_size but does not shuffle the data to maintain consistency and reproducibility during testing.

**File Structure**
All generated matrices are saved in the NumPy .npy format in the specified output directory. Each file is named using the original PDB file name followed by the amino acid index, ensuring easy tracking and analysis of each matrix with its corresponding structure.

**Scripts Execution**
To run the entire process, execute the following scripts in order:

Conversion Script: Converts raw data to the required format.
Cropping Script: Processes and crops the data.
Balancing Script: Balances the data to ensure model robustness.
Training Script: Trains the models using the processed data.
For prediction, run the prediction script at the end.

**Model Weights and Prediction**
The trained model weights are stored in .pth files. To make predictions:

Load the model weights.
Set the model to evaluation mode.
Read the feature data stored in .npy files.
Convert the data into tensor format and expand the dimensions to fit the model's input requirements.
Both training results and prediction results can be found in the output folder.

**Result Readability Conversion**
For readability, after extracting amino acid information, the script matches this information with the data in the original atom prediction file. This matching process involves:

Extracting the numbers representing specific amino acids from the original atom prediction file.
Finding the corresponding amino acid information for these numbers.
Adding this information back to the atom prediction file.
Each record will be appended with the corresponding amino acid sequence number and name.

**How to Run**
Navigate to the respective model folder (AngPPIS, DisPPIS, or SecPPIS).
Run the conversion, cropping, balancing, and training scripts in order.
Please copy the pdb folder in the main directory to the path of the three models, and an additional original pdb file is required in SecPPIS. Due to file size issues, if necessary, please contact us to obtain all training files or run scripts to create training files yourself
For prediction, run the prediction script after training.
Check the output folder for results.

**The Python versions used in this article are Python 3.11.5 and PyTorch2.1.2**

**Contact**
For any questions or further assistance, please contact Wenyan Wu at weny60006@gmail.com.
