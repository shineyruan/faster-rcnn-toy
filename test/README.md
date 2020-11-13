# HW4 Test Cases
All the files can be loaded using torch.load()

## Test Scripts
All the Python test scripts are located in `src/BoxHead.py`. When running test scripts, please go to the root folder and specify:
```bash
python src/BoxHead.py
```

## Ground Truth
Test Case for the create_ground_truth.
Each test contains a dictionary with the following entries:
* proposals: input of the create_ground_truth that corresponds to the proposals produced by the RPN
* gt_labels: input of the create_ground_truth that corresponds to the ground truth labels of the ground truth bounding boxes
* bbox: input of the create_ground_truth that correspond to the ground truth bounding boxes
* labels: output of the create_ground_truth. the ground truth labels assigned to each proposal
* regressor_target: output. the target of the regressor for each proposal. It only makes sense to compare your regressor output
with the given one for the proposals that are not assigned to the background ( for proposals assigned to background the regressor target in the 
given test case is an arbitary vector)

## MultiScaleRoiAlign
Test Cases for the MutliScaleRoiAlign
Each test contains a dictionary with the following entries:
* fpn_feat_list: input of the MultiScaleRoiAlign which corresponds to the list of features produced by the FPN for a batch of
images
* proposals: input of the MultiScaleRoiAlign which correspond to the top L proposals produced by the RPN for a batch of images
* output_feature_vectors: output of MultiScaleRoiAlign containing the flatten pooled fetures for each proposal

 
## Loss
Test Cases for the compute loss function.
Each test contains a dictionary with the following entries
* clas_logits: the input of the compute_loss which corresponds to the predictions of the classifier
* box_preds: the input of the compute_loss which corresponds to the prediction of the regressor
* labels: the input of the compute_loss which corresponds to the ground truth label of each proposal
* regression_targets: the input of the compute_loss which corresponds to the ground target of the regressor
* effective_batch: the effective batch that we use
* random_permutation_foreground: the indexes of the sampled proposals that we used from the set of no-background proposals (foreground proposals).
This allows us to check the function without having the randomness of the sampling process. For examples to take the regression
targets using this we can do :\
foreground_ind=(labels>0).nonzero()\
foreground_target=regression_targets[foreground_ind[random_permutation_foreground]].squeeze()
* random_permutation_background: similar with the random_permutation_foreground but now we have the indexes of the sampled proposals from the set of background proposals
* loss_clas: the loss of the classifier regularized by the number of sampled background and foreground proposals (which is almost always the effective batch size)
* loss_reg: the loss of the regressor regularized by the number of sampled background and foreground proposals
