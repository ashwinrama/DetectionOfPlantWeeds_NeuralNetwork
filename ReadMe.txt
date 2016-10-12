----------------------------------------------------------------------
Weed Population Detection in Live Grass Using Modified Backpropagation Algorithm
----------------------------------------------------------------------

MATLAB Source codes:

 WeedDetection_MBP.m
----------------------------------------------
Describes the SBP and MBP algorithms.
The 2 algorithms can be switched by making
the lambda (line 17) zero for SBP and non-zero
for MBP

This program trains a (3-8-1) layer network and
outputs a scatter plot of network response and
performance index curves.

The reflectance data are supplied in the folders
for different light and weather conditions.

nnm_train.txt nnm_validate.txt nnm_test.txt files
are required to be in the current working directory


-----------------------------------------------

  WeedDetection_Test.m

This program is used to test the network performance
for different weather patterns. Copy the any reflectance
data file and place it in the CWD with the name nnm_eval.txt

The program outputs the scatter plot showing network 
response.
------------------------------------------------------------

HiddenLayerSelection.xls

This excel file has data to shown why (3-8-1) network configuration
is chosen for the experiment.
---------------------------------------------------------------

Questions: email: asramac@okstate.edu
