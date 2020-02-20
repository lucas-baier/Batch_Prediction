# Batch_Prediction
This is the source code for my master's project on predicting faulty batches.

The objective of the work is to predict faulty batch runs in the process industry. 
At first, batch runs are normalized regarding their lenght based on Dynamic Time Warping with help of the R package dtw (https://cran.r-project.org/web/packages/dtw/). Subsequently, I implemented multiway principal component analysis identifiying batches with unnormal behaviour (Nomikos, Paul and MacGregor, John F. (1994). “Monitoring batch processes using multiway principal component analysis”. In: AIChE Journal 40.8).

Additionaly, I applied non-linear kernel principal component analysis for improved results (Lee, Jong-Min, Yoo, ChangKyoo, Choi, Sang Wook, Vanrolleghem, Peter A, and Lee, In-Beum (2004). “Nonlinear process monitoring using kernel principal component analysis”. In: Chemical Engineering Science 59.1, pp. 223–234.).

The source code is split into four parts:
The folder "alignment" contains code for applying DTW on batches with different lenghts. The folders "mpca" and "kpca" contain the relevant files for performing the MPCA and KPCA analysis respectively. In "Visulation", there are some files for creating additional plots.
