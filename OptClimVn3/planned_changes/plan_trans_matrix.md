# Planned changes to better handle Covariances and tranform_matrix

## Problems
- Current code around Covariance calculation is a bit horrid. Want to add some way of empirically scaling covariances and have 
same logic for all covariance matrices. 
- need a way of having transform matrix be regularized or truncated from config file. 

So bump version of config file to 4. (from 3)

Structure of each covariance matrix entry is a dict containing the following keys:
  path -- gets passed through expand to convert to a path
  diagonalize -- if True diagonalize the matrix
  importance_scaling -- scaling to apply to elements of covariance matrix.
  Anything else ending in _comment gets striped and the directory ** expanded is passed to read_covariances. 

transform_matrix gets a control block called transform_matrix in the covariances block. Has the following elements
regularize -- value to regularize the matrix with.  If None no regularization is done.
min_evalue -- minumum fraction of max eigenvalue that  eigenvalue must have for that eigenvector/value to contribute to transform matrix. This is an implementation truncation
  If None no truncation is done
warn_scale -- ratio of min to max eigenvalues (after truncation) below which a warning is given. If None no warning given.
Example case (used for testing ) have default values consistent with those used in vn3 transMatrix code.

# Add new methods
1 read_covariance -- reads in covariance and does processing returnign None (nothing found) or pandas dataframe of cov matrix
2 transform_matrix -- computed transform_matrix from TotalCov matrix applying regularisation 

make readCovariance and transMatrix in vn4 raise NotImplementedError. 
Have transform_matrix in vn3 call transMatrix. 

These will generate changes across the code base.  So need to update to use new configurations. 

