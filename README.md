# DSC180A-Methodology-5

Test script for running different types of kernels with added label noise.

The current output is sent to `out/results.json`. These can currently be viewed by running the cell in `notebooks/Visualization.ipynb`.


Different combinations of kernels and noise can be run by modifying the `script-params.json` file found in the config folder. The modifiable tags are as follows:

`min_noise` : float - The minimum value of noise to be applied to each kernel
  
`max_noise` : float - The maximum value of noise to be applied to each kernel
  
`noise_step` : float - The step size for iterating through `min_noise` to `max_noise`
  
`p_kernels` : list -> numeric - Each power of kernel that will be used
  
`c_modifiers` : list -> numeric - Each modifier of c that will be used (Value that modifies the denominator of the kernel function)
  
`num_classes` : integer - The number of classes to be used from the dataset (currently only works with 2 classes)
  
