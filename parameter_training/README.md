# Angle Finding

1. Install JuliQAOA. JuliQAOA can be installed [here](https://github.com/lanl/JuliQAOA.jl)
2. Run `python3 save_maxcut_cost_values_to_npz.py`. This will write several large NPZ files to the directory `cost_values/`. This will take at least several hours. 
3. Run `julia angle_finding.jl` to run the angle finding procedure on a specific problem instance. See [this page](https://lanl.github.io/JuliQAOA.jl/dev/angle_finding/). 
Note that this will also take a long time to run, especially for larger p. 


`learned_angles/` contains the list of learned QAOA angles that were used in this study. 
