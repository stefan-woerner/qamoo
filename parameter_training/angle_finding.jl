# See https://lanl.github.io/JuliQAOA.jl/dev/angle_finding/
using JuliQAOA
using NPZ

num_qubits = 27
prob_index = 0
num_obj = 3
mixer = mixer_x(num_qubits)

filename = "maxcut_0s_$(num_obj)o_$(prob_index)_$(num_qubits)q"
obj_vals = npzread("cost_values/"*filename*".npz")["data"]

p = 1
println("Beginning p=$p angle finding")
angles_x, evs_x = find_angles_bh(p, mixer, obj_vals, max=false, niter=1);
angles_x = [clean_angles(a, mixer, obj_vals) for a in angles_x]
println("Angles", angles_x)
println("Energies", evs_x)

