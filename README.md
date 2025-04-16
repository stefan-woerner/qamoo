# Quantum Approximate Multi-Objective Optimization (arXiv:2503.22797)
Code and data accompanying the paper 
[Quantum Approximate Multi-Objective Optimization](https://arxiv.org/abs/2503.22797) 
by Ayse Kotil, Elijah Pelofske, Stephanie Riedmüller, Daniel J. Egger, Stephan Eidenbenz, Thorsten Koch, and Stefan Woerner.

The code uses Python 3.10 and most requirements can be installed via `pip install -e .`. 
Additionally, the python code requires `pygmo` (2.19.7) which is best installed following the guidance here: https://esa.github.io/pygmo2/ (also see the comments below on the installation).

To train parameters, [JuliQAOA](https://arxiv.org/abs/2312.06451) needs to be installed:
https://github.com/lanl/JuliQAOA.jl.

Adjusted versions of the exact classical algorithms DPA-a and DCM have been used.
The corresponding code can be found here: https://github.com/stephaniehoffmann/classical-algorithms-moip
The scripts that show how to use these modified algorithms are provided here.

The repository is structured as follows:
- `classical_algorithms`: Scripts to run the exact classical algorithms (DPA-a / DCM) provided in https://github.com/stephaniehoffmann/classical-algorithms-moip.
- `data`: Problem instances and results data.
- `figures`: All figures shown in the paper.
- `notebooks`: Notebooks to run the algorithms (except DPA-a, DCM, and JuliQAOA, cf. corresponding folders) and create figures.
- `parameter_training`: Code to run JuliQAOA to train parameters.
- `qamoo`: Python code for all algorithms presented in the paper (except DPA-a / DCM, cf. corresponding folder).


### Installing pygmo2

The following installs `pygmo` using `homebrew`: 

```
brew install cmake boost
brew install pagmo
brew install pybind11
```
 
Change to project folder:
 
```
git clone https://github.com/esa/pygmo2.git
cd pygmo2
mkdir build 
cd build 
cmake .. -DPYTHON_EXECUTABLE=$(which python3)
make
sudo make install
```
