# Saltation simulation code
The code may be compiled (on a UNIX-like system) by invoking either

> ./compile.sh Saltation_ConstantPrevalenceBranching.cpp

(for the constant prevalence branching model simulations)

or
> ./compile.sh Saltation_SIRS_ABM_strainspecific.cpp

(for the agent-based simulations with waning immunity and possibility of tunably strain-specific immunity).

In both cases, the resulting executable will be called 'branching'. 

The above assumes that compile.sh has first been made executable, by e.g. executing

> chmod +x compile.sh

