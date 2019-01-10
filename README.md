# Comparing Community Detection Algorithms

In this final project, the goal was to explore various graphs and to find community structures within them. The following algorithms were developed:

* Spectral clustering 
* Modularity optimization [1]
* Label propagation [2]
* DeepWalk [3]

Several datasets from Stanford Large Network Dataset Collection [4] were utilized for the comparison.

The spectral clusering algorithm showed its power in this project. Label propagation proved to be a fast algorithm, but very unstable: results varied from run to run significantly. DeepWalk had a solid performance with all graphs, but failed to achieve best possible performance with any graph. Label propagation had poor performance with all graphs when compared to other approaches. However, it was significantly faster to calculate.

The results are easily reproducible with algo_gym.py script.  Run python algo_gym.py -h for different options. The outputs from  different algorithms are saved into log.csv. Requirements can be installed with pip using the requirements.txt file. Beware, some of the algorithms have a significant runtime. 

## References
1. Clauset, A., Newman, M.E. and Moore, C., 2004. Finding community structure in very large networks. Physical review E, 70(6), p.066111.
2. Raghavan, U.N., Albert, R. and Kumara, S., 2007. Near linear time algorithm to detect community structures in large-scale networks. Physical review E, 76(3), p.036106.
3. Perozzi, B., Al-Rfou, R. and Skiena, S., 2014, August. Deepwalk: Online learning of social representations. In Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data min- ing (pp. 701-710). ACM.
4. Leskovec, Jure, and Andrej Krevl. ”SNAP Datasets:Stanford Large Network Dataset Collection.” (2015)
