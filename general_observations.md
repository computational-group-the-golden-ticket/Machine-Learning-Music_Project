1. Run the net over the notes first. 
2. There is more error as we go further, should we implement something for 
this?
Perhaps this will guide to a better idea: 
https://machinelearningmastery.com/what-is-imbalanced-classification/
3. Make tests with the optimizator to how iw behaves respect the number of
steps that it makes.
4. Read why to use the Adamoptimizer: https://machinelearningmastery.com/
adam-optimization-algorithm-for-deep-learning/
5. Use a mapping from the distributions withouth giving the songs the same
weight.
6. Sum something additional to the cost function. 
7. Change interval_len, batch len variables to have a dependency with the 
number of notes that can be reolved or at least give better names.
8. See the meaning of the conservativity.
9. Review mask in training. 