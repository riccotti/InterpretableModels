# Evaluating the Stability of Interpretable Models
**Riccardo Guidotti, Salvatore Ruggieri**    
Department of Computer Science, University of Pisa, Italy  
riccardo.guidotti@unipi.it, salvatore.ruggieri@unipi.it

Interpretable classification models are built with the purpose of providing a comprehensible description of the decision logic to an external oversight agent. When considered in isolation, a decision tree, a set of classification rules, or a linear model, are widely recognized as human-interpretable. However, such models are generated as part of a larger analytical process, which, in particular, comprises data collection and filtering. Selection bias in data collection or in data pre-processing may affect the model learned. Although model induction algorithms are designed to learn to generalize, they  purse optimization of predictive accuracy. It remains unclear how interpretability is instead impacted. Using this software, we conduct an extensive experimental analysis to investigate whether interpretable models are able to cope with data selection bias as far as interpretability is concerned. The software consists of an experimental framework parametric to pre-processing methods and classification models. The approach is implemented, released as open source, and extensible to new models and methods.

## References

[1] R. Guidotti, S. Ruggieri. [On The Stability of Interpretable Models](https://arxiv.org/abs/1810.09352). International Joint Conference on Neural Networks (IJCNN 2019) : paper N-19575. IEEE, July 2019.
