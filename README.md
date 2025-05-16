# PSO-FeatureFusion
** A General Framework for Fusing Heterogeneous Features via Particle Swarm Optimization
##  Overview
Bioinformatics often deals with complex and high-dimensional biological datasets derived from heterogeneous sources. Traditional feature extraction and integration methods frequently overlook the interdependencies among biological entities (e.g., drugs, diseases, proteins), which limits the performance of predictive models.

**PSO-FeatureFusion** is a novel framework that integrates **Particle Swarm Optimization (PSO)** with **Neural Networks** to optimize and fuse feature representations across multiple biological entities. It aims to capture intricate inter-feature relationships while maintaining entity-specific characteristics.

## Key Features

- Joint optimization and feature fusion using PSO
- Flexible architecture compatible with:
  - Neural Networks
  - Machine Learning and Statistical Models
  - Network-based Methods
- Built-in support for binary classification tasks
- Easy extension to other biological prediction problems

## Implementation 

The framework was tested on two real-world problems:
1. **Polypharmacy Side Effect Prediction** – using drug-drug interaction features.
2. **Drug-Disease Association Prediction** – using diverse features from drug and disease profiles.

### Prerequisites
```bash
pip install -r requirements.txt
