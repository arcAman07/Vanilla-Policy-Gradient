# Vanilla Policy Gradient Implementation

This repository contains a PyTorch implementation of the Vanilla Policy Gradient algorithm, applied to the CartPole-v1 environment from OpenAI Gym. This project serves as an educational resource for understanding and implementing basic policy gradient methods in Reinforcement Learning.

## Algorithm Overview

The Policy Gradient algorithm is a foundational method in Reinforcement Learning for learning a policy directly. Unlike value-based methods, policy gradient methods optimize the policy directly, making them particularly useful for continuous action spaces and stochastic policies.

Key features of this implementation:
- Neural network policy using PyTorch
- Categorical action sampling
- Reward discounting and normalization
- Policy gradient loss computation and optimization

## Original Paper

This implementation is based on the concepts introduced in the following paper:

Williams, Ronald J. "Simple statistical gradient-following algorithms for connectionist reinforcement learning." Machine learning 8.3 (1992): 229-256.

[Link to the paper](https://link.springer.com/article/10.1007/BF00992696)

## Implementation Details

The code structure is as follows:

1. `Policy` class: Defines the neural network architecture for the policy.
2. `compute_returns` function: Calculates discounted returns.
3. `train_vpg` function: Implements the main training loop for the policy gradient algorithm.

The policy network consists of two hidden layers with ReLU activations, followed by a softmax output layer for action probabilities.

## Usage

To run the training:

```python
python vanilla_policy_gradient.py