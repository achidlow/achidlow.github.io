---
title: "Beyond Naive Bayes: Part 1"
date: 2017-08-23 19:35 +0800
layout: post
category: blog
tags:
 - machine-learning
 - probabilistic-graphical-models

# image: /assets/images/markdown.jpg
# headerImage: false
# star: true
author: adam

description: Improving predictions and learning structure through relaxed independence assumptions
---

_This short series of posts aims to be a brief introduction to the framework of probabilistic graphical models by using the familiar Naive Bayes as a springboard to the more general class of Bayesian network structures._

## Why Probabilistic Graphical Models?

If you've done any machine learning, you've almost certainly already used algorithms that can be understood as specific cases under the more general framework of probabilistic graphical models, such as Naive Bayes, Hidden Markov Models, Restricted Boltzmann Machines, Gaussian Mixture Models, and Latent Dirichlet Allocation. In fact the scope of the framework extends beyond just machine learning to the case where we have already defined a probabilistic model for our data and want to use it to answer probability queries in a computationally efficient manner. What links all these things together, thus warranting viewing PGMs as a framework rather than just an issue of taxonomy, is the representation of the probabilistic model as a graphical structure with variables as the nodes, and independencies and causal relationships encoded in the connectivity of the graph.

The framework of PGMs is a huge field that relies on results from many domains, but in particular is dependent on results from computer science, probability theory, and statistical learning. Whilst it's not as hot right now in the machine learning research community as the framework of neural networks, it offers several standout reasons that warrant it's study by a practitioner. 

Firstly, as already alluded to, it can help us achieve a deeper understanding of many well known generative models and the relationships between, which helps in evaluating the trade offs between them. Also alluded to already is the ability to answer more general queries with our learned models than just predicting a target variable, such as providing marginal probabilities over all variables, calculating MAP estimates, determining casual relationships[^1], and more.

The ability to incorporate prior domain knowledge to various degrees in a natural way is yet another fairly unique property of PGMs. For example an expert could provide all or part of the network structure, provide priors over the parameters of the network, or inform us of important unobserved/latent variables. Alternatively if our goal is knowledge discovery, we can hold back this extra knowledge and use it to validate our learned structures. 

PGMs can also handle "messy" data, such as data where there are values missing at random. They can also handle categorical data natively.

Finally, by incorporating utility functions, PGMs can facilitate decision making in the face of uncertainty.

## What is a PGM?

## Naive Bayes as a PGM

Naive Bayes is a family of bayesian classifiers that has a very natural interpretation as a probabilistic graphical model. It is possibly the simplest such type of


### Footnotes

[^1]: For an interesting example of this see _[Sachs et al. (2005)](http://science.sciencemag.org/content/sci/308/5721/523.full.pdf)_, which applied Bayesian network learning to protein signaling pathways, predicting a novel casual relationship which was then experimentally verified.

{::comment}
_In this post we'll show how to make Naive Bayes less naive by introducing tree structured dependencies using the Chow-Liu algorithm, resulting in a model with lower bias. We'll also use this as a very brief introduction to the broader class of probabilistic graphical models._

Talking points:
- Naive Bayes isn't so much an algorithm as it is a network structure 
- Naive Bayes owes its name to the very strong independence assumptions it makes about features. Specifically it assumes that all the features are conditionally independent given the class.
- Bayesian networks are probabilistic graphical models with directed edges (undirected is Markov, hybrids are possible and even common)
- A general distribution P can be represented using the chain rule for probabilities. This form makes no assumptions about indepedencies.
- Chain rule for bayesian networks
- trivially every network can be represented K_n
- narrow down the independecies (however in empirical distribution these are never there)
- perfect maps (not always possible with Bayes Nets)
- 

Equations:

$$
  P(C,X_1,...,X_n) = P(C) \prod_{i=1}^n P(X_i|c) \\
  P(X_1,...,X_n) = P(C) \prod_{i=1}^n P(X_i|c)
$$



![naive bayes graphical structure](/assets/images/naive-bayes.png){:class="float-right"}


<div style="clear: both;"></div>


{:/comment}
