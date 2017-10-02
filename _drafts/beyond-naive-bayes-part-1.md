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

If you've done any machine learning, you've almost certainly already used algorithms that can be understood as specific cases of probabilistic graphical models, such as Naive Bayes, Hidden Markov Models, Restricted Boltzmann Machines, Gaussian Mixture Models, and Latent Dirichlet Allocation. This probably makes PGMs sound like a very broad topic. In fact it's even broader than that; the scope of the framework extends beyond just machine learning, such as to the case where we have already defined a probabilistic model for our data and want it to answer our probability queries. What links it all together, thus warranting PGMs as a more than just a taxonomy, is the representation of probabilistic models as graphs, with variables as the nodes, and independencies and causal relationships encoded in the connectivity.

The framework of PGMs relies on results from many domains, but in particular is dependent on results from computer science, probability theory, and statistical learning. Whilst it's not as hot right now in the ML research community as neural networks, it offers several standout reasons that warrant it's study by a practitioner. 

Firstly, as already alluded to, it can help us achieve a deeper understanding of many well known generative models and the relationships between, which helps in evaluating the trade offs between them. Also alluded to is the ability to answer more general queries with our learned models than just predicting a target variable, such as providing marginal probabilities over all variables, calculating MAP estimates, determining casual relationships[^1], and more.

The ability to incorporate prior domain knowledge to various degrees in a natural way is yet another fairly unique property of PGMs. For example an expert could provide all or part of the network structure, provide priors over the parameters of the network, or inform us of important unobserved/latent variables. Alternatively if our goal is knowledge discovery, we can hold back this extra knowledge and use it to validate our learned structures. 

Finally, from a data science perspective, PGMs are very good at handling "messy" data, such as missing observations, and they can also handle categorical data natively.

## What is a Bayesian network?

A Bayesian network is one of the two most common types of graphical models. It is a directed acyclic graph, where each variable in the distribution is an individual node on the graph. The other most common type of graphical model is a Markov network, which is undirected. Partially directed or "Hybrid" networks are also possible, and there are other representations used in certain algorithms that might map multiple variables to a single node, but generally when we're dealing with a PGM as either an input or an output it will be a Bayes net, Markov net, or some hybrid of the two.

In a Bayes net, each variable/node is associated with a conditional probability distribution, also known as the local probabilistic model. It is a key constraint that the only other variables in the local distribution are the parents of the node. So if we let $$P(X_1,...,X_n)$$ be the distribution defined by any Bayes net with a given structure $$\mathcal{G}$$, and denote by $$\mathrm{Pa}_{X_i}^\mathcal{G}$$ the parents of node $$X_i$$ then it follows that:

 $$P(X_1,...,X_n)=\prod_{i=1}^n P(X_i \vert \mathrm{Pa}_{X_i}^\mathcal{G})$$


Which is known as the chain-rule for Bayesian networks, in reference to to the more general chain-rule for probability, which states that the following holds for _any_ distribution:

$$P(X_1,...,X_n) = \prod_{i=1}^n P(X_i \vert X_1,...,X_{i-1})$$


From these two equations we can deduce that a Bayesian network is merely a re-parameterization of a probability distribution, and as such does not place any restrictions of what types of distributions it can encode. To see why this is true, consider the most general case of a complete directed acyclic graph, drawn here for five variables:

![complete directed acyclic graph of five nodes](/assets/images/K5-DAG.png){:class="centred-image"}

Even if we don't make any assumptions about independence in a distribution, we can always represent it with such a graph. Inductively, starting with a single node, and every time we add a node, connecting it with edges from every previously added node, at every step we will have a complete directed acyclic graph such that it's expansion via the chain rule for probabilities is exactly the same as it's expansion via the chain rule for Bayesian networks, i.e.:

$$\prod_{i=1}^n P(X_i \vert X_1,...,X_{i-1}) = \prod_{i=1}^n P(X_i \vert \mathrm{Pa}_{X_i}^\mathcal{G})$$

Consequences:
- any two variables that do not have a directed path between them are conditionally independent a priori.

## Naive Bayes as a Bayes net

![naive bayes graphical structure](/assets/images/naive-bayes.png){:class="float-right"}

Naive Bayes is a family of bayesian classifiers that has a very natural interpretation as a probabilistic graphical model. It is possibly the simplest such type of

<div style="clear: both;"></div>

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

{:/comment}

Equations:

$$
  P(C,X_1,...,X_n) = P(C) \prod_{i=1}^n P(X_i|c) \\
  P(X_1,...,X_n) = P(C) \prod_{i=1}^n P(X_i|c)
$$





