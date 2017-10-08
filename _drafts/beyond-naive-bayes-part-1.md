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

description: A brief and practical introduction to Probabilistic Graphical Models through augmenting Naive Bayes with the Chow-Liu algorithm
---

_This short series of posts aims to be a brief and practical introduction to the framework of probabilistic graphical models by using the familiar Naive Bayes as a springboard to the more general class of Bayesian network structures._

## Why Probabilistic Graphical Models?

If you've done any machine learning, you've almost certainly already used algorithms that can be understood as specific cases of probabilistic graphical models (PGM), such as Naive Bayes, Hidden Markov Models, Restricted Boltzmann Machines, Gaussian Mixture Models, and Latent Dirichlet Allocation. This probably makes PGMs sound like a very broad topic. In fact it's even broader than that; the scope of the framework extends beyond just machine learning, such as to the case where we have already defined a probabilistic model for our data and want it to answer our probability queries. What links it all together, thus warranting PGMs as a more than just a taxonomy, is the representation of probabilistic models as graphs, with variables as the nodes, and independencies and causal relationships encoded in the connectivity.

The framework of PGMs relies on results from many domains, but in particular is dependent on results from computer science, probability theory, and statistical learning. Whilst it's not as hot right now in the ML research community as neural networks, it offers several standout reasons that warrant its study by a practitioner. 

Firstly, as already alluded to, it can help us achieve a deeper understanding of many well known generative models and the relationships between them, which helps in evaluating their respective trade-offs. Also alluded to is the ability to answer more general queries with our learned models than just predicting a target variable, such as providing marginal probabilities over all variables, calculating MAP estimates, determining casual relationships[^1], and more.

The ability to incorporate prior domain knowledge to various degrees in a natural way is yet another fairly unique property of PGMs. For example an expert could provide all or part of the network structure, provide priors over the parameters of the network, or inform us of important unobserved/latent variables. Alternatively if our goal is knowledge discovery, we can hold back this extra knowledge and use it to validate our learned structures. 

Finally, from a data science perspective, PGMs are very good at handling "messy" data, such as missing observations, and can also handle categorical data natively.

## What is a Bayesian network?

A Bayesian network is one of the two most common types of graphical models. It is a directed acyclic graph (DAG), where each variable in the distribution is an individual node on the graph. The other most common type of graphical model is a Markov network, which is undirected. Partially directed or "Hybrid" networks are also possible, and there are other representations used in certain algorithms that might map multiple variables to a single node, but generally when we're dealing with a PGM as either an input or an output it will be a Bayes net, Markov net, or some hybrid of the two.

In a Bayes net, each variable/node is associated with a conditional probability distribution (CPD), also known as the local probabilistic model. It is a key constraint that the only other variables in the local distribution are the parents of the node. So if we let $$P(X_1,...,X_n)$$ be the joint distribution defined by any Bayes net with a given structure $$G$$, and denote by $$\mathrm{Pa}_{X_i}^G$$ the parents of node $$X_i$$ then it follows that:

 $$P(X_1,...,X_n)=\prod_{i=1}^n P(X_i \vert \mathrm{Pa}_{X_i}^G)$$

Which is known as the chain-rule for Bayesian networks, in reference to to the more general chain-rule for probability, which states that the following holds for _any_ distribution:

$$P(X_1,...,X_n) = \prod_{i=1}^n P(X_i \vert X_1,...,X_{i-1})$$

Putting these two equations together, we can show that a Bayesian network is merely a re-parametrisation of a probability distribution. We can convince ourselves of this through a simple constructive argument: taking the CPDs from the chain rule for probability with some fixed but arbitrary ordering as the local distributions, we then connect the graph according to $$\mathrm{Pa}_{X_i}^G = \{X_1,...,X_{i-1}\}$$. The graph then has a topological ordering of $$X_1,...,X_n$$ and is therefore acyclic, satisfying the definition of a Bayesian network.

In other words, any distribution can be encoded as a Bayes net with the structure of a transitive tournament (i.e. an acyclic orientation of the complete graph), represented here for five variables:

![complete directed acyclic graph of five nodes, in topological order](/assets/images/K5-DAG-topo-order.png){:class="centred-image"}

However, just because we can, doesn't mean we should! In fact, in order to reduce the number of parameters, which helps with both learning and inference, we would like to use a graph structure $$G$$ for the distribution $$P$$ that has the minimum number of edges possible, whilst still being able to encode $$P$$. Such a graph is called a minimal I-map (independency map) for $$P$$, which is a relation of the conditional independencies[^2] that hold in $$P$$ and those that are induced by $$G$$.

Finding the set of conditional independencies that hold in a given graph structure is beyond the scope of this post, instead we continue to take a constructive approach.

For example, say there are two unrelated screening tests for a certain medical condition, which occurs in the population with some know probability. Neither test is perfect though, each test has its own sensitivity and specificity. If we conduct one screening test, naturally that would give us information as to the likely outcome of the other test. However, if we had some way of directly observing whether a patient has the medical condition, say through surgery, then knowing the outcome of one test would likely give us no extra information about the other. In this case we might assume that the two tests are conditionally independent if we know whether the patient really has the condition or not. So we could factorise the joint distribution like so:

<p>
$$
\begin{align}
P(D, T_1, T_2) &= P(D) P(T_1, T_2 \vert D) \\
               &= P(D) P (T_1 \vert D) P(T_2 \vert D)
\end{align}
$$
</p>

From which we can construct the simple network:

![bayesian network example](/assets/images/3Node-BayesNet-Diagnostic.png){:class="centred-image"}

Compare this with the expansion via the chain rule without the assumption of conditional independence, and the associated Bayes net structure.

{::comment}
Finding the set of conditional independencies that hold in a given graph structure is beyond the scope of this post unfortunately, but there are a few results that we should take note of to help with conceptualising Bayes nets. Firstly, as we noted in construction of a Bayes net from a general distribution, the ordering of the expansion was arbitrary. Thus even in the case of a distribution with no conditional independencies, there are multiple minimal I-maps. This holds true in general, there can be many possible I-maps for a given set of conditional independencies. We can view these sets as equivalence classes, inducing a relation over DAGs called I-equivalence. 

Of further note is that we carefully phrased the definition of a minimal I-map, in that we did not require the conditional independencies that hold in the distribution to be exactly equal to those induced by the graph, which is known as a perfect I-map. Not all distributions have a Bayes net that is a perfect map. Usually this is the case when there is local structure in the CPDs, since Bayesian network structures only capture relationships at the variable level.
{:/comment}

## Naive Bayes as a Bayesian network

![naive bayes graphical structure](/assets/images/naive-bayes.png){:class="float-right"}

Naive Bayes is a family of bayesian classifiers that have very natural interpretations as a probabilistic graphical models. It is possibly the simplest such type of network

<div style="clear: both;"></div>

{::comment}
Consequences:
- any two variables that do not have a directed path between them are conditionally independent a priori.



_In this post we'll show how to make Naive Bayes less naive by introducing tree structured dependencies using the Chow-Liu algorithm, resulting in a model with lower bias. We'll also use this as a very brief introduction to the broader class of probabilistic graphical models._

Talking points:
- Naive Bayes isn't so much an algorithm as it is a network structure 
- Naive Bayes owes its name to the very strong independence assumptions it makes about features. Specifically it assumes that all the features are conditionally independent given the class.
- narrow down the independecies (however in empirical distribution these are never there)
- perfect maps (not always possible with Bayes Nets)

More formally, if $$I(\mathcal{P})$$ is the set of conditional independencies that hold in a distribution $$\mathcal{P}$$, and $$P_{G}$$ is the joint distribution induced by a Bayesian network with graph $$G$$, then $$G$$ is an I-map for $$P$$ if and only if $$I(P_{G}) \subseteq I(P)$$, and is a minimal I-map if and only if the removal of any edge would render it no longer an I-map. Notice that we did not require $$I(P_{G}) = I(P)$$, so even if an I-map is minimal, there may still be conditional independencies which hold in $$P$$ but are not asserted by $$G$$. The existence which is called a perfect I-map, and may not always exist as a Bayesian network for a given distribution.

{:/comment}

### Footnotes

[^1]: For an interesting example of this see _[Sachs et al. (2005)](http://science.sciencemag.org/content/sci/308/5721/523.full.pdf)_, which applied Bayesian network learning to protein signalling pathways, predicting a novel casual relationship which was then experimentally verified.

[^2]: As a quick reminder, two random variables $$X$$ and $$Y$$ are conditionally independent given $$Z$$, written as $$X \perp Y \vert Z$$, if and only if $$P(X, Y \vert Z) = P(X \vert Z) P(Y \vert Z)$$.
