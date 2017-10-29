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

_This short series of posts aims to be a brief and practical introduction to the framework of probabilistic graphical models, by using the familiar Naive Bayes as a springboard to the more general class of Bayesian network structures._

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

Continuing in the medical diagnosis setting, here is a real-world example of a Bayesian network:

![ICU-alarm bayesian network](/assets/images/ICU-Alarm.png){:class="centred-image"}

The ICU-ALARM network from _[Beinlich et al. (1989)](http://cs.brown.edu/courses/csci2420/assignments/alarmNetwork.pdf)_ was designed to monitor the condition of patients in Intensive Care Units, and is an example of a network constructed via knowledge engineering, rather than through machine learning. In it, we can see nodes for various diagnoses towards the top, and observed variables mostly as leaf nodes, sometimes connected via latent variables. Using conditional independence assumptions here doesn't just simplify the presentation, it also makes the computational problem tractable. With 37 different nodes, each representing a categorical variable, there is a total of ~500 local CPD parameters. If the graph was fully connected, even with the simple case where all the variables were binary, we would still have ~137 billion independent parameters.

## Naive Bayes as a Bayesian network

Naive Bayes is a family of Bayesian classifiers[^3] that have very natural interpretations as probabilistic graphical models. It is defined by it's assumption of class-conditional independence. That is, if $$C$$ is the class random variable and $$X_1,...,X_n$$ are the features, then it assumes that $$X_i \perp \mathbf{X}_{-i}\ \vert\ C$$ [^4]. Which means we can write the joint probability as:

$$P(C,X_i,...,X_n) = P(C) \prod_i^n P(X_i \vert C)$$

From which we can construct the network:

![naive bayes graphical structure](/assets/images/naive-bayes.png){:class="centred-image"}

Note the difference here with the example disease network; in that case, there was reason to believe the features were independent. In the general case, when applying Naive Bayes, the class-conditional independence is just a simplifying assumption. However, even when this assumption is clearly violated, it can still produce surprisingly good results. 

Take for example the problem of spam classification using bag-of-words features; clearly the occurrence of different words are not independent, even if we know if the email is spam or not. Certain words occur more often within the same sentence, and of course the overall subject matter of the email will also effect certain correlations. We can tackle the first problem by taking bi-grams or tri-grams as features, but there's no such simple solution to the other problems. The end result of such violated conditional-independence assumptions is that the actual probability estimates Naive Bayes gives us tend to be skewed towards the extremes.

## Tree-Augmented Naive Bayes and the Chow-Liu Algorithm

Let's look at the simplest possible extension to Naive Bayes, which will allow us to take into account some correlations between different variables, whilst still being fairly simple in terms of implementation, computational complexity, and number of independent parameters.

### Footnotes

[^1]: For an interesting example of this see _[Sachs et al. (2005)](http://science.sciencemag.org/content/sci/308/5721/523.full.pdf)_, which applied Bayesian network learning to protein signalling pathways, predicting a novel casual relationship which was then experimentally verified.

[^2]: As a quick reminder, two random variables $$X$$ and $$Y$$ are conditionally independent given $$Z$$, written as $$X \perp Y\ \vert\ Z$$, if and only if $$P(X, Y \vert Z) = P(X \vert Z) P(Y \vert Z)$$.

[^3]: A Bayesian classifier is simply one which predicts according to $$\hat{y} = \underset{y}{\arg\max} P(Y=y\vert X=x)$$. It has the nice property of being theoretically optimal.

[^4]: Using the shorthand notation $$\mathbf{X}_{-i} = \{X_j \vert i \neq j \}$$.
