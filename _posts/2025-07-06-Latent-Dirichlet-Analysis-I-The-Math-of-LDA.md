---
layout: post
title:  "Latent Dirichlet Analysis I - The Math of LDA"
date:   2025-07-06 00:00:00 +0000
categories: TopicModeling LDA
---


intro...


**pic**



# Outline 
- ...
- ...



# Introduction

Latent Dirichlet Allocation (LDA), introduced in <a href="">this 2003 paper</a>, is a clustering method widely used for the topic modeling of text documents in a corpus, with applications extending to other domains.

As a quick sidenote, it is unrelated to Linear Discriminant Analysis, a dimension-reduction technique which shares the acronym LDA.

Latent Dirichlet Allocation is an unsupervised learning method, meaning topic-labels for the documents are not provided during training, nor produced by the model. Instead, we provide the number of topics $K$, and expect the analysis to group and distinguish documents in a meaningful way (if that fails, adjust $K$ or review your training and preprocessing tactics). By injecting some human interaction, we can utilize a process by which we make predictions in a supervised way (called sLDA), and we'll explore that in the next article.



# Intuition

LDA is a soft-clustering method, facilitating overlapping topic assignments. Rather than each document belonging to exactly one cluster (as a method like k-means would do), LDA assigns:

- A probability distribution over topics to each document.
  - e.g., a document is 40% represented by fiction, 20% by science, and so on with regard to other topics.

- A probability distribution over words to each topic.
  - e.g., a topic is represented 5% by "quantum", 2% by "teleport", and so on with regard to other words (perhaps thousands of them).

Imagine a library where many books are strewn across the tables and floor, with no labels or organization. This collection of books represents our corpus. You need to create an algorithm to sort them into sections, but you do not know which categorizations exist beforehand. The books are our documents, the unknown categorizations are our topics, and the words are clues as to which topic a book belongs to. If a book has words like "experiment" and "hypothesis", the latent category corresponding to 'science' may be the one it predominantly belongs to; however, it will also belong many other categories with smaller (often much smaller) probability, due to the fact that many words in our vocabulary transcend a particular subject.

To obtain distributions rather than crisp predictions, we typically follow a generative, Bayesian process, iteratively refining expectations through sampling (though alternative methods exist). The term 'latent' refers to the fact that we are uncovering hidden structure, and 'Dirichlet' refers to the Dirichlet distribution, described below. 



# The Dirichlet Distribution

Recall that the Beta distribution is dubbed the probability distribution for probabilities, because its values (and not just the associated probabilities) take on the range of $0$ to $1$. The Beta is a special case of the Dirichlet where $K$, the number of categories, is equal to $2$. The Dirichlet is an abstraction of the Beta, dubbed a "distribution over distributions", specifically with regard to categorical data, as it obeys 'the simplex constraint' by generating probability vectors that sum to 1 for any discrete number of categories, serving as a prior for multiple Multinomial distributions.

The probability distribution/mass functions for the above-mentioned distributions are as follows, but I don't intend for you to extract a complete intuition from the formulas alone.

**formula for Beta**

**formula for Multinomial**

**formula for Dirichlet**

If you have read my <a href="">first article</a> on continuous probability distributions (the second on distributions in general), you may have noticed that the Dirichlet is the only one not visualized, and this is due to the nature of its geometry requiring some additional explanation. It can be characterized as a <a href="">simplex</a>, which is the simplest possible <a href="">polytope</a> (a geometric object with flat sides) in any given dimension. A simplex can be visualized as a line in 1D space, a triangle in 2D space, and a tetrahedron in 3D space. Below, I use the 'stick-breaking' analogy to illustrate. Our LDA results, while not strictly a visualization of the Dirichlet, may provide the most intuitive explanation.


## The Stick-Breaking Analogy

Let's say you have a stick with a length of $1$ (in any unit of measurement), which we'll consider to represent 100% of a probability mass. You proceed to break the stick into $K$ pieces, representing probabilities $p_1, p_2, \ldots, p_K$. 

The length of the first piece broken off represents $p_1$, and is drawn from a Beta distribution, leaving a probability mass of $1 - p_1$ behind. We break another piece $p_2$ from the remainder, and so on, with the $K$ pieces always forming a probability vector. If the end points are connected, we have a triangle after two breaks, a tetrahedron after three breaks, and then a generalized simplex in the dimensions beyond our physical realm. The Dirichlet distribution governs these splits, generalizing the Beta distribution from $K=2$ to any $K$.

The Dirichlet's parameters $\mathbf{\alpha} = \{\alpha_1, \alpha_2, \ldots, \alpha_K\}$ shape the splits.

- Large $\alpha_i$ favor bigger pieces for component $i$
- Equal $\alpha_i$ yield balanced splits
- Small $\alpha_i$ produce sparse splits near the simplex's edges (vertices or boundaries). To plot this in a triangle with equal-length edges, we can use shading or a color-scale to represent the level of probability concentration along the various edges.

**visual**

Before we get to visualizing LDA, I would like to discuss a middle ground between the mathematical symbols of the Dirichlet and the plots of model results, which is probabilistic graphical modeling using plate notation. 


# Plate Notation

Plate notation is a form of probabilistic graphical model (PGM), which you may be familiar with if you have experience with Bayesian modeling (through a library like PyMC). These use nodes (circles) and arrows or edges to convey the relationship between model parameters, as in Bayesian modeling, the parameters are treated as random, and influenced by upstream parameters.

**visual**

Plate notation visually represents PGMs using:

- Circles (nodes) for random variables.
- Rectangular plates for replication over samples or dimensions.
- Arrows for dependencies between variables.


### Unigram Model

Below, node $x$ represents a particular sequence or word count. The $D$ in the encompassing plate can be thought of as a loop over dimensions, and the $N$ in the plate encompassing that can be thought of as a loop over samples. 

**Unigram Plate Diagram**

The formula this represents is:

$p(x) = \prod_{j=1}^D p(x_j)$

We would represent this in pseudocode as:

```
for i = 1 to N:
  for j = 1 to D:
    x(i,j) ~ p(x)
```


### Mixture Model

Next, we'll step it up to a mixture model, which involves a combination of multiple ($K \gt 1$) component distributions.

**mixture img**

In this case, the probability of a document observing a specific sequence or word count is as follows:

<p>$p(x) = \sum_z p(z) \prod_{j=1}^D p(x_j | z)$</p>

And we would represent this in pseudocode like the following:

```
for i = 1 to N:
  z(i) ~ p(z)
  for j = 1 to D:
    x(i,j) ~ p(x|z = z(i))
```


### LDA Plate Notation

As mentioned above, LDA models documents as mixtures of topics, with each word assigned a topic.

**LDA Plate Diagram**

Perhaps it's best to start with the pseudocode on this one:

```
α, β = fixed parameters
for i = 1 to N:
    θ(i) ~ Dirichlet(α)
    for j = 1 to D:
        z(i,j) ~ Multinomial(θ(i))
        x(i,j) ~ Multinomial(β, z(i,j))
```


<!--
** merge the below **

Key parameters:
- $\alpha$: Dirichlet prior for document-topic proportions $\theta$
- $\beta$: Topic-word distribution (fixed or learned)
- $\theta$: Document-specific topic proportions, drawn from $\text{Dirichlet}(\alpha)$
- $z$: Topic assignment for each word, drawn from $\text{Multinomial}(\theta)$
- $x$: Observed word, drawn from $\text{Multinomial}(z, \beta)$

To explain:
- $\alpha$ and $\beta$ are fixed outside plates, indicating they’re global
- $\theta$, sampled per document, governs topic proportions
- $z$, sampled per word within each document, assigns topics based on $\theta$
- $x$, the observed word, depends on $z$ and $\beta$
- Unlike mixture models, LDA assigns a topic to each word, not the entire document, enabling fine-grained topic mixtures


The probability of a document is:

<p>$p(x) = \int_\theta p(\theta | \alpha) \sum_z p(z | \theta) \prod_{j=1}^D p(x_j | z, \beta) d\theta$</p>




A more concentrated document-topic distribution, driven by a large alpha (α) in LDA, means documents are more likely to be dominated by a smaller number of topics. This results in the count of documents being more concentrated around specific topics, as each document's topic mixture has higher probabilities for fewer topics rather than being spread across many.

A more concentrated topic-word distribution, driven by a large beta (β) in LDA, means topics are dominated by fewer words (i.e., higher probabilities for specific words within each topic).









reworded by Grok


Key Parameters: $\alpha$ (Dirichlet prior for document-topic proportions $\theta$), $\beta$ (topic-word distribution, fixed or learned), $\theta$ (document-specific topic proportions, drawn from $\text{Dirichlet}(\alpha)$), $z$ (topic assignment per word, drawn from $\text{Multinomial}(\theta)$), and $x$ (observed word, drawn from $\text{Multinomial}(z, \beta)$).

Global Parameters: $\alpha$ and $\beta$ are fixed outside plates, serving as global hyperparameters influencing the model.

Document-Level Process: $\theta$ is sampled per document to determine topic proportions, governing the mixture of topics within each document.

Word-Level Process: $z$ is sampled per word within each document, assigning topics based on $\theta$, and $x$ (the observed word) is drawn based on $z$ and $\beta$.

Distinctive Feature: Unlike mixture models, LDA assigns a topic to each word, enabling fine-grained topic mixtures rather than a single topic per document.




-->

