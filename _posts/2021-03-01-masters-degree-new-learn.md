---
layout: post
title: "The topics I would do a PhD in ML/AI"
description: ""
thumb_image: "documentation/sample-image.jpg"
tags: [personal-thoughts, machine-learning,]
---

# Introduction

{% include image.html path="publishing-trends.png"
                      path-detail="publishing-trends.png"
                      width="100%"
                      alt="Topic Growth Trend" %} 
I recently finished my master's degree and I got so much time in these two years to absorb the AI and ML research space. My only goal during the degree was to let my curiosity run wild and follow it without any fears of failures. To fulfil my goal I built a [search engine over CS ArXiv](https://sci-genie.com) and built a lot of models from the papers I read via what I found through the search engine.  The goal allowed me to build and break all ideas I found interesting. The exercise of reading/building made me I fall in love with this domain. I also realized that knowledge in this domain endowed me with new superpowers. 

At end of the degree, I came to a very important realization. I was in a dilemma of whether it is worth doing a PhD or move towards getting a job and working in the industry. Although I love developing these fancy AI and ML systems, but I am not a huge fan of writing papers where there are so many being written (Figure above shows monthly publishing trends on CS ArXiv). The other aspect is that it's important to find problems worth spending 5 years of my life as there are already so many people are working on problems in this domain! 

 <!-- I was really enjoying tinkering and building in the AI/ML space. -->

# Top Thought Directions I Would Explore If I Ever Did a PhD.

The purpose of this blog post is to highlight some ideas, I would explore if I ever pursued a PhD in the next 1-2 years. This post is not meant to be educational. It is meant to inspire ideas and discussions. The other thing I would humbly say is that even after reading/practicing the techniques in this field for the past two years I have barely scratched the surface. The other thing I acknowledge is that many things I write here might have research already done that I am not aware of! Too Many people are working in this field have done this way longer so everything I write here should be taken with a grain of salt as my knowledge may also have gaps. Here Goes. These are the set of ideas I would explore for a PhD if I can't let my creative juices just run wild. 

## 1. Intelligence as Emergence 
Inherently, all fancy machine learning models interpolate within the training data distribution. The current day trend based on this is to throw large enough data at a big enough model, then it can potentially interpolate from large enough patterns seen in the data. A lot of people call this fancy names like the [Manifold Hypothesis](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/). In simplistic terms the ML model finds a *smooth differentiable manifold in the space of data* for which it can have a mapping in the output space. The optimization helps change this mapping between the manifold and the output space and the model helps define the mapping. The formulation of this manifold and its mapping generally happens based on the learning task and inductive biases possessed by the model. Strategies like self-supervised learning with large enough data help capture a large size of this manifold. This is one of the reasons I think why GPT-3 works so well. [With what attention mechanism is doing](https://jalammar.github.io/illustrated-gpt2/) and what the manifold hypothesis implies, we are only creating a larger boundary for interpolation. 

A [recent paper from Don Begio's group](https://arxiv.org/pdf/1806.01261.pdf) included some latest SOTA techniques in the list of inductive biases incorporated in different architectural patterns. Below is a table summarizing that : 
{% include image.html path="inductive-biases.png"
                      width="100%"
                      path-detail="inductive-biases.png"
                      alt="OODG Stats" %}


Attention is king at the current moment (May 2021) based on its popularity in the research community and its performance in various state-of-the-art tasks. Many researchers are actually saying GPT-3 like models can be "intelligent". I don't buy this. GPT-3 like tech makes one hell of a parser which can put many people out of jobs, but it's not the way to reach a "General Purpose Machine"/AI because intelligence is more than *just* interpolation. There are many aspects such as [analogizing](https://arxiv.org/abs/2102.10717), [causal reasoning](https://www.youtube.com/watch?v=mfh4fp_8oPg), [common sense or intuitive physics](https://www.youtube.com/watch?v=llYCXO9Ajj0) (to name a few) that are important for agents *possessing some degree of intelligence.*

<!-- Better inductive biases can be embodied in models which can help capture more traits of intelligence like  -->

Since I have been following this space, I have become am a huge an of thought directions like the ones from [Fie Fie Li with a physics inspired approach of formalizing intelligence](https://www.youtube.com/watch?v=mjWnYO6Zltc) or Karl Fiston's formalism's on the [emergence of intelligence in self-organizing systems with the free energy principle](https://www.nature.com/articles/nrn2787) or [even Kennith Stanley's thesis of the objective itself being the problem](https://www.youtube.com/watch?v=lhYGXYeMq_E). There is one common thing across the thesis's of many of these people is that they assume that intelligence will emerge and is not incepted at the start. This is what truly excites me as a future research direction. I am totally with the idea that intelligence is an emergent phenomenon over an incepted one to truly be generalizable. 

I were to work on problems in AI for five years, I would consider this thought direction which influences the problems I select.
<!-- We are where we are as humans because we stand on shoulders of our ancestors who discovered things which greatly accelerated where we are. *Our intelligence has collectively grown and is colletively shared*. The intelligence of all humans in the past 50 years emerged because of shared knowledge from created by the ancestors of the past.  -->
## 2. How do we define utility ? Features are not bugs. 

{% include image.html path="rl-papers-stats.png"
                      path-detail="rl-papers-stats.png"
                      alt="RL Paper Publishing trends Stats" %}

The above is plot of the number of papers published on ArXiv every month on Reinforcement Learning. Ever since AlphaGo the number of people who have been publishing in RL is has grown and there are just so many algorithms with so many variations based on the way we frame problems. The biggest trick that the RL community ever played was the sleek footnote on the bottom of every paper saying that reward function `R()` *is assumed to be* monotonically increasing function. 

*Sure, to make my robot move around and get me coffee, all I need to do is find this monotonically increasing function that helps me get the desired behavior, OR I'll just give it a few years to hit/trial on sparse rewards and pray for it to work after that.*

What ever people may say there are [countless proofs](https://media.neurips.cc/Conferences/NIPS2018/Slides/jpineau-NeurIPS-dec18-fb.pdf) in the [brittleness](https://arxiv.org/abs/1709.06560) and [variance in RL](https://arxiv.org/abs/2005.12729) due to parametric sensitivity creating reproducibility issue. In my opinion, RL is an amazing framework but we need a better way to understand hyper-non-convex rewards functions and understand the causal effects of utilities much deeper. 

A famous though experiment on the nature of utilities is the [paper clip experiment](https://en.wikipedia.org/wiki/Instrumental_convergence#Paperclip_maximizer). 

>Suppose we have an AI whose only goal is to make as many paper clips as possible. The AI will realize quickly that it would be much better if there were no humans because humans might decide to switch it off. Because if humans do so, there would be fewer paper clips. Also, human bodies contain a lot of atoms that could be made into paper clips. The future that the AI would be trying to gear towards would be one in which there were a lot of paper clips but no humans.
> -- Nick Bostrom

The fascinating thing is that this scenario makes sense. That are many cases of RL algorithms with reward functions tuned according to our intuitions which make the agent [behave in ways we never imagined](https://openai.com/blog/faulty-reward-functions/). If we ever want TRULY General Purpose Technology (GPT) then we need a better understanding of the implications of the way we optimize it. The best example of optimizing systems without understanding implication of optimizations is the polarization in society enabled by of a lot of social media companies optimizing their news feeds purely for engagement creating impacts in the output space (society) for which we had no proper understanding. 

If we want to understand the impact of utilities we need more depth and study over the macro and microanalysis of the effects induced by the optimization of utilities (Economists can roll their eyes on this). 

## 3. How do we define the *degree* of out of distribution generalization? 
{% include image.html path="OOD-G-Stats.png"
                      path-detail="OOD-G-Stats.png"
                      alt="OODG Stats" %}

Out of distribution generalization is the buzz word that comes along in many papers. The above image shows a trend plot of mentions of "out of distribution generalization" in ArXiv papers over the past few years. As generalization is very important to AI, so many papers derive esoteric reasoning and anecdotal examples on how models are generalizing. Even Don Bengio discusses so many ideas on tackling OODG with concepts from sparse factor graphs and "consciousness priors". But as someone who wants to understand quantitatively what it means to "Generalize" *I believe that finding a mathematical framework for describing the degree of generalization based on some from of task structure can make a massive difference on how we perceive AI systems*. 

A very good example to explain degree of generalization is an anecdote from the famous TV show Silicon Valley. In one episode [Jin Yang's company gets acquired because it creates a hot dog classifier which works very well on penile imagery](https://www.youtube.com/watch?v=AJsOA4Zl6Io). This is the best example of a degree of generalization. A classifier generalizing from hot-dog to penile images is a good generalization because inherently the model picked up the common biases in the data needed for the downstream task. 

Ideally, if a classifier that is just seeing cats and dogs, classifies a lion or a wolf as a cat or dog respectively then I would consider this a superb generalization for that classifier. The other thing is that to measure OODG we can explicitly take into account the inherent hierarchical relationships(hot-dog -> penile imagery or  cat -> lion, dog -> wolf) of different tasks/data. If there can be a mathematical framework to help describe the degree of generalization across different tasks for different models then we can have even better understanding of the black boxes we build. The framework can derive from concepts known in math. Formulating a framework that helps describe the degree of generalizing can make a huge difference on the way we understand the models we build. The current day trend of SOTA chasing guides us away from what we as humans call General Intelligence because we don't have a way to fruitfully compare degree of generalization of a model just based on test set accuracy results on some dataset. 


## 4. Interpretability from the lens of abstractions. 
- Can the abstractions within the underlying the data have a language for description and a framework for measurement. 
    - Dogs have ears and so do cats and humans. 
        - Ear is an abstraction that is encapsulated by entities which possess auditory sensing capabilities. 
        - There are many abstractions in underlying data which may be common across many input training points. 
        - Can make machines discover and organize the abstractions better. 

- These abstractions can have many uses:
    - The abstractions provide a way to perform reasoning over different things. 
        - Ears and eyes are abstractions common across many beings. 
            - But they would have a different influence when providing a description of beings. This kind of explicit information is seldom used and given by machines. 
            - A lot of time such knowledge is not Tacit and it's understanding is emergent in humans. 

    - These abstractions can be used as individual reference frames for the measurement of future changes. (Like how we do with Git.)
        - We use git because git provides us a way to reason about the changes in abstraction over a syntax tree.

- Neural networks discover many abstractions within the data, but it is hard to explicitly understand the interpretation of the weight of different abstractions that affect the outcome of the neural network. 


## 5. Finding a ImageNet moment in learning non-stationary dynamics for MARL

Every time we use RL it is assumed with Markovian dynamics that take into account a **stationary** environment. **What it means is that the transition function ( $$\mathcal{P}(s_{t+1} \mid s_t,a_t)$$ ) will be unchanged. Meaning there is will no change in the probability of the next state given the current state and actions.** Let's break this down. If you put few robots learning concurrently in an environment, then the transition probabilities $$\mathcal{P}(s_{t+1} \mid s_t,a_t)$$ of the next state will no longer stay same because the other agents learning simultaneously in this setting can [*induce a non-stationary* environment WRT the individual agent](https://arxiv.org/pdf/1911.10635.pdf). Why does this happen ? Well fundamentally $$\mathcal{P}(s_{t+1} \mid s_t,a_t)$$ captures the probability of the next state $$s_{t+1}$$ given state $$s_{t}$$ and action $$a_{t}$$. With multiple robots learning at the same time, $$s_{t}$$ for an individual robot will fluctuate as the decisions taken by other robots keeps changing (They are all learning). This is a becomes a problem as we have more agent learning at the same time. Adding to the complexity, interaction between agents may be of different types. Interactions may be strategic/competitive or collaborative. 

If we want robots and AI's integrated in society to help humanity, we need ways to train many of them at the same time. We also need ways robots can learn and derive cue's from human interaction. This direction of research has very powerful future implications which allows us to go to Mars and plant colonies using robot armies. Check out [this paper](https://arxiv.org/pdf/1707.09183.pdf) if such a problem interests you.  

## 6. Rich places we can leverage ML tools for Causal reasoning.  
1. Computers become debuggers. 
    - Debugging is the art of causal reasoning in practice.  
        - The art is at its best when debugging distributed systems. 
        - Real practical application would be a general purpose system which can figure cause of a distributed system's failure. 
2. Can AI's play CTF's. 
    - CTF's are a playground for programming ingenuity and game-like intellect. 
        - For machines to win in a CTF against humans would be the first ImageNet moment for causal reasoning scenarios. 

