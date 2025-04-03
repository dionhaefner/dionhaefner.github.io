Title: The totally reasonable effectiveness of execution-driven science
Date: 2025-04-02
Tags: Science
Author: Dion
related_posts:

This post has originally been published on [**Pasteur Labs Insights**](https://pasteurlabs.ai/insights/execution-driven-science).

---

At Pasteur Labs, we work every day to uplift simulation workflows to the machine age. Concretely, this means we turn bleeding-edge research and technology ([Simulation Intelligence](https://arxiv.org/abs/2112.03235)) into real-world capabilities:

- We implement and modify recently published AI methods to work at industrial scales with real & simulated data.
- We measure how experimental differentiable physics routines hold up in end-to-end applications (in [computational engineering](https://cdfam.com/differentiable-physics-in-digital-engineering/), for example).
- We build simulators that are data-driven, GPU-native, and [automatically differentiable](https://pasteurlabs.ai/insights/jax).
- We automate the definition, training, evaluation, and deployment of hybrid AI simulators.
- All the while, we favor robust, general workflows over one-off solutions.

This particular mix &mdash; where science meets engineering, technology enthusiasm meets sobering reality, pie-in-the-sky meets down-to-earth thinking &mdash; provides many opportunities but also unique challenges. Scientists and engineers at Pasteur Labs are constantly faced with compromise: How do we answer fundamental questions in a way that leads to measurable impact without getting lost in details that don't matter, nor providing superficial solutions that don't hold up in the real world? We believe part of the answer lies in two paradigms that are pervasive in everything we do, namely, use-inspired science and execution-driven science.

### **Use-inspired science** &mdash; the **what**. *(covered elsewhere)*

Use-inspired science encodes the fact that neither application-ignorant basic science nor singular-purpose applied science is sufficient to solve classes of fundamental, real-world problems. This is ingrained in Pasteur Labs at the deepest level (including the [name of our company](https://en.wikipedia.org/wiki/Pasteur%27s_quadrant)).

### **Execution-driven science** &mdash; the **how**. *(covered here!)*

Execution-driven science is the art of navigating complexity by scoping, iterating on, surgically modifying, and stacking existing solutions, and failing fast when encountering dead ends. In particular, execution-driven science allows us to build end-to-end systems of non-trivial complexity &mdash; emphasizing execution makes the immense challenges and combinatorial search spaces we are faced with tractable. It helps us guide our day-to-day work towards what is possible, feasible, and worthwhile.

The *how* of research is typically much less clearly defined than the *what*, although it is at least as important. Execution-driven science is a particularly powerful way to define this process when faced with the need to build systems that withstand the complexities of the real world. And in the moment, it's a valuable reminder to the individual researcher facing an overwhelming or constantly-moving research problem.

## Simulating the world

I clearly remember the moment I fell in love with physics. It was the moment I realized it made reality *computable*.

Suddenly, I looked at the world with fresh eyes, computing the time it would take for objects to fall or trains to stop or elevators to arrive. How could a simple mathematical equation like $t=\sqrt{2 s / a}$ generate predictions that we can observe in the real world?

I later realized that it wasn't so easy, and that the simplified physical laws we learn in school don't always hold up to the complexity of the real world. A second revelation came when I saw we could model complex physical systems by creating approximate solutions, solved by computers. Later still, I decided to write a [simulator](https://veros.readthedocs.io/en/latest/) to forecast ocean dynamics (see [Fig. 1](#ocean)).

<figure id="ocean">

<img src="{static}/images/execution-driven-science/ocean.png" alt="Ocean simulation" />

<figcaption>Fig. 1: A high-resolution ocean simulation, executed on a single high-end workstation about the size of an AC unit. From <a href="https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021MS002717">https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021MS002717</a>.</figcaption>
</figure>

The ocean is home to a range of complex physical phenomena, like mesoscale turbulence (the vortices or “eddies” most visible around the equator), boundary currents (like the Gulf Stream), an entire zoo of internal and surface waves, and an all-encompassing overturning circulation. Yet it is governed by only 8 so-called primitive equations.

After deriving them from first principles, we can easily write down the mathematical rules that generate the entire wealth of observed ocean dynamics. To actually simulate them, we need to conjure an unholy stack of tricks, approximations, numerical hacks, heuristics, and empirical laws, adding up to well above ten thousand lines of code.

As a grad student, seeing an ocean simulation unfold on a supercomputer in real time was sublime. For the longest time, I believed that science was inherently about simplicity: finding the most concise explanation for a phenomenon, or an elegant theory that predicts observations. Today I know there are many ways to do science. One of them is building systems &mdash; like climate models, fusion reactors, and space telescopes &mdash; via execution-driven science, bridging the gap between research lab and real world. Execution-driven science still emphasizes simplicity (or rather, [parsimony](https://en.wikipedia.org/wiki/Occam%27s_razor)), but introduces *compounding* as an additional guiding principle.

## Science that's more than the sum of its parts

Execution-driven science is about standing on the shoulders of giants and making sure they're all standing up straight. It's about taking the pieces others have built, polishing them, and stacking them in a way that pushes the boundaries of what's possible. And then adding some new pieces where they matter the most, and shatter records.

<figure id="neuralgcm">

<img src="{static}/images/execution-driven-science/neuralgcm.png" alt="NeuralGCM architecture" />

<figcaption>Fig. 2: A prime example of execution-driven science: NeuralGCM, an AI-enabled system for climate modelling that is on par with some of the best traditional climate models and orders of magnitude faster. Each of the depicted boxes is unremarkable in isolation, but in combination they create something revolutionary. From <a href="https://arxiv.org/pdf/2311.07222">https://arxiv.org/pdf/2311.07222</a>.</figcaption>
</figure>

[NeuralGCM](https://arxiv.org/pdf/2311.07222) is an AI-driven system for atmospheric simulations (the atmosphere is typically the most computationally expensive component of a climate model). Taken at face value, each of its parts seems unremarkable ([Fig. 2](#neuralgcm)): a simple encoder-decoder architecture; a differentiable dynamical core (that is, hand-written traditional simulator) in Python; a feed-forward neural network to machine-learn a data-driven correction; and an off-the-rack solver for ordinary differential equations (ODEs). But acting together, this is the first system that demonstrates that AI can aid with simulations on climatic time scales, consisting of tens of thousands to millions of iterations, at an efficiency that has never been seen before:

> *For both weather and climate, our approach offers orders of magnitude computational savings over conventional GCMs* [(Kochkov et al. 2023)](https://arxiv.org/abs/2311.07222).

The point here is *not* that all it took was to combine some existing methods and reap the benefits. What actually happened is much more subtle, where success came from a combination of different factors:

- **A rock solid understanding of the problem, to focus on what matters.** That is, using AI to speed up computations, while keeping existing knowledge around in the form of a differentiable dynamical core, iterative ODE solver, and structure of the problem. This particular combination of methods fixes many existing shortcomings of pure-AI and pure-simulator systems.

- **Pragmatic, high-quality execution.** All code is implemented in the high-performance Python framework JAX and executed on accelerators (Tensor Processing Units, TPU). This is not just computationally efficient. &mdash; keeping things entirely in Python also allows for rapid experimentation. This combination is likely close to Pareto-optimal: Yes, the system could potentially run faster if painstakingly optimized, at a huge expense in terms of human time. Or it could be much easier to develop, but not be nearly as powerful. Finding the right trade-off is precisely what it takes to iterate fast and effectively.

- **Unprecedented scale.** A resource-efficient implementation allows researchers to scale up aggressively and reach real-world scales. At its finest resolution, NeuralGCM is trained for 3 weeks on 256 TPU devices, which is way beyond the norm in AI for climate, and [AI in general](https://www.youtube.com/watch?v=BItseOa1DcM).

This is an example of execution-driven science at its best, and demonstrates several techniques on how to wield it to push the envelope. There are many more examples like it throughout science and industry, and they follow similar patterns; for example:

- [PKDGRAV3](https://arxiv.org/pdf/1609.08621), a massive-scale simulator for astrophysics that leverages GPUs and advances in numerical methods to achieve unseen efficiency.

- [PySR](https://arxiv.org/abs/2305.01582), which made symbolic regression widely accessible through a high-performance Julia backend and a Python frontend.

- [AlphaGo](https://www.nature.com/articles/nature16961), a breakthrough in reinforcement learning that pushed the boundaries of what's possible with AI, based around a relatively simple but efficient tree search algorithm.

- (Everything we build at Pasteur Labs to realize the platform for Simulation Intelligence.)


💡 Notice how some of those examples are clearly use-inspired, while others are not? This illustrates how use-inspired science and execution-driven science act along two different dimensions &mdash; you can have one without the other, or be neither use-inspired nor execution-driven.

## Explore violently, fail fast, be pragmatic

Let's take a closer look at what it takes to apply execution-driven science in practice to solve hard, real-world problems.

- **Get really good at executing stuff.** Good execution relies on top-notch technical skills, and a deep understanding of the fundamentals. *(No one said this was easy, sorry!)*

- **Build end-to-end as early as possible.** Prefer to build the first prototype that has all major parts of the final system as soon as you possibly can. Ideally, this computes data that looks like the final outputs all the way from the initial inputs. That prototype will probably perform horribly! But it allows you to see the system in its entirety, and makes it crystal clear whether you missed something important. It will also provide you a performance baseline, so you can start monitoring whether your changes make things better straight away.

- **Fail fast when things don't work out.** This is a tricky one. How do we decide that things won't work out, considering all the situations when we just need to keep at it a little longer? For me the best visualization of a good heuristic comes from the [A* algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm) for pathfinding (see Fig. 3). Rather than going down rabbit holes indefinitely, we alternate between explore and exploit, pursue strategies that work, but stay flexible and switch things up once hitting a wall. This is true to the spirit of execution-driven science: the value is in the system we build to solve problems, not sticking with any particular component, and always keeping sight on the true objective (like the A* heuristic).

<figure id="astar">

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/CgW0HPHqFE8?si=25iQM_oBx_E7mde3" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

<figcaption>Fig. 3: Illustration of the A* algorithm navigating major cities (Chicago and Rome). A healthy mix of doubling down on promising routes and switching lanes when getting stuck allows the algorithm to navigate a massive search space without falling back to random exploration nor mindless exploitation.</figcaption>
</figure>

- **Refine the things that matter, always.** To navigate complexity with combinatorial search spaces (that arise from stacking many different components) we must be selective about the battles we pick. Don't sink time into devising a clever solution to a problem that doesn't manifest. And remember, a major benefit to building end-to-end prototypes early in the process is the ability to measure actual bottlenecks right away. Make use of it!

- **Consider scaling up.** Good execution allows you to do more within the same budget (of people, compute, time, …). Spend some of your efficiency gains on more processed data, bigger models, and more powerful pipelines where it provides the largest benefit.

- **Demonstrate it works.** It's easy to get confused in complex systems. Ensure you have proper end-to-end testing in place, so that there can be no ambiguity whether problems are actually solved (and remember that it's easiest to fool oneself).

And in the end, no amount of good advice will replace your best judgment, so don't be afraid to use it.

## Science or engineering?

People ask us sometimes: is this really science, or is it engineering? We don't believe the distinction is a useful one &mdash; there's certainly a lot of both, often in synergistic ways. Regardless, well-conducted execution-driven and use-inspired science exhibits many of the core values of the scientific method:

- It aims to provide fundamental solutions to a class of (real-world) problems, rather than specific instances of them.

- It depends on a thorough understanding of the problem at hand, involved components, and resulting system, to navigate complexity and identify minimally invasive changes with maximum impact.

- It is reproducible, transparent, and extensible, allowing other researchers to build on top of results.

## Execution-driven science in the wild

At Pasteur Labs, the journey towards next-gen simulation continues &mdash; we're still far away from humanity's dream to simulate everything. Models need to become orders of magnitude faster, cheaper, more accurate, grounded in reality and causality. Today, we believe that data-driven, AI-infused methods are a central piece to compute the parts of reality that elude classical approaches, and to tackle problems that do not have a concise mathematical representation. Execution-driven science has proven to be a powerful tool in this quest, and we're excited to see where it will take us next.

And while the musings above are my earned insights, they are in great part realized because of the shared pursuits, rigor, continous support, and lived examples of top-notch use-inspired and execution-driven research from my teammates at Pasteur Labs.

Eager to help? 👉 [pasteurlabs.ai/careers](https://pasteurlabs.ai/careers)