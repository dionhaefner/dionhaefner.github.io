Title: Learning to play Yahtzee with Advantage Actor-Critic (A2C)
Slug: yahtzotron-learning-to-play-yahtzee-with-advantage-actor-critic
Date: 01-04-2021
Tags: Machine Learning, Reinforcement Learning, Python
Author: Dion
related_posts:

My in-laws are really into the dice game [Yatzy](https://en.wikipedia.org/wiki/Yatzy) (the Scandinavian version of Yahtzee).

If you're unfamiliar with the game, here's a brief summary of the rules [from Wikipedia](https://en.wikipedia.org/wiki/Yatzy#Gameplay):

> Players take turns rolling five dice. After each roll, the player chooses which dice to keep, and which to reroll. A player may reroll some or all of the dice up to two times on a turn. The player must put a score or zero into a score box each turn. The game ends when all score boxes are used. The player with the highest total score wins the game.

Sounds easy enough, right?

My in-laws, who have much more experience than me, made very quick decisions, but I couldn't see if they were really better than mine. Is it better to give up on getting a Yahtzee early on, or should you delay that for as long as possible? Is going for straights even worth it? How important is the bonus, really?

<figure>
    <img src="{static}/images/yahtzotron/Dice-2.svg" style="width: 2em">
    <img src="{static}/images/yahtzotron/Dice-3.svg" style="width: 2em">
    <img src="{static}/images/yahtzotron/Dice-4.svg" style="width: 2em">
    <img src="{static}/images/yahtzotron/Dice-6.svg" style="width: 2em">
    <img src="{static}/images/yahtzotron/Dice-6.svg" style="width: 2em">
    <figcaption>What to go for? A straight? Or keep the sixes? The right answer isn't obvious, and depends on both your and your opponent's scorecard.</figcaption>
</figure>

While playing (and losing), I could never shake the feeling that I had no idea *whether my strategy was good or not*. Yahtzee is luck-based to a large degree, so it's hard to judge whether you suck at the game or whether you're just unlucky.

So, I finally thought to myself:

> This sounds like a game that should be easy to learn for a bot. Maybe it can teach me how to play!

Specifically, I wanted to **build a bot that could learn to play Yahtzee close to perfection through self-play** (via reinforcement learning, RL). Turns out, I was wrong about the *easy* part, but a few weeks of intensive labor later I was done with my creation.

<figure class="lesson">
<figcaption>Lessons</figcaption>
<p>Throughout this article, you will find some of the more salient lessons I learned in boxes like this one.</p>
</figure>

[TOC]

# The Making of Yahtzotron

<figure>
    <img src="{static}/images/yahtzotron/sass.png" style="width: 100%; max-width: 400px;">
    <figcaption>The nerve on this guy.</figcaption>
</figure>

Some spoilers first:

Yahtzotron's average final score is about 5% below perfect play, which is definitely competitive with experienced human players. Training time of the final agent is about 2 hours on a single CPU. [And it's available on GitHub, for you to try!](https://github.com/dionhaefner/yahtzotron)

However, for a long time, it seemed like I bit off more than I could chew. As a first reinforcement learning project this was definitely a challenge. So, let me present to you the long and winding path towards a strong reinforcement learning agent, so that others may learn from my hubris.

If you want to skip ahead to the fun part, you can [watch me play a full game at the end of this article.](#the-results-are-in)

## Why Reinforcement Learning?

[Yahtzee has a known solution for perfect play](http://yahtzee.org.uk/optimal_yahtzee_TV.pdf) (and so does Yatzy). So, why go through all of this to learn a "solved" game?

I believe that a strong self-taught agent is still valuable, even if there *is* a known solution to the game.

Perhaps the biggest factor is efficiency. Exact solutions to probabilistic games have to search through all possible outcomes to identify the best action. This is computationally costly (without optimization, this scales exponentially with the number of game steps). Common optimization strategies such as [dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming) are difficult to implement correctly, and efficient implementations need to be tailored to the problem at hand.

A second factor is flexibility. An agent that learns through self-play is robust to minor rule changes (this is why we can learn Yahtzee and Yatzy with the same agent). We can also *change the objective*. Most exact solutions to Yahtzee optimize the average game score, but I personally tend to *play to win* (a good winning agent might have to take more risks when it is behind, and play it safe when it's ahead).

And finally, it can help us understand how to build efficient RL agents, so we can eventually tackle problems that do not have a known solution.

## RL Frameworks -- The JAX + DeepMind Stack

I wanted to use this opportunity to learn a new framework, so I decided to implement the training loop in [JAX](https://github.com/google/jax).

Unlike Pytorch and Tensorflow, JAX doesn't supply a high-level interface for machine learning (instead, it relies on third-party libraries for that). I settled on the DeepMind stack: [Haiku](https://github.com/deepmind/dm-haiku) for neural networks, [optax](https://github.com/deepmind/optax) for optimization, [rlax](https://github.com/deepmind/rlax) for reinforcement learning components.

Overall, the experience was pleasant, but not without obstacles. I had to file several bug reports while working on Yahtzotron. For more serious projects I would probably just go with [Pytorch](https://pytorch.org/) right now.

<figure class="lesson">
<figcaption>Lesson 1</figcaption>
<p>Only go with the JAX ecosystem if you are prepared to implement most of the logic yourself. Also, be ready to work around bugs or performance issues.</p>
<p><i>(JAX is evolving fast. This advice is probably outdated soon, so make sure to give JAX a chance. Most of it is awesome already.)</i></p>
</figure>

## Implementing Yahtzee / Yatzy

I started with coding up the rules for Yahtzee and Yatzy in pure Python.

For this, I introduced 2 classes: `Ruleset` and `Scorecard`. `Ruleset` encapsulates the core rules of the game: Which categories they are, how they are scored, and what bonuses there are (see e.g. [the ruleset for Yatzy](https://github.com/dionhaefner/yahtzotron/blob/master/yahtzotron/rulesets/yatzy.py)). `Scorecard` uses a `Ruleset` internally to keep track of filled categories and scores for each player.

In hindsight, I do like this extensible approach. However, I made one crucial mistake, which was not to wrap the game in [OpenAI's gym](https://gym.openai.com/). I didn't see how I could make it work with the changing action space, so I didn't bother because I thought I didn't need it.

I later realized that not having the game implemented in `gym` was a huge disadvantage. It made it so I couldn't test my agent on other, simpler problems while debugging. And it also made it that I couldn't switch out the A2C agent for a different one (such as [PPO](https://openai.com/blog/openai-baselines-ppo/)) when I would have liked to.

<figure class="lesson">
<figcaption>Lesson 2</figcaption>
<p>Implement your problem in <a href="https://gym.openai.com/">gym</a>. You might think you don't need to, but you probably will.</p>
</figure>


## Detour: Genetic Optimization

I was intrigued to try [genetic optimization](https://en.wikipedia.org/wiki/Genetic_algorithm) as a baseline. [This article gives an overview how genetic optimization can work in RL contexts.](https://towardsdatascience.com/reinforcement-learning-without-gradients-evolving-agents-using-genetic-algorithms-8685817d84f) The idea is deceptively simple:

1. Initialize a league of agents with random weights.
2. Let them play each other repeatedly.
3. Compute a fitness based on how well each agent did.
4. Populate a new league by drawing agents at random, weighted with their respective fitness. (You could also do *sexual* procreation by drawing 2 agents and combining their weights every time, but it's not really necessary.)
5. Mutate the weights of each agent by a small random number (e.g. drawn from a Gaussian with zero mean and small variance).
6. Repeat for as long as necessary.

<figure>
<img src="{static}/images/yahtzotron/genetic.svg" style="width: 100%; max-width: 350px;">
<figcaption>Genetic optimization in a nutshell.</figcaption>
</figure>

As expected, genetic optimization was easy to implement. All we need is an agent that can play games - no loss functions, no optimizers. You can treat your agent as a black box that you just evaluate based on its fitness.

**But this simplicity comes as a cost**. In my tests, the league advanced quickly at first, up to a mean score of about 130. But by then, progress had slowed down so much that it seemed stuck. (A decent game score is 200+.)

It is important to remember that real-life evolution needs one key ingredient to work: *time*. Unfortunately, we don't have millions of years on our hands to wait for the perfect Yahtzee machine to evolve. So let's get back to something more intelligent.

<figure class="lesson">
<figcaption>Lesson 3</figcaption>
<p>Genetic optimization is simple to implement and can give you a decent baseline performance with little effort, but don't expect super-human agents to come out of this.</p>
</figure>

## Advantage Actor-Critic

After the detour to genetic optimization, it is time to return to reinforcement learning. I decided to try the A2C (advantage actor-critic) algorithm next.

If you are not familiar with A2C, [here is an amazing introduction in the form of a cartoon.](https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752)

The basic idea is pretty straightforward. The agent consists of 2 parts, the actor and the critic. Both receive the current state of the game as input.

- The **critic** predicts the *value* of the current state in the form of what it thinks the total reward will be at the end of the game (in our case, the final score of the agent + an optional winning bonus).

    The critic's loss is essentially the accuracy of those predictions, evaluated through a mechanism called temporal difference learning or TD-λ. Temporal differencing accounts for the fact that the near future is safer to predict than the far future, and that rewards now are better than equal rewards later.

- The **actor** predicts a probability with which each action should be taken, according to the current *policy*.

    Its loss is based on something called the *advantage*: If the picked action was better than expected (based on what the critic estimated for it), the actor should take it more often, and its probability increases (and vice-versa for an action that was worse than predicted).

The total loss of the agent is then the summed loss of both actor and critic, plus an additional entropy loss that makes sure that the model keeps exploring different options.

For the implementation in JAX / Haiku, I used [bsuite's A2C agent](https://github.com/deepmind/bsuite/blob/a07485f497b72669f1058639fa806b6127c4c6a9/bsuite/baselines/jax/actor_critic/agent.py) as a template.
One nice thing about JAX is how readable and "non-magical" the loss function looks:

```python
def loss(
        weights,
        observations,
        actions,
        rewards,
        td_lambda=0.2,
        discount=0.99,
        policy_cost=0.25,
        entropy_cost=1e-3,
    ):
        """Actor-critic loss."""
        logits, values = network(weights, observations)
        values = jnp.append(values, jnp.sum(rewards))

        # replace -inf values by tiny finite value
        logits = jnp.maximum(logits, MINIMUM_LOGIT)

        td_errors = rlax.td_lambda(
            v_tm1=values[:-1],
            r_t=rewards,
            discount_t=jnp.full_like(rewards, discount),
            v_t=values[1:],
            lambda_=td_lambda,
        )
        critic_loss = jnp.mean(td_errors ** 2)

        if type_ == "a2c":
            actor_loss = rlax.policy_gradient_loss(
                logits_t=logits,
                a_t=actions,
                adv_t=td_errors,
                w_t=jnp.ones(td_errors.shape[0]),
            )
        elif type_ == "supervised":
            actor_loss = jnp.mean(cross_entropy(logits, actions))

        entropy_loss = -jnp.mean(entropy(logits))

        return policy_cost * actor_loss, critic_loss, entropy_cost * entropy_loss
```

*(These are actually 2 losses in 1, either A2C or a supervised loss for the actor. This becomes relevant during pre-training, see next section.)*

The only real struggle was to figure out how to handle action space constraints. In Yahtzee, the first 2 actions of the turn are keep actions (which dice should be kept for the next roll). The last action is a category action (which score category we should use for the roll). Both have a different number of possible actions. How do we encode this with a single output?

I decided to bake this into the network architecture, and replace the predicted logits with `-inf` if the action was invalid. By doing this however I opened Pandora's Box, because `NaN` values started to pop up everywhere (where JAX functions couldn't handle infinities). To work around this, in the loss function, I replace `-inf` with the smallest possible float instead.

*(I have since read that people usually just give a negative reward to impossible actions and let the model learn the rules by itself. Perhaps I should have done that instead.)*

**Finally, after implementing A2C, I had it all laid out.** Here is how Yahtzotron plays a turn:

<figure>
<img src="{static}/images/yahtzotron/yzt-flowchart.svg" style="width: 100%;">
<figcaption>How Yahtzotron plays a turn. The agent uses its value output to determine the value of the strongest opponent, which is used as an input later on. Then, it uses its policy output to select actions, which finally lead to a turn score (reward).</figcaption>
</figure>

If you look closely, you will find some more features that I haven't mentioned yet: Once, at the start of the turn, Yahtzotron uses its value output (from the *critic*, see above) to predict the value of the currently strongest opponent. This is used as an input to all decisions if Yahtzotron is playing to *win* (as opposed to maximizing expected score). Another thing I haven't mentioned yet is the strategy output, which we will [return to later](#human-learning-machine-teaching).

Unfortunately, there's no way to sugarcoat it: **initial performance of the agent was terrible.** It seemed to be unable to learn anything more sophisticated than super greedy, semi-random play with a mean score of about 100.

In the following sections, I will describe how I managed to convince the agent to go above this local maximum.

<figure class="lesson">
<figcaption>Lesson 4</figcaption>
<p>A more complicated model could mean that you need to try harder to make it work (with potentially greater reward).</p>
</figure>

## Pre-training via Greedy Look-up Table

To help the agent learn a better strategy, I decided to pre-train it on a simple baseline policy via supervised learning.

As the baseline policy I first used a naive, greedy strategy: Pick the action that yields the maximum expected score across all categories after the next roll. This can be pre-computed in a look-up table within a few seconds (there are only 252 distinct roll combinations for a single roll).

Unfortunately, this proved to be a very weak baseline. Being *this* greedy is highly suboptimal in Yahtzee, because you end up filling valuable categories early on that you might need as a buffer later (like Chance). With a mean score of around 120, this was able to make the agent somewhat better, but nowhere near optimal play.

## Pre-training via Advantage Look-up Table

Next, I thought about how I as a human approach the game. It occurred to me that human players don't think in discrete "keep" actions. Rather, they decide for a *category* to go for, and then pick the keep action that they think maximizes the score for this category.

But how to pick the optimal category? As a human I'm playing mostly opportunistic. If my current roll looks like it might become a better-than-average result for a category, I go for it.

For example, rolling `1 6 6 6 6` is certainly better than average for the "sixes" category, worse than average for the "ones" category, and much better than average for the "Yahtzee" category. I would try to roll a Yahtzee here (and keep the sixes).

So, this is the quantity that I use to pick the best category for this agent: The maximum expected score of each category given the current roll, minus the average score for this category across all rolls (I call that quantity advantage, as in A2C learning). We can pre-compute this with the same look-up table as in the greedy case - we just need to also compute the expected score for each category across all rolls.

This baseline is much much stronger, with an average score of about 220 for Yahtzee und 200 for Yatzy. This is a great baseline to lift our agents to the next level.

<figure class="lesson">
<figcaption>Lesson 5</figcaption>
<p>Think about how you as a human approach the game. Simple heuristics often make for a strong baseline that you can use to pre-train your model.</p>
</figure>

## Parameter Tuning

What followed next was lots. of. parameter. tuning.

RL agents have a large number of hyperparameters:

- network architecture (e.g. number of layers and neurons);
- learning rate & number of epochs;
- reward discount & TD-λ;
- reward norm & winning reward;
- relative weight of policy, value, entropy loss terms;
- batch size (here: number of players per game).

*(possibly repeated for multiple learning stages)*

I found that parameter tuning was even more important for this RL application than what I'm used to from "regular" Deep Learning.

Especially λ (as in TD-λ) had a huge influence on performance. λ can range from 0 to 1, where 0 implies maximum greed (only care about the reward of the next action), and 1 maximum patience (rewards later are as good as rewards now). I first started with a high λ of around 0.9 because I thought that patience was the right strategy, but it also made it harder for my agent to learn the right patterns.

Ultimately, I found it best to start out with a low λ (0.2), which I gradually increase during training to 0.8. This lets the agent learn simple greedy patterns first before strategizing more about the long run.

To give you an idea of the complexity of this tuning process, here is the function responsible for varying some of the hyperparameters during training:

```python
def get_default_schedules(pretraining=False):
    """Get schedules for learning rate, entropy, TDlambda."""
    if pretraining:
        return dict(
            learning_rate=optax.constant_schedule(5e-3),
            entropy=optax.constant_schedule(1e-3),
            td_lambda=optax.constant_schedule(0.2),
        )

    return dict(
        learning_rate=optax.exponential_decay(1e-3, 60_000, decay_rate=0.2),
        entropy=(
            lambda count: 1e-3 * 0.1 ** (count / 80_000) if count < 80_000 else -1e-2
        ),
        td_lambda=optax.polynomial_schedule(0.2, 0.8, power=1, transition_steps=60_000),
    )
```

<figure class="lesson">
<figcaption>Lesson 6</figcaption>
<p>When doing RL, don't write your model off before doing at least some parameter tuning. In particular, try varying your time differencing parameter early on (λ).</p>
</figure>


## Other Stuff I Did That Didn't End Up Working

**Predict only categories**: I thought it would be more human-like to predict a category to go for and just take the keep actions that maximize expected score for that category (similar to the baseline agent). This did lead to somewhat faster training and the same final performance, but ultimately I knew that perfect play would be impossible with this, so I removed it.

*Don't dumb down your environment, your agent will learn to deal with complication*.

**Deterministic rolls**: I figured it could help to give all agents in a game the same dice rolls, so they could explore different options without luck picking the winner. Nope. Made things worse.

*Don't remove randomness if it's an integral part of your task.*

**Tailored neural network architectures**: Instead of a "dumb" single-head feed-forward network I tried other architectures that I thought would be more fitting for the structure of the game. For example, I put a layer with only 5 neurons (because there are 5 dice) before the keep actions output layer. There was no positive performance impact.

*No need to tinker too much with the architecture, an MLP will learn just fine.*

**JAXify everything**: I wrote (almost) the whole turn logic in JAX, only to find that it was much slower than leaving it in NumPy. This is because games are played sequentially, so all arrays just contain a few dozen elements, which is not enough to amortize the overhead of using JAX.

*Stick to NumPy for small array operations.*

<figure class="lesson">
<figcaption>Lesson 7</figcaption>
<p>Sometimes, less is more.</p>
</figure>


# The Results Are In

## Showmatch Time!

Here, you can watch me play (and lose to) Yahtzotron.

<figure>
  <script id="asciicast-kXQNIhZ0LlC9Mn11ZsHlPjrvH" src="https://asciinema.org/a/kXQNIhZ0LlC9Mn11ZsHlPjrvH.js" async data-cols="64" data-rows="32" data-theme="monokai"></script>
  <figcaption>Yes, it rolled a Yatzy in the first round.</figcaption>
</figure>

If you're interested in playing against Yahtzotron yourself, [here's the code and the instructions](https://github.com/dionhaefner/yahtzotron).

## The Numbers

Let's have a look at how the final agents perform.

First up, we'll have a four-way tournament across 10 000 games. This way, we can see what the average final score is. We will also test whether an agent trained with a winning bonus is better at winning than agents without it.

```text
$ yahtzotron evaluate pretrained/yahtzee-score.pkl pretrained/yahtzee-score.pkl pretrained/yahtzee-score.pkl pretrained/yahtzee-win.pkl -n 10000 --ruleset yahtzee

100%|███████████████████████| 10000/10000 [09:34<00:00, 17.42it/s]
Agent #1 (pretrained/yahtzee-score.pkl)
---------------------------------------
 Rank 1 | ██████ 2515
 Rank 2 | ██████ 2550
 Rank 3 | ██████ 2510
 Rank 4 | █████ 2425
 ---
 Final score: 236.2 ± 59.2

... (agent 2 and 3 similar to agent 1)

Agent #4 (pretrained/yahtzee-win.pkl)
-------------------------------------
 Rank 1 | ██████ 2581
 Rank 2 | █████ 2446
 Rank 3 | █████ 2463
 Rank 4 | ██████ 2510
 ---
 Final score: 235.8 ± 59.4
```

*(ties are counted as the higher rank for both agents)*

As it turns out, we achieve mean scores of arond 236, and the play-to-win agent is indeed winning more often despite having a slightly lower mean score! (Of course, it is also coming last more often - riskier plays don't always pay off.)

Needless to say, the trained agent also consistently beats the [greedy](#pre-training-via-advantage-look-up-table) and random baseline agents.

```text
Agent #1 (pretrained/yahtzee-score.pkl)
---------------------------------------
 Rank 1 | █████████████ 616
 Rank 2 | ████████ 384
 Rank 3 |  0
 ---
 Final score: 239.7 ± 62.2

Agent #2 (greedy)
-----------------
 Rank 1 | ████████ 390
 Rank 2 | █████████████ 610
 Rank 3 |  0
 ---
 Final score: 218.7 ± 52.9

Agent #3 (random)
-----------------
 Rank 1 |  0
 Rank 2 |  0
 Rank 3 | ████████████████████ 1000
 ---
 Final score: 44.2 ± 17.7
```

Still, 236 is not an optimal score (perfect play is around 254). The situation is better for Yatzy, where we get a mean score of 241 (perfect play is around 248).

I suspect that we see this gap because Yahtzee has some "weird" rules when rolling multiple Yahtzees in a game. More than, say, 3 Yahtzees should occur very rarely, so the agent has a hard time learning what to do. On the other hand, these unicorn games can lead to some very high final scores, thus having a (relatively) big impact on the mean score.

For the typical game, these agents shold be reasonably close to optimal.


# Final Thoughts

## Why Was This So Hard?

I'd like to take some time to reflect *why* Yahtzee is such a challenging problem for RL.

Achieving perfect play in Yahtzee is challenging (impossible?) for *humans* because it requires perfectly calibrated probabilities. You will need a flawless mental model to evaluate the risk that is associated with each action. This is hard because it is more quantitative than our intuition can handle.

On the other hand, machines are extremely quantitative, so learning quasi-perfectly calibrated probabilities should not be a problem. Here, the problems I've observed are two-fold:

1. Getting stuck in a local optimum. Easy to learn strategies like greedy play are too hard to beat by incremental improvements.

2. Missing obvious best plays -- obvious to humans, that is -- because the situations when they are needed are too rare.

    One example are "hail mary" plays where the agent is hopelessly behind, and can only hope to win by gambling on getting a Yahtzee. Most of the time it won't work, so the agent won't learn that it is actually advantageous.

    I'm not sure how a RL agent can solve situations like these reliably. An obvious solution are Monte-Carlo tree search methods that actually play out the consequences of each decision, but then the learning process would be dependend on the game rules again -- something I wanted to avoid.

Anyhow, it took me by surprise how fast this little side project essentially turned into a research problem. Reinforcement learning is hard!

## Human learning -- Machine Teaching

Remember the introduction, when I said that I wanted Yahtzotron to teach *me* to get better at the game? So far, we haven't really done anything in this direction.

Creating an agent that is good at a task is one thing. Another -- and, in my opinion, much more valuable -- thing is to *transfer that knowledge back to us humans*. This is what I call **human learning**.

Achieving this is incredibly hard, and I don't think there is a universal recipe for this yet.

The way I approached this with Yahtzotron is to enable the agent to *"think out loud"*. For this, I trained another neural network that predicts the final (category) action taken after the first and second roll. I call this the *strategy network*.

The strategy network gives you its best guess what Yahtzotron might be going for when picking dice to roll. Usually, this is quite convincing:

<blockquote>
> My turn!<br>
> Roll #1: [3, 3, 3, 5, 6].<br>
> I think I should go for Threes, so I'm keeping [3, 3, 3].<br>
> Roll #2: [3, 3, 3, 3, 4].<br>
> I think I should go for Threes or Yatzy, so I'm keeping [3, 3, 3, 3].<br>
> Roll #3: [1, 3, 3, 3, 3].<br>
> I'll pick the "Threes" category for that.
</blockquote>

The lines starting with "I think I should go for..." are based on the output of the strategy network.

*(There is still a lot of luck involved, so the prediction isn't always right.)*

With this in place, it becomes somewhat more transparent how Yahtzotron is making decisions. Of course, there's still a long way to go towards a real *machine teacher*.

---

I hope you have enjoyed this read.

If you want to give Yahtzotron a try, [just visit the repository](https://github.com/dionhaefner/yahtzotron) and follow the instructions.

Good luck! 🎲🎲🎲🎲🎲

<!-- article end -->

<style>
    .lesson {
        border: 1px dashed #aaa;
        border-radius: 10px;
        padding: 5px;
    }

    .lesson figcaption {
        font-size: 100% !important;
        font-weight: 600;
        font-style: normal !important;
    }
</style>
