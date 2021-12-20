Title: How we found the causes of rogue waves from data
Date: 2021-07-17
Tags: Science, Machine Learning, Geophysics, Python
Author: Dion
related_posts:
Status: draft

<figure>
    <img src="{static}/images/rogue-waves/hokusai.jpg" style="max-width: 250px; margin: 0;">
    <img src="{static}/images/rogue-waves/rogue.jpg" style="max-width: 250px; margin: 0;">
    <figcaption>Recreation of Hokusai's "Great wave off Kanagawa" in a wave tank. From <a href="https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/laboratory-recreation-of-the-draupner-wave-and-the-role-of-breaking-in-crossing-seas/65EA3294DAFD97A50C8046140B45F759">McAllister et al., 2018</a>.</figcaption>
</figure>

*Most of what I describe in this blog post is published in the peer-reviewed articles [Real-world rogue wave probabilities](https://www.nature.com/articles/s41598-021-89359-1) and [FOWD: A Free Ocean Wave Dataset for Data Mining and Machine Learning](https://journals.ametsoc.org/view/journals/atot/aop/JTECH-D-20-0185.1/JTECH-D-20-0185.1.xml).*

## Rogue waves

Imagine this:

> You are standing on the main deck of a ship, 4m above the water, on a voyage across the Atlantic. The sea is pretty calm, the waves are long and regular, and seem to be gently stroking the hull of the ship, well below your vantage point. But then you need to hop backwards. Your wet feet and the retreating water on the deck are the only sign left of the wave that just washed over the railing, and disappeared as fast as it came.

Or this:

> Your ship is navigating through a storm, and the sea is rough - waves are piling up, reaching heights of 8m and more. You are on the bridge, the highest point of the ship, 20m above the water. After every crest it feels like the vessel is dropping into a hole in the sea. Suddenly, you feel weightless for far too long. As the wall of water ahead smashes the bridge windows, you are enveloped by the cold sea.

What do both events have in common? They are examples of encounters with *rogue waves*.

A rogue wave is, by definition, any wave that is at least twice as high as the significant wave height (which is roughly the average wave height a trained human observer would report). This is what I set out to study for my PhD at the University of Copenhagen.

When I began to get into the research literature about them, I was stunned: Every study seemed to focus on something different! 

Now, a couple of years down the road, it seems like we have succeeded to shine some light on some of these processes.
**So, in this blog post, I'll walk you through how we used machine learning to find the physics behind rogue waves.**

Basically, the goal is that you will understand this poster of mine:

<figure>
    <a href="{static}/images/rogue-waves/egu-rogue-wave-poster.png">
        <img src="{static}/images/rogue-waves/egu-rogue-wave-poster-thumbnail.png">
    </a>
    <figcaption>Poster <a href="https://meetingorganizer.copernicus.org/EGU21/EGU21-1189.html">from a presentation I held at the General Assembly of the European Geophysical Union 2021<a>. (click for larger version)</figcaption>
</figure>

*This post is mostly about statistics and machine learning, so I won't get into too much detail about the rogue wave phenomenon. If you want to learn more about that, this description by [NOAA](https://oceanservice.noaa.gov/facts/roguewaves.html) is a good start, or this article in [Quanta magazine](https://www.quantamagazine.org/the-grand-unified-theory-of-rogue-waves-20200205/) that is really well written. And the [Rogue wave Wikipedia article](https://en.wikipedia.org/wiki/Rogue_wave) goes a bit more in depth on the various theories surrounding rogue wave formation.*

## Big data and little data

Rogue waves are rare, but not as rare as you might think. About 1 in 10,000 waves is a rogue wave, which means that we find about 1 of them per day at any given location. (This might sound scary, but keep in mind that a 2m wave in a 1m sea is also counted as a rogue wave, even though it isn't dangerous.)

Extreme events like rogue waves are *big data* and *little data* problems at the same time. You need to collect a lot of raw data, but you will usually have only few interesting events to work with. This means that you will need to use methods that scale well *and* are robust to sample size effects -- we will return to this in a bit.

First up, we need data. Lots of it.

<figure>
    <img src="{static}/images/rogue-waves/cdip.png" style="max-width: 400px;">
    <figcaption>Locations of the 150+ CDIP measurement stations. Screenshot from <a href="https://cdip.ucsd.edu/m/deployment/station_view/">the CDIP station view</a>.</figcaption>
</figure>

We decided to scrape the entire [CDIP buoy database](https://cdip.ucsd.edu), which contains data from over 150 locations across the US and its overseas territories.  his is hundreds of Gigabytes of data -- in total, it contains over 700 years of time series, sampled every 0.78 seconds. It is just based on the up and down motion of the buoys on the waves, and looks something like this:

<figure>
    <img src="{static}/images/rogue-waves/timeseries.png" style="max-width: 250px;">
    <figcaption>An example timeseries. Shaded region indicates significant wave height.</figcaption>
</figure>

Now it is straightforward to just go through every wave in this dataset and record its height. We can also compute the significant wave height, and find all rogue waves in the dataset. But that is only half of the story: now we know which waves were rogue waves, but how do we identify patterns in this data?

[Our first paper](https://journals.ametsoc.org/view/journals/atot/aop/JTECH-D-20-0185.1/JTECH-D-20-0185.1.xml) is about how we can transform these time series measurements into something that allows us to characterize the sea state. We find over 4 billion waves in total, and compute dozens of characteristic sea state parameters that describe the conditions in which each wave occurred (over a Terabyte of data). So now that we have that, we can proceed with the analysis.

Now, remember how I said that we need to account for sample size effects? Why would that be the case when we have billions of data points?

Well, it's complicated. For one, rogue waves are rare, so we only have about 100,000 *positive* events (where a rogue wave occurred). That is a lot less than 1 billion (but luckily we can also learn something from the cases when a rogue wave did *not* occur). Secondly, we are interested in how the rogue wave risk varies with various parameter, so we will have to look at conditional probabilities. If we don't test for significance, how are we going to know whether we have enough data in a physical regime to make a statement? The next section describes how we approached this problem.


## Real-world rogue wave probabilities

Now, the goal is to learn how the probability of encountering a rogue wave ($p$) depends on the sea state ($X$). But we don't have access to $p$, only to observations $y$ which can be 1 (rogue wave) or 0 (no rogue wave). 

The requirements for our machine learning algorithm are:

1. Must be able to handle a 10,000 to 1 class imbalance in non-separable data.
2. It should output well-calibrated probabilities, to make sure that we stay in a physically meaningful regime. For example, a prediction of $p = 0.5$ would be nonsensical, because rogue waves are outliers by definition -- every second wave being a rogue is impossible.
3. It has to deliver some sort of uncertainty estimate so we know whether we have enough data.
4. The goal isn't really to predict rogue waves, but to learn *why* they occur, so we need something interpretable.
5. Has to scale to billions of data points.

To make things worse, we have no idea how $p$ depends on $X$ -- is it a linear dependency? Are there jumps? Is there any dependence at all?

This is a tough list, and we couldn't really find anything that ticks all boxes, so we came up with our own method. The basic idea is to split every parameter into a number of bins $x_i$, and assume that $p$ is constant within each bin (this is basically a glorified histogram, and takes care of the fact that we don't have a functional form for $p$). 

Then, we find $p(x_i)$ with some Bayesian data analysis. Assuming that the number of rogue wave measurements follows a binomial distribution, we find that $p$ is Beta distributed:

$$p(x_i) \sim \text{Beta}(n^+ + \alpha_0, n^- + \beta_0)$$

where $n^+$ is the number of rogue measurements within $x_i$, $n^-$ the number of non-rogue measurements, and $\alpha_0$ and $\beta_0$ represent a prior that ensures that we stay within the right probability range.

<figure>
    <img src="{static}/images/rogue-waves/pred-power-all-merged.png">
    <figcaption></figcaption>
</figure>

## The hunt for the white whale

So far, our analysis only looked at 1 parameter at a time, so we still don't know how high the maximum rogue wave risk gets. What if all parameters can align with each other to create "hot corners" of feature space where we find significantly more rogue waves than so far?

Finding these regions in 12 dimensions is hard. In high-dimensional settings, histograms are useless, because the data is spread out too much. 

The basic idea is something like this:

<figure style="max-width: 100%;">
    <img src="{static}/images/rogue-waves/clustering.png" style="max-width: 400px;">
    <figcaption></figcaption>
</figure>

We take the raw measurements and fit a random forest classifier on it with a fixed number of samples per leaf (say, 100,000, i.e., about 10 rogue waves on average). 

<figure style="max-width: 100%;">
    <img src="{static}/images/rogue-waves/cluster-kde-all.png">
    <figcaption></figcaption>
</figure>


## From association to causation

We now have good evidence that crest-trough correlation is a very important predictive parameter for rogue waves. But that doesn't really tell us anything about the *causes* of rogue waves yet. How are we supposed to know whether a high crest-trough correlation is a direct cause of rogue waves, or whether we are looking at confounding? In fact, many of the parameters are correlated to each other:

<figure>
    <img src="{static}/images/rogue-waves/correlation-matrix.png" style="max-width: 400px;">
    <figcaption>Correlation matrix of sea state parameters</figcaption>
</figure>



## Outlook

Now you should have understood the main message of our paper -- that real-world rogue waves do seem to be caused mainly by linear superposition, and that crest-trough correlation is the best parameter to predict them.

There are some other things we studied in our paper, like the role of kurtosis (a measure for extremeness of a sea state), and how things are looking for rogue *crests*, but that would be too much to go into here.