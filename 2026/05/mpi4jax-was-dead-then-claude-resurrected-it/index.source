Title: mpi4jax was dead, then Claude resurrected it
Date: 2026-05-19
Tags: Thoughts, Software
Author: Dion
related_posts:

I maintain a moderately successful open source project called [`mpi4jax`](https://github.com/mpi4jax/mpi4jax) that's useful in its small niche of people abusing JAX to do scientific computing. `mpi4jax` lets you run JAX code across multiple machines by calling MPI under the hood, without leaving the compiled execution graph. It's really a bridge between JAX's "custom call" mechanism that allows you to register your own primitives and the MPI C API.

The core contribution has been a bunch of Cython code that ensures we're not calling back into Python from compiled XLA code, which is a big no-no for performance. It's been around for a few years now, and been quietly chugging along.

As I'm sure most maintainers in the JAX ecosystem know, JAX isn't exactly the most stable platform to build on, especially if you depend on non-public APIs and/or undocumented behavior (as you have to in any non-trivial project). Hell, often enough things get deprecated even *if* you depend on documented behavior. That's understandable, given how fast JAX has been evolving, but it also means that maintaining a project like `mpi4jax` can be a constant game of whack-a-mole - admittedly much more so in the past than it is now, but still. It's not uncommon for a new JAX release to break `mpi4jax` in some way, and then it's up to me to figure out how to fix it. Sometimes it's a simple fix, but other times it can be a major overhaul.

In mid-2025, things started to get really bad. JAX deprecated the custom call mechanism that `mpi4jax` relied on and replaced it with the new [FFI interface](https://jax.readthedocs.io/en/latest/ffi.html). I could still work around this by importing from (now) private APIs, but the long-term prospects were dire: to properly migrate to the new API, [I'd have to rewrite all Cython bindings in C++](https://github.com/jax-ml/jax/discussions/33405), which span several thousand lines of code. It looked like this was `mpi4jax`'s final nail in the coffin:

<figure>
<img src="/images/mpi4jax/ded.png" />
<figcaption>I guess that's it then?</figcaption>
</figure>

## Enter Claude

> ME: "Hey Claude, can you re-write these 3,000 lines of Cython code in C++ for me and generate bindings with nanobind?"
> 
> CLAUDE: "Sure thing! This will only take a second."

About 2 hours later, I had a [huge PR](https://github.com/mpi4jax/mpi4jax/pull/295) that upgraded 100% of `mpi4jax` to the new FFI interface, and in doing so also got rid of all the Cython code and replaced it with C++. To be fair, the initial code wasn't quite ready to merge. I spent a few more hours wrangling build dependencies, fixing type mismatches for MPI handles across platforms, getting the CUDA stream handling right, and coaxing CI back to green. But those are problems I know how to solve. The hard part, the actual Cython-to-C++ rewrite with correct FFI bindings for every MPI operation, was done, and done well enough. It was honestly a bit surreal.

<figure>
<img src="/images/mpi4jax/alive.png" />
<figcaption>mpi4jax is alive again.</figcaption>
</figure>

I don't think this migration would have happened without Claude. The activation energy was just too high for a mass rewrite of gnarly Cython-C++ glue code in a project I maintain on my own time.

OSS maintainers talk a lot about how AI made everything worse for them, and I'm sure their experiences are real. But there's also a class of OSS projects that weren't exactly drowning in human contributions before, that suddenly can do things that would have been completely infeasible before. Pretty awesome if you ask me!