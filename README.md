# Seq2Seq Scale AI Assignment

## About Me
Name: Samuel Sommerer

Email: `sommerer@usc.edu`

## Description
The first thing that popped into mind when trying to solve this problem was to use a transformer. 
They're the gold-standard for seq2seq tasks. Subsequently, I found an existing solution for this
problem on GitHub that also uses a transformer. You can find that repo here: `https://github.com/jaymody/seq2seq-polynomial`.
This repo already had a transformer implemented that achieved good test accuracy and was under
5 million trainable parameters, so I decided to use this repo as a starting point and forked it.

I made several modifications to the existing transformer. First, I introduced label smoothing to
when calculating cross entropy loss to combat overfitting. This ended up not being really necessary
as I wasn't training for enough epochs for overfitting to be a problem (Google Colab limited my GPU usage). Second, I manually implemented
a ReZero encoder layer for the transformer. The idea was to decrease model convergence time and
cut down on training time. I got ideas for how to implement the ReZero encoder layer here:
`https://github.com/tbachlechner/ReZero-examples`.

