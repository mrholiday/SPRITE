# SPRITE #

Major refactoring of Michael Paul's SPRITE code.  Supports multi-threaded
training and easier implementation of new models.

## Dependencies ##

+ [JCommander](http://jcommander.org/#Download "JCommander") (included at
the moment)
+ Tested with JDK 1.7
+ Python 2.7 (for printing topwords and learned parameters)

## Building ##

    find . -iname *.java > source_files.txt
    javac -cp ./lib/jcommander-1.49-SNAPSHOT.jar:. @source_files.txt

## Training ##

### Sample Usage ###
    
    # Trains a 20 topic LDA model on ACL abstracts
    OUTPUT_DIR=${SPRITE_HOME}/LDA_output
    mkdir ${OUTPUT_DIR}
    cd ${SPRITE_HOME}/src/
    java -cp ../lib/jcommander-1.49-SNAPSHOT.jar:. models/factored/impl/SpriteLDA -Z 20 -nthreads 2 -step 0.01 -iters 2000 -samples 100 -input ../resources/test_data/input.acl.txt -outDir ${OUTPUT_DIR} -logPath ${OUTPUT_DIR}/acl.lda.log

In the current implementations, 200 iterations of Gibbs sampling are
performed as a burn-in period, then we alternate between one step of
Gibbs sampling and one gradient udpate step until *-iters* iterations expire.

To see the command line arguments for a given model, run the model
implementation with the *--help* flag. For example, here are the arguments
for the *SpriteSupertopicAndPerspective* model:

+ *-Z*: Number of topics. Default: 0
+ *-C*: Number of components for supertopic factors. Default: 5
+ *-numFactors*: How many supertopic factors to build our model with. Default: -1
+ *-input*: Path to training file.
+ *-logPath*: Where to log messages.  Defaults to stdout.
+ *-outDir*: Where to write output.  Defaults to directory where training file is.
+ *-iters*: Number of iterations for training total. Default: 5000
+ *-samples*: Number of samples to take for the final estimate. Default: 100
+ *-step*: Master step size (AdaGrad numerator). Default: 0.01
+ *-likelihoodFreq*: How often to print out likelihood/perplexity.  Setting less than 1 will disable printing. Default: 100
+ *-deltaB*: Initial value for delta bias (on \widetilde{\theta}). Default: -2.0
+ *-omegaB*: Initial value for omega bias (on \widetilde{\phi}). Default: -4.0
+ *-sigmaAlpha*: Stddev for alpha. Default: 1.0
+ *-sigmaBeta*: Stddev for beta. Default: 10.0
+ *-sigmaDelta*: Stddev for delta. Default: 1.0
+ *-sigmaDeltaB*: Stddev for delta bias. Default: 1.0
+ *-sigmaOmega*: Stddev for omega. Default: 10.0
+ *-sigmaOmegaB*: Stddev for omega bias. Default: 10.0
+ *-seed*: Seed for pseudorandom number generator.  Value of -1 will use clock time. Default: -1
+ *-nthreads*: Number of threads that will sample and take gradient steps in parallel. Default: 1


### Input Format ###

Document ID, observed factor scores, and text in each view are each
separated by tabs.  Tokens within each view and component scores for an
observed factor are separated by a single space.  See sample input files
under *resources/test_data* for examples.

+ *input.acl.txt*: ACL abstracts -- no observed factors
+ *input.debates.txt*: US congress floor debate transcripts -- single component observed factor, conservative (positive) -> liberal (negative)
+ *input.ratemd.txt*: doctor reviews from ratemd.com -- single component observed factor, average doctor review over multiple aspects

## Modeling ##

SPRITE is a framework flexible enough to encompass many common classes
of topic models.  To implement your own models, take a look at the
implementations under *src/models/factored/impl*  For information on the
meaning of each hyperparameter refer to the SPRITE paper (citation at
bottom).

There are three objects you need to worry about when building a new model:

### SpriteThetaPrior ###

This object keeps track of the prior for per-document topic distributions.
The constructor arguments determine which factors influence the topic
distribution prior and how hyperparameters are updated during training.
Arguments:

+ *factors*: Array of Factors that influence the document->topic distribution.  Factor is described below.
+ *Z*: Number of topics.
+ *views*: Experimental.  Which views draw from this prior.  Set to array
{0} for now (single view).
+ *initDeltaBias*: Initial value for delta bias term.
+ *sigmaDeltaBias*: Determines strength of Gaussian prior centered at 0 for delta bias.  Lower value -> stronger prior -> delta bias pulled more strongly towards 0.
+ *optimizeMe*: If false, alpha/delta parameters will not be updated.  This
goes for factors influencing this prior as well as the delta bias term.

### SpritePhiPrior ###

Keeps track of prior over word distribution for each topic.  Constructor arguments similar to SpriteThetaPrior:

+ *factors*: Array of Factors that influence topic->word distribution.
+ *Z*: Number of topics.
+ *views*: Set to {0} -- same as SpriteThetaPrior
+ *initOmegaBias*: Initial value for omega bias term
+ *sigmaOmegaBias*: Strength of zero-centered Gaussian prior on omega bias.
+ *optimizeMe*: If false, beta/omega parameters are not updated -- also
applies to factors feeding into this prior.

### Factor ###

Factors influence the prior on document and topic distribution.  Factors
can be observed, where the alpha values are provided as explicit
supervision in the input file, or latent, in which case an alpha vector is
inferred for each document.  The Factor object keeps track of a set of
alpha, delta, beta, and omega parameters in the topic/document priors.
Constructor arguments for factor:

+ *numComponents*: How many components are included in this factor.  This
determines the number of alpha component values per document, beta and
delta parameters per topic, and omega vectors.
+ *viewIndices*: Only worry if dealing with multiple views.  Set to {0}
otherwise (single view).
+ *Z*: Number of topics per view.  For a single view, set to {Z} where Z is
the number of topics.
+ *rho*: The Dirichlet sparsity hyperparameter (see the SPRITE paper). If greater than or equal to 1, no sparsity is learned over beta
parameters.  If less than 1, this will determine how strongly we prefer a sparse beta (each topic influenced by one/few components of this factor).  This will learn betaB sparsity bits for each beta.
+ *tieBetaAndDelta*: If true, then beta and delta will be tied together.  This allows alpha to influence the topic as well as document distribution.
+ *sigmaBeta*: Determines strength of zero-mean Gaussian prior on beta parameters.
+ *sigmaOmega*: ... prior on omega parameters.
+ *sigmaAlpha*: ... prior on alpha parameters.  Only applicable if this factor is not observed (unsupervised).
+ *sigmaDelta*: ... prior on delta parameters.
+ *alphaPositive*: If true, alpha is exponentiated to ensure it is strictly positive.
+ *betaPositive*: Exponentiate beta.
+ *deltaPositive*: Exponentiate delta.
+ *factorName*: A descriptive name.  When writing out learned parameters
for this factor, the filename will contain this name.
+ *observed*: If true, then alpha for each document will be read in from the input file and not inferred.
+ *optimizeMeTheta*: If true, then the alpha/delta parameters will be updated via gradient descent on log-likelihood of training corpus.
+ *optimizeMePhi*: If true, then beta/delta will be updated.  Otherwise, these are fixed to their initial values.

### More examples ###

Here are a few examples of common models you can run out-of-the-box.
Looking into the initialization of each model is instructive.

    # Learning parameters
    LEARNING_PARAMS=" -Z 20 -C 5 -nthreads 2 -step 0.01 -iters 2000 -samples 100 "
    
    # Train DMR model on debates data
    OUTPUT_DIR=${SPRITE_HOME}/dmr_output; mkdir ${OUTPUT_DIR}
    java -cp ../lib/jcommander-1.49-SNAPSHOT.jar:. models/factored/impl/SpriteDMR ${LEARNING_PARAMS} -input ../resources/test_data/input.debates.txt -outDir ${OUTPUT_DIR} -logPath ${OUTPUT_DIR}/debates.dmr.log
    python scripts/topwords_sprite_factored.py ${OUTPUT_DIR}/input.debates.txt --numscores 1 > ${OUTPUT_DIR}/input.debates.topics # Print out topics
    
    # Train DMR variant where beta/delta are tied (also influence topic distribution) on debates
    OUTPUT_DIR=${SPRITE_HOME}/dmr_wordDist_output; mkdir ${OUTPUT_DIR}
    java -cp ../lib/jcommander-1.49-SNAPSHOT.jar:. models/factored/impl/SpriteTopicPerspective ${LEARNING_PARAMS} -input ../resources/test_data/input.debates.txt -outDir ${OUTPUT_DIR} -logPath ${OUTPUT_DIR}/debates.dmr_wordDist.log
    python scripts/topwords_sprite_factored.py ${OUTPUT_DIR}/input.debates.txt --numscores 1 > ${OUTPUT_DIR}/input.debates.topics
    
    # Joint supertopic-perspective model on ratemd (in SPRITE paper).  Supertopic factor has 5 components.
    OUTPUT_DIR=${SPRITE_HOME}/joint_output; mkdir ${OUTPUT_DIR}
    java -cp ../lib/jcommander-1.49-SNAPSHOT.jar:. models/factored/impl/SpriteSupertopicAndPerspective ${LEARNING_PARAMS} -input ../resources/test_data/input.ratemd.txt -outDir ${OUTPUT_DIR} -logPath ${OUTPUT_DIR}/ratemd.joint.log
    python scripts/topwords_sprite_factored.py ${OUTPUT_DIR}/input.ratemd.txt --numscores 1 > ${OUTPUT_DIR}/input.ratemd.topics
    
### Multiple Views ###

Although this library was written to allow a document to consist of tokens
in multiple views (each view with its own topic distributions, but having
a shared prior) we have not verified that the multiview implementation is
correct.  Refer to src/models/factored/impl/*2View.java for
implementations.  Use at your own risk.

### Notes and Advice ###

+ See the models in *src/models/factored/impl/* for sample implementations.
To create an implementation, initialize thetaPriors, phiPriors,
and the factors that feed into them.  See the constructors for
*prior.SpriteThetaPrior*, *prior.SpritePhiPrior*, *models.factored.Factor*
for details.
+ *thetaPriors.length == phiPriors.length == numViews*  These priors keep
track of \widetilde{\theta} and \widetilde{\phi} for each view.  Factors
are not tied to a specific view.
+ The order in which *Factor[] factors* lists observed factors is the same
order as how they are read from the input file.  Ordering of latent factors
shouldn't matter.
+ Setting *-logPath* will continue to write the stdout as well as the log
file.
+ Legal command line arguments are listed in *utils.ArgParse.Arguments*.
Feel free to extend these as you see like.  Calling a model implementation
with *--help* prints legal arguments.
+ The best stddev values (set with the argument *-sigmaAlpha*, *-sigmaBeta*,
and so on) are typically in the range of 0.1, 1.0, and 10.0.
+ For short documents like tweets, it is helpful to initialize delta to smaller
values than the default, using the *-deltaB* argument. We recommend -4.0.

## Topwords ##

Once a model is trained, you probably want to see what topics are learned,
and how the factors influence document/topic distributions for each topic.
You can do so with *topwords_sprite_factored.py*.  Run with '--help' for
options.

### Usage ###

    topwords_sprite_factored.py [-h] [--numscores NUMSCORES] [--includeprior] [--numtopwords NUMTOPWORDS] basename

+ *basename* points to where the .assign file is written, w/o ".assign" (see examples above) 
+ *--numscores NUMSCORES* an integer, number of observed factors
+ *--includeprior* whether to include prior pseudcounts when building topics.  Defaults to False.
+ *--numtopwords NUMTOPWORDS* number of top words to print out per topic or omega.  Defaults to 20

## TODO ##

+ Maven for build tool
+ Calculate held-out perplexity/do prediction of factors
+ Add option for annealing (as done in the original SPRITE paper)
+ Set up arbitrary graph with configuration file and manipulate components
by command line -- not major priority
+ Verify multiview support is working correctly
+ Add annealing of sparsity term -- rho is fixed for all iterations in the current implementation.
+ Low priority: support for more arbitrary functions to generate the prior

## Cite ##

If you use this library, please cite:

Paul, Michael J., and Mark Dredze (2015) SPRITE: Generalizing topic models with structured priors. Transactions of the Association for Computational Linguistics 3: 43-57.

## Contact ##

Adrian Benton <adrian.benton@gmail.com>

Michael J. Paul <mpaul39@gmail.com>
