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
    java -cp ../lib/jcommander-1.49-SNAPSHOT.jar:. main/models/factored/impl/SpriteLDA -Z 20 -nthreads 2 -step 0.01 -iters 2000 -samples 100 -input ../resources/test_data/input.acl.txt -outDir ${OUTPUT_DIR} -logPath ${OUTPUT_DIR}/acl.lda.log

In the current implementations, 200 iterations of Gibbs sampling are
performed, then we alternate between one step of Gibbs sampling and one
gradient udpate step until *-iters* iterations expire.

### Input Format ###

Document ID, observed factor scores, and text in each view are each
separated by tabs.  Tokens within each view and component scores for an
observed factor are separated by a single space.  See sample input files
under *resources/test_data*.

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
+ *sigmaDeltaBias*: Determines strength of gaussian prior centered at 0 for delta bias.  Lower value -> stronger prior -> delta bias pulled more strongly towards 0.
+ *optimizeMe*: If false, alpha/delta parameters will not be updated.  This
goes for factors influencing this prior as well as the delta bias term.

### SpritePhiPrior ###

Keeps track of prior over word distribution for each topic.  Constructor arguments similar to SpriteThetaPrior:

+ *factors*: Array of Factors that influence topic->word distribution.
+ *Z*: Number of topics.
+ *views*: Set to {0} -- same as SpriteThetaPrior
+ *initOmegaBias*: Initial value for omega bias term
+ *sigmaOmegaBias*: Strength of zero-centered gaussian prior on omega bias.
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
+ *rho*: If greater than or equal to 1, no sparsity is learned over beta
parameters?  If less than 1, this will determine how strongly we prefer a sparse beta (each topic influenced by one/few components of this factor).  This will learn betaB sparsity bits for each beta.
+ *tieBetaAndDelta*: If true, then beta and delta will be tied together.  This allows alpha to influence the topic as well as document distribution.
+ *sigmaBeta*: Determines strength of zero-mean gaussian prior on beta parameters.
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
    LEARNING_PARAMS=" -Z 20 -nthreads 2 -step 0.01 -iters 2000 -samples 100 "
    
    # Train DMR model on debates data
    OUTPUT_DIR=${SPRITE_HOME}/dmr_output; mkdir ${OUTPUT_DIR}
    java -cp ../lib/jcommander-1.49-SNAPSHOT.jar:. main/models/factored/impl/SpriteDMR ${LEARNING_PARAMS} -input ../resources/test_data/input.debates.txt -outDir ${OUTPUT_DIR} -logPath ${OUTPUT_DIR}/debates.dmr.log
    
    # Train DMR variant where beta/delta are tied (also influence topic distribution) on debates
    OUTPUT_DIR=${SPRITE_HOME}/dmr_wordDist_output; mkdir ${OUTPUT_DIR}
    java -cp ../lib/jcommander-1.49-SNAPSHOT.jar:. main/models/factored/impl/SpriteTopicPerspective ${LEARNING_PARAMS} -input ../resources/test_data/input.debates.txt -outDir ${OUTPUT_DIR} -logPath ${OUTPUT_DIR}/debates.dmr_wordDist.log
    
    # Joint supertopic-perspective model on ratemd (in SPRITE paper)
    OUTPUT_DIR=${SPRITE_HOME}/joint_output; mkdir ${OUTPUT_DIR}
    java -cp ../lib/jcommander-1.49-SNAPSHOT.jar:. main/models/factored/impl/SpriteSupertopicAndPerspective ${LEARNING_PARAMS} -input ../resources/test_data/input.ratemd.txt -outDir ${OUTPUT_DIR} -logPath ${OUTPUT_DIR}/ratemd.joint.log
    
### Multiple Views ###

Although this library was written to allow a document to consist of tokens
in multiple views (each view with its own topic distributions, but having
a shared prior) we are not confident that the multiview implementation is
correct.  Refer to src/models/factored/impl/*2View.java for
implementations.  Use at your own risk.

### Notes ###

+ See the models in *src/models/factored/impl/* for sample implementations.
They are pretty straightforward -- just initialize thetaPriors, phiPriors,
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

## Topwords ##

Once a model is trained, you probably want to see what topics are learned,
and how the factors influence document/topic distributions for each topic.
You can do so with *topwords_sprite_factored.py*

### Usage ###

    python scripts/topwords_sprite_factored.py /PATH/TO/BASENAME NUM_OBSERVED_FACTORS CSV_POLAR_FACTORS > /PATH/TO/BASENAME.topics

+ */PATH/TO/BASENAME* points to where 
+ *NUM_OBSERVED_FACTORS* is an integer, number of observed factors
+ *CSV_POLAR_FACTORS* is a comma-separated list of factors that have a
single component and you want the most positive and negative words listed
(e.g., sentiment).  For these factors, both the highest and lowest weighted
words are printed separately.

The number of top words to print out for each topic can be changed in the
script.  To generate topics for a set of models:

    python scripts topwords_batch.py /PATH/TO/OUTPUT/DIR1 /PATH/TO/OUTPUT/DIR2... 

if trained model output is written to the directories listed.  This will
treat no factors as polar (will only print out top N components).  Topic
output will be written to a *.topics* file for each model.

## TODO ##

+ Maven for build tool
+ Calculate held-out perplexity/do prediction of factors
+ Set up arbitrary graph with configuration file and manipulate components
by command line -- not major priority
+ Low priority: support for more arbitrary functions to generate the prior
+ Verify multiview support is working correctly

## Cite ##

If you use this library, please cite the following:

Paul, Michael J., and Mark Dredze. "SPRITE: Generalizing topic models with structured priors." Transactions of the Association for Computational Linguistics 3 (2015): 43-57.

## Contact ##

*<INSERT CONTACT INFO>*