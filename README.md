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
    java -cp ../lib/jcommander-1.49-SNAPSHOT.jar:. main/models/factored/impl/SpriteLDA -Z 20 -nthreads 2 -step 0.01 -iters 2000 -samples 100 -input  -outDir ${OUTPUT_DIR} -logPath ${OUTPUT_DIR}/acl.lda.log

### Input Format ###

Document ID, observed factor scores, and text in each view are each
separated by tabs.  Tokens within each view and component scores for an
observed factor are separated by a single space.  See the sample input files
under *resources/test_data*.  

### Notes ###

+ Models that sample topics across multiple views are experimental,
at the moment (e.g., Sprite2ViewLDA).  We do not guarantee correctness or anything else -- use at your own risk.
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
