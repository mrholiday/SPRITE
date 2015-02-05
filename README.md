# SPRITE #

Reimplementation of Michael Paul's SPRITE code.  Supports multiple views,
multi-threaded training, and easier implementation of new models.

## Dependencies ##

+ [JCommander](http://jcommander.org/#Download "JCommander") (included at
the moment)
+ Tested with JDK 1.7
+ Python 2.7 (for printing topwords and learned parameters)

## Building ##

    find . -iname src/*.java > source_files.txt
    javac -cp .lib/jcommander-1.48-SNAPSHOT.jar @source_files.txt

## Training ##

### Usage ###

    java -cp ./lib/jcommander-1.48-SNAPSHOT.jar 

### Input Format ###

Document ID, observed factor scores, and view text are each separated by
tabs.  Tokens within each view and component scores are separated by a
single space.  See the sample input files under *resources/test_data*.

### Notes ###

+ See the models in *src/models/factored/impl/* for sample implementations.
Setting up a new model.  They are pretty straightforward -- just initialize
thetaPriors, phiPriors, and the factors that feed into them.  See the
constructors for *prior.SpriteThetaPrior*, *prior.SpritePhiPrior*,
*models.factored.Factor* for details.
+ *thetaPriors.length == phiPriors.length == numViews*  These priors keep
track of \widetilde{\theta} and \widetilde{\phi} for each view.  Factors
are not tied to a specific view.
+ The order in which *Factor[] factors* lists observed factors is the same
order as how they are read from the input file.  Ordering of latent factors
shouldn't matter.
+ Legal command line arguments are listed in *utils.ArgParse.Arguments*.
Feel free to extend these as you

## Topwords ##


### Usage ###

    python scripts/topwords_sprite_factored.py /PATH/TO/BASENAME NUM_OBSERVED_FACTORS<int> CSV_POLAR_FACTORS<String> > /PATH/TO/topwords.txt

+ *NUM_OBSERVED_FACTORS* is an integer, number of observed factors
+ *CSV_POLAR_FACTORS* is a comma-separated list of factors that have a
single component and you want the most positive and negative words listed
(e.g., sentiment)

## TODO ##

+ Make omega collapse views, *W* the union of vocabulary of all views
+ Switch to Maven for build tool
+ Set up arbitrary graph with configuration file and manipulate components
by command line.  Not major priority at the moment.
+ Calculate held-out perplexity and do prediction
+ Write shell scripts to qsub on the grid
+ Low priority: support for more arbitrary functions in the graph, like
