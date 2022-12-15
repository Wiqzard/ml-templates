# ml-templates

## basic structre

* data_provider
    * data_factory.py: responsible for processing data
    * data_loader.py: contains class responsible for creating a pytorch dataset
* exp
    * exp_basic.py: basic components of the experiment
    * exp_main.py: full class responsible for training, validation, testing and prediction
* layers
    * contains components of model
* models
    * contains different models, composition of layers
* tests
    * contains files for testing components of project
* utils
    * files for helper functions etc.
* run.py
    * file to execute with defined arguments
