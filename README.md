# Deep Learning Project

This is the code which was used to perform the assignment of the 2020-2021 version of the Deep Learning Course at the RUG.

## Running an experiment
If one wishes to perform a parameter sweep like we did within our paper, execute python3 from the console while providing the path to the folder that contains the folder with the celebA images. In addition, the amount of simulation epochs can be specified:
```bash
python3 main.py --folder CELEB_A_PATH --epochs 5
```

## Running the latent explorer
Please note that the latent explorer was mainly used for internal exploration of the results and as such is not very polished. If one wishes to run it, run the main file in the latentexplorer folder. Four arguments need to be specified in order for this application to run. The location of the folder containing the folder that contains the celebA images, the folder that contains the encoder model, the folder that contains the decoder model, and the average class vectors that belong to that combination of decoder and encoder:
```bash
python3 main.py --celeba CELEB_A_PATH --enc PATH_TO_ENC --dec PATH_TO_DEC --avgVecs PATH_TO_AVG_VECS_FILE
```

## Project structure
### custom_loss
All code with regards to the implementation of our custom loss functions is stored here. As for our loss we combine several loss functions, we have created a class we deem the loss calculator, which will calculate the total loss as a combination of one or more loss functions.

### data
Contains code that is used to read in data

### latentexplorer
contains the latent explorer GUI program, see above.

### other
Contains a text document which contains resources that were used during the creation of the program.

### results
Contains pickled dictionaries holding the outcomes of a variety of experiments.

### util
Contains scripts that were used sporadically or for specialized purposes, or that did not fit in anywhere else in the project structure.

### vae
The code with regards to the variational autoencoder itself can be found here.
