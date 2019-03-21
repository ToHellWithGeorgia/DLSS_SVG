# Deep Learning Super Sampling on SVG Renderer

We build a recurrent autoencoder to perform supersampling task on the SVG renderer output. Please read below for how to install and build the model.
 
### Prerequiste

You need to have the following package installed in order to run the model

* pytorch 1.0
* matplotlib
* tqdm

The dataset has been pre-generated and can be found in:
```
draw_svg/svg/rand/dataset
```
If you are interested in how the dataset is generated, please refer to:
```
draw-svg/svg/rand/gen_rand.py  
draw-svg/svg/rand/svg2raw.py
draw-svg/svg/rand/raw2png.py  
draw-svg/svg/rand/splitdata.py  
```

### Run the model

There is a step-to-step model training instructions in:
```
DLSS_vae/visualize.ipynb
```
which we recommend you to check out. If you want to train your own model, you can run the following command:
```
cd DLSS_vae
python train.py --name [name] --lr [learning rate] --num_epochs [number of training epochs] --batch_size [n]
```
You can find the detailed arguments in:
```
DLSS_vae/args.py
```

This will save the best model, dev set images for each epoch in the following folder:
```
DLSS_vae/save/train/[name]/
```
We already have some training examples in the folder you can refer to.

### Contributions
Zhihan Jiang: Data generation scripts, model building and debugging
Wensi Yin: Model building and researching

