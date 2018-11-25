For implementing-:

Installing the required dependencies-:
> Open the command prompt
> navigate to folder (inside the folder)
> enter-:
```
pip install opencv-contrib-python
conda install -c menpo dlib
pip install imutils
```

Just for trial purpose I have given a dataset as images of elon musk , leonardo de caprio , and steve jobs, I would recommend deleting all those three folders, If you wish to train it to your face

Creating Dataset:
For creating your won dataset of images , if you could just run
```
python create_dataset.py
```
Which would take your photos and train them (in one go), important point to remember is there should be more then one class(train it on more than one member)


If you wish to upload your own set of images as dataset you can do so, and run
```
python train_present.py
```

for testing the model
```
python test.py
```
