For implementing-:


Clone this repo-:

1)
```
git pull

```
2) clone the repo by pressing the cline button


Installing the required dependencies-:
> Open the command prompt
> navigate to folder (inside the folder)
> enter-:
```
pip install opencv-contrib-python
conda install -c menpo dlib
pip install imutils
```

At present I have seeded the dataset folder with elon musk , leonardo decaprio and steve jobs images , You can fill in your photos and test that out 

Here we are producing embeddings, You can use them to, train your own model(try experimenting)
```
python test.py
```
You can find your embeddings in output folder after running the model
During the process of training you can view which part of the image (face) is being converted to embeddings
