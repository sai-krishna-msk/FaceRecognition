
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

### Use cases

> Steps you need to follow based on your use case are presented below 

##### First time

- Collect dataset

   Intially you need to train your model with people you want it to recognize, For which you need to have a dataset (You need to have more than on class(meaning you need to train the model on more than one member), So when prompted "How many people you want it to train on" enter two people)

  ```bash
  python create_dataset.py
  ```

  

- Train the images collected

  For training those images you need to execute the following

  ```bash
  python train_present
  ```



- Predict

  Finally to put your model to test execute the following

  ```bash 
  python test.py
  ```



##### Retrain

> If you want to add more people later for your model to identify,You can just follow the above steps your mode will simply append to the list of people it can identify 



### Note:

- You can change the training size by changing the config file
- Make sure when prompted for the name of the class/person you do not enter a name with space in it