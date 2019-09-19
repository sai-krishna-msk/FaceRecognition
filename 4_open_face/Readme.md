

Installing the required dependencies-:
> Open the command prompt
> navigate to folder (inside the folder)
> enter-:
```
pip install opencv-contrib-python
conda install -c menpo dlib
pip install imutils
```



#### Steps you need to follow based on your use case are presented below 



- Collect dataset

   Intially you need to train your model with people you want it to recognize, For which you need to have a dataset (You need to have more than on class(meaning you need to train the model on more than one member), So when prompted "How many people you want it to train on" enter two people)

  ```bash
  python create_dataset.py
  ```

  

- Train the images collected

  For training those images you need to execute the following

  ```bash
  python train_present.py
  ```



- Real time

  Finally to make real time predictions 

  ```bash 
  python realTime.py
  ```

  - Authentication


  Finally to use it  for authentication

  ```bash 
  python auth.py
  ```


- To re-train the model from scrtch first run the following script to clear the model and the dataset and  start from collect dataset 

```bash
python delete_current.py

```



##### Retrain

> If you want to add more people in the future after you intially train, just follow the above steps, your model will simply append to the list of people it can identify 



