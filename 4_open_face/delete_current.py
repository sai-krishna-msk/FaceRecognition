import os 
import shutil

datasetURL = os.getcwd()+"/dataset"
modelsURL = os.getcwd()+'/output'


def delete_files():

	
	try:
		for each in os.listdir(datasetURL):
			shutil.rmtree(datasetURL+"/"+each)
	except:
		print("Dataset folder empty !")

	try:
		for each in os.listdir(modelsURL):
			
			os.remove(modelsURL+"/"+each)
	except:
		print(modelsURL+"modelsfolder empty !")
		return None
	with open(modelsURL+"/flag.txt" , 'w') as f:
		f.write("flag !")
		f.close()


delete_files()