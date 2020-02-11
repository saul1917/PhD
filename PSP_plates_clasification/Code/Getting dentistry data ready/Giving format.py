
# coding: utf-8

# In[1]:


#libs
import pandas as pd
import os
import shutil
import random


# In[2]:



#the csv that is going to be used
df = pd.read_csv('Complete Data Analysis.csv')
#directories used 
root_dir = '/home/miuser/Downloads/1. Previous data dentistry project/All/' 
current_dir = os.path.dirname(os.path.realpath('__file__'))
filenames = []
train_filenames = []
#dev_filenames = []
test_filenames = []

train = current_dir+"/train/"     
test = current_dir+"/val/"  


# In[3]:


"""
function: this function requests from the csv and the details in the name from image
if the plate evaluated is recommended to be discarded or not. 
Once we have the details we want from the name of the image, as the artifact number and the case 
We ask the csv the state of that data, if it was discarded or not
Then we copy the image into the folder requested 
"""

def order_initial_dataset(): 
    if not os.path.exists(current_dir+"/yes"):
        os.makedirs(current_dir+"/yes")
    if not os.path.exists(current_dir+"/no"):
        os.makedirs(current_dir+"/no")     
    #this for explores the directories, subdirectories and files from given rootdir 
    for subdir, dirs, files in os.walk(root_dir):
        #each file 
        for file in files:
            
            #this helps to recognize which artifact and which case was used in the image example: "TEST_001_empty.tif" will be ['TEST', '001', 'empty.tif']
            """
            name of file is going to be splited by _ 
            [0] test 
            [1] raw image number
            [2] art 
            [3] artefact number
            [4] state and extension

            this function moves the artifacts to folders 
            yes or no, depending on the results in the csv of the raw data 
            """

            split_name = file.split("_")
            row = []
            #there are some images that are from artifact 0 (no artifact superimposed)
            if(len(split_name) == 1):
                #new image from no cmos 
                label = 'N'
            elif(str(split_name[2]) != "empty.tif"):
                #request the row with the specific properties given by, using pandas
                row = df.loc[(df['Artifact Number'] == int(split_name[3])) & (df['Raw Image Number'] == int(split_name[1]))]
                #requesting the label from that column in the row requested before
                label = row['Observer Discard'].iloc[0]
            else:
                row = df.loc[(df['Artifact Number'] == 0) & (df['Raw Image Number'] == int(split_name[1]))]
                #requesting the label from that column in the row requested before
                label = row['Observer Discard'].iloc[0]

            #this is the whole address of the file 
            src_file = os.path.join(subdir, file)

            #if corresponds to yes discard 
            if(label == 'Y'):
                #put it in the respective folder
                shutil.copy(src_file, current_dir+"/yes")

            #if corresponds to no discard 
            elif(label == 'N'):
                #put it in the respective folder
                shutil.copy(src_file, current_dir+"/no")
            
order_initial_dataset()


# In[18]:


new = df.loc[df['Observer Discard'] == 'N']


# In[16]:


df


# In[14]:


"""
function: checks if some directories exists 
    if they don't exist they are created automatically
"""
def check_dirs(): 
    if not os.path.exists(current_dir+"/train"):
        os.makedirs(current_dir+"/train")
        os.makedirs(current_dir+"/train/yes")
        os.makedirs(current_dir+"/train/no")
    if not os.path.exists(current_dir+"/val"):
        os.makedirs(current_dir+"/val")
        os.makedirs(current_dir+"/val/yes")
        os.makedirs(current_dir+"/val/no")

check_dirs()


# In[5]:


"""
    function: The files contained in the folders from yes/no are going to be sorted randomly and then separate into 
    70% training and 30% validation 
    The commented options are for creating a test and validation set 100%
"""
def get_test_and_train(yes_no_str):
    
    global train_filenames
    global test_filenames
    global filenames
    train_filenames = []
    test_filenames = []
    filenames = []
    print(len(train_filenames))
    print(len(test_filenames)) 
    #this for explores the directories, subdirectories and files from given rootdir 
    
    
    for subdir, dirs, files in os.walk(current_dir+"/"+yes_no_str+"/"):
        #each file 
        print(files)
        for file in files:
            
            filenames.append(file)
    # make sure that the filenames have a fixed order before shuffling
    filenames.sort()  
    random.seed(230)
    # shuffles the ordering of filenames (deterministic given the chosen seed) 
    random.shuffle(filenames) 

    split_1 = int(0.7 * len(filenames))
    #split_2 = int(0.9 * len(filenames))

    train_filenames = filenames[:split_1]
    #dev_filenames = filenames[split_1:split_2]
    test_filenames = filenames[split_1:]
    


# In[6]:



"""
    function: copies and paste the respective images from the results given by the function called 
    get_test_and_train() to the respective folders
"""

def set_folders(yes_no_str):
    print(len(train_filenames))
    #print(train_filenames)
    print(len(test_filenames))
    #print(test_filenames)
    for filename in train_filenames: 
        #source address
        src_file = os.path.join(current_dir+"/"+yes_no_str+"/", filename)
        #copy the file to the new source 
        shutil.copy(src_file, train+yes_no_str+"/")  

    for filename in test_filenames: 
        src_file = os.path.join(current_dir+"/"+yes_no_str+"/", filename)
        #copy the file to the new source 
        shutil.copy(src_file, test+yes_no_str+"/")  
    
            
      


# In[7]:



get_test_and_train("yes")
set_folders("yes") 
yestrain = train_filenames
yestest = test_filenames
get_test_and_train("no")
set_folders("no") 
notrain = train_filenames
notest = test_filenames


# In[100]:



expected_label_column = list(df["Observer Discard"])

yes = expected_label_column.count("Y")
no = expected_label_column.count("N")
print("yes: " + str(yes))
print("no: " + str(no))


# In[99]:


print(train_filenames)


# In[12]:


name = "1.2.840.114257.1.1636752997079821607203186099770679.jpg"
split = name.split("_")
print(len(split))


# In[53]:


for subdir, dirs, files in os.walk("/home/miuser/Downloads/Ariana/4. cnn classify psp plates/2. Getting dentistry data ready/train/"):
    print(subdir)
    print(dirs)
    print(len(files))
            


# In[81]:



def listcommon(testlist,biglist):
    return list(set(testlist) & set(biglist)) 

print(listcommon(yestest, notrain))


# In[96]:


notrain.index("1.2.840.114257.1.16367529928793386811487505817502119358.jpg")

