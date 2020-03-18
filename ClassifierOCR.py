import numpy as np
import csv
import cv2 as cv2
import glob
import os
from sklearn.ensemble.forest import RandomForestClassifier
import xlrd 
 
def all_same(items): #describe whitespace or not
    return all(x == 0 for x in items)

def computeIndex(val,thresh): #Calculating Whitespaces Indexes
    val1=-1
    val2=-1
    row=thresh.shape[0]
    for j in range(val,row):
        
        print(j,val1,val2)
        if(all_same(thresh[j,:])!=1 and val1==-1):
            val1=j
        elif (val1!=-1 and all_same(thresh[j,:])==1):
            val2=j
            break
        elif (val1!=-1 and j==row-1):
            val2=j
    return val1,val2

f = open("file.txt", "w")
#TRAINING CODE 
loc = ("Filealpha.xlsx")  
wb = xlrd.open_workbook(loc) 
sheet = wb.sheet_by_index(0) 
sheet.cell_value(0, 0)  

training_data = []
classes=[]
label=[]

images=[0]*79
i=0
while(i<79):
   # fam = (unpickle("/home/linux/Desktop/cifar-10-batches-py/data_batch_"+str(i)))
    data="English Alphabet Dataset/"+str(i+1)  
    lab=sheet.cell_value(i, 0)
    images[i]=[]
    if(type(lab)==float):
        lab=int(lab)
    print("label ",lab)
    label.append(lab)
    for filename in glob.glob(data+'/*.png'):
        im=cv2.imread(filename,0)
        images[i].append(im)
    i=i+1
  
windowSize = (32,32)
blockSize = (16,16)
cellSize = (8,8)
nbins = 9

desc = cv2.HOGDescriptor(windowSize,blockSize,(8,8),cellSize,nbins)

d = desc.compute(images[0][0])
length,width = d.shape
#newArr = np.zeros([len(images),length],np.float32)
newArr=np.zeros((len(images)*30,length))
labelArr = np.empty(len(images)*30, dtype='|S6')
#print(images[0][0])
count=0
for i in range(len(images)):
    for j in range(len(images[i])):
        d = (desc.compute(images[i][j])).reshape(-1)
        newArr[count]= d
        labelArr[count] = label[i]
        count+=1
classifier = RandomForestClassifier(n_estimators = 50, max_depth=None, min_samples_split=2, random_state=0)
classifier.fit(newArr,labelArr)
print(classifier.score(newArr,labelArr))


#Predictions
src=cv2.imread("Test.png",0)
row,col=src.shape

ret, thresh2 = cv2.threshold(src,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)#Thresholding
thresh=np.array([0 if thresh2[i,j]==255 else 255 for i in range(row) for j in range(col) ],np.uint8).reshape(row,col) #Reversing foreground with background

indexValue=0
avgSpaceSize=0
check=0
while (True):
    val1,val2=computeIndex(indexValue,thresh) #Cropping a row to write        
    print(val1,val2)
    indexValue=val2+1
    
    if(val1==-1 or val2==-1): #IF all words are extracted
        break
    patch1=thresh[val1:val2,:] #Extracting row from the array
    rowPatch,colPatch=patch1.shape
        
    patch1 = cv2.rotate(patch1, cv2.ROTATE_90_CLOCKWISE) #Rotating the row
    
    output = cv2.connectedComponentsWithStats(patch1) #Finding Components
    annotations=output[2]
    for i in range(1,output[0]):
        c=i
        patch=[]
        patch=patch1[annotations[c,1]:annotations[c,1]+annotations[c,3],annotations[c,0]:annotations[c,0]+annotations[c,2]]
        
        expandedPatch=np.zeros((patch.shape[0]+2,patch.shape[1]+2))
        expandedPatch[1:-1,1:-1]=patch
        patch=expandedPatch
        print(patch.shape)
        
        if(annotations[i,4]<30 or check==1):
            check=0
            continue
        
        if(i<((output[0])-1)):#Calculating SpaceSize
            spaceSize=(annotations[c+1,1])-(annotations[c,1]+annotations[c,3])
            print(spaceSize)
            if(spaceSize<0):
                check=1
            else:
                avgSpaceSize+=spaceSize
                avgSpaceSize=avgSpaceSize/2

        rimg=cv2.resize(patch,(52,48))
        patch2=np.zeros((60,60),np.uint8)
        patch2[6:54,4:56]=rimg
        patch=patch2
        patch = cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE) #Rotating back the character
       
        d1 = desc.compute(patch) #Computing HOG
        font=str(classifier.predict(d1.reshape(1,-1))) #Predicting
        font=font[3:-2]
        
#        if(annotations[i,4]<40 and (font=='q' or font=='o' or font=='o')): #ignoring dots
#            continue
#        
        print(font)
        f.write(font)
#        cv2.imshow("1",patch)
#        cv2.waitKey(0)
        if(spaceSize>(avgSpaceSize+2) and check!=1):
            f.write(" ")
    f.write('\n')
  
   
f.close()