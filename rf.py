import cv2
import os 
import numpy as np
from random import shuffle  
from tqdm import tqdm 
import matplotlib.pyplot as plt
from radiomics import featureextractor  
import SimpleITK as sitk
from sklearn.feature_extraction import image
from skimage.feature import greycomatrix, greycoprops
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
IMG_SIZE=1000
TRAIN_DIR = 'C:/ivey/nailgun/train'
TEST_DIR = 'C:/ivey/nailgun/test1'
# function for creating labels
def label_img(img): 
	word_label = img.split('_')[-1]
	 
	if word_label == 'good.jpeg': return 1
	elif word_label == 'bad.jpeg': return 0
# training dataset
def create_train_data(): 
	
    training_data = [] 
    cont=[]
    homo=[]
    eng=[]
    diss=[]
    ASM=[]
    corr=[]
    label1=[]
	# tqdm is only used for interactive loading 
	# loading the training data 
    for img in tqdm(os.listdir(TRAIN_DIR)): 
  
        # labeling the images 
        label = label_img(img) 
        label1.append(label)
        path = os.path.join(TRAIN_DIR, img) 
        img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img1= cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img1=img1.astype(np.uint32)   
        # calculating GLCM           
        g = greycomatrix(img1, distances=[1], angles=[0], levels=256,symmetric=True, normed=True)
        # calculating features
        cont1 = greycoprops(g, 'contrast')
        cont.append(cont1)
        diss1= greycoprops(g, 'dissimilarity')
        diss.append(diss1)
        homo1 = greycoprops(g, 'homogeneity')
        homo.append(homo1)
        eng1 = greycoprops(g, 'energy')
        eng.append(eng1)
        corr1 = greycoprops(g, 'correlation')
        corr.append(corr1)
        ASM1= greycoprops(g, 'ASM')
        ASM.append(ASM1)
        
        training_data.append([np.array(img), np.array(label)]) 

    shuffle(training_data) 
    np.save('train_data.npy',training_data)
    
    return training_data,label1,cont, diss, homo, eng, corr, ASM 
def process_test_data(): 
    testing_data = [] 
    cont_t=[]
    homo_t=[]
    eng_t=[]
    diss_t=[]
    ASM_t=[]
    corr_t=[]
    label2=[]
    for img in tqdm(os.listdir(TEST_DIR)):
        #label = img.split('_')[-1]
        label = label_img(img) 
        
        label2.append(label)
        path = os.path.join(TEST_DIR, img)
    	
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        #print(img)
        img2 = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        g = greycomatrix(img2, distances=[1], angles=[0], levels=256,symmetric=True, normed=True)
        cont2 = greycoprops(g, 'contrast')
        cont_t.append(cont2)
        diss2= greycoprops(g, 'dissimilarity')
        diss_t.append(diss2)
        homo2 = greycoprops(g, 'homogeneity')
        homo_t.append(homo2)
        eng2 = greycoprops(g, 'energy')
        eng_t.append(eng2)
        corr2 = greycoprops(g, 'correlation')
        corr_t.append(corr2)
        ASM2= greycoprops(g, 'ASM')
        ASM_t.append(ASM2)
        
        testing_data.append([np.array(img2),np.array(label)]) 
    shuffle(testing_data)

	
    np.save('test_data.npy', testing_data) 
    return testing_data,label2,cont_t, diss_t, homo_t, eng_t, corr_t, ASM_t
#calling functions	 
train_data,label1,cont, diss, homo, eng, corr, ASM = create_train_data() 
test_data,label2,cont_t, diss_t, homo_t, eng_t, corr_t, ASM_t = process_test_data() 
#empty dataframe
df=pd.DataFrame(columns=['cont', 'diss', 'homo', 'eng', 'corr', 'ASM','label'])


# adding values to the dataframe 
df['cont']=cont
df['diss']=diss
df['homo']=homo
df['ASM']=ASM
df['corr']=corr
df['eng']=eng
df['label']=label1
feat=df[['cont','diss' , 'homo', 'eng', 'corr']]
y=df['label']
#print(df)
X_train, X_test, y_train, y_test = train_test_split(feat, y, test_size=0.3)
# instantiate RF
clf=RandomForestClassifier(n_estimators=100,warm_start=True)
 
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
    # Calculate accuracy of the trained model

score = metrics.accuracy_score(y_test,y_pred)
print('accuracy:')
print(score)
print(clf.feature_importances_)
# testset
dfe=pd.DataFrame(columns=['cont_t', 'diss_t', 'homo_t', 'eng_t', 'corr_t','label'])

    
dfe['cont_t']=cont_t
dfe['diss_t']=diss_t
dfe['homo_t']=homo_t
dfe['ASM_t']=ASM_t
dfe['corr_t']=corr_t
dfe['eng_t']=eng_t
dfe['label']=label2
#print(dfe)
test_feat=dfe[['cont_t','diss_t' ,'homo_t', 'eng_t', 'corr_t']]
test_pred=clf.predict(test_feat)
test_pred_prob=clf.predict_proba(test_feat)
test_y=dfe['label']

score = metrics.accuracy_score(test_y,test_pred)

print('True labels:')
print(label2)
#print(test_pred_prob)

test_predi=[]
for i in range(len(test_pred)):
    if test_pred[i]==1: 
        test_predi.append('good')
    else:
        test_predi.append('bad')
print('predicted labels:')
print(test_predi)
fig = plt.figure() 
#visualizing output
for num, data in enumerate(test_data[:20]):
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(5, 5, num+1)
    orig = img_data        
    if test_pred[num]==1: 
        str_label='good'
    else:
        str_label='bad'
    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()


