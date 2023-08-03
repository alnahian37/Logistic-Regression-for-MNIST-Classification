# %%
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import random
import pickle
from tqdm import tqdm

# %%


######## Change it according to python script and data location ########
cwd=os.path.realpath(__file__)
script_name=os.path.basename(cwd)
cwd=cwd.replace(script_name,'')
print('Current Working Directory: ',cwd)
#os.chdir(r'C:\\Users\\nahia\Google Drive (nahian.buet11@gmail.com)\\Spring-22 Drive Folder\\ECSE 6850\\HW\\Programming 1')
os.chdir(cwd) #Change it to your own directory



# %%

#function for 1 hot label encoding
def label_encoding(label):
    
    y = np.zeros([5,len(label)])
    
    for i in range(len(label)):
        y[int(label[i]-1),i] = 1
        
    return y

#function for computing classification performance
def performance_metrics(y_true,y_pred_ind):
    y_pred=np.zeros(y_true.shape)
    for i in range(len(y_pred_ind)):
        y_pred[i,y_pred_ind[i]]=1
    
    errors=np.zeros(y_pred.shape[1])
    accuracies=np.zeros(y_pred.shape[1])
    for i in range(y_pred.shape[1]):
        errors[i]=np.sum(y_pred[:,i]!=y_true[:,i])/y_true.shape[0]
        accuracies[i]=1-errors[i]
    return accuracies,errors
    


#function for loading data
def load_data(f_loc, im_size):
    f_list = os.listdir(f_loc)
    x_data = np.ones([len(f_list),im_size+1])

    for i in range(len(f_list)):
        im = mpimg.imread(f_loc + '/' + f_list[i])
        im = np.reshape(im,[1,im_size])
        x_data[i:i+1,0:im_size] = im
    
    x_data = x_data/255    #normalization
    x_data = np.float32(x_data)
    x_data = x_data.transpose()
    
    return x_data




# %%

#loading Train data
train_dir = r'train_data'
im_size = 784
train_data = load_data(train_dir, im_size)

#loading Test data
test_dir = r'test_data'
test_data = load_data(test_dir, im_size)


# %%


#loading and encoding Train Labels
path_to_label = r'labels'
tr_labels = np.loadtxt(path_to_label+'/'+'train_label.txt')
train_label = label_encoding(tr_labels)
train_label = np.float32(train_label)

#loading and encoding Test Labels
te_labels = np.loadtxt(path_to_label+'/'+'test_label.txt')
test_label = label_encoding(te_labels)
test_label = np.float32(test_label)

# %%
print("Train data shape:",train_data.shape)
print("Train label shape:",train_label.shape)
print("Test data shape:",test_data.shape)
print("Test label shape:",test_label.shape)


# %%
Y=np.transpose(train_label)

print(Y.shape)

X=np.transpose(train_data)

print(X.shape)

X_test=np.transpose(test_data)
print(X_test.shape)
Y_test=np.transpose(test_label)
print(Y_test.shape)



#Function for parameter update
def param_update(X,Y,W):
    A=np.exp(X@W)
    Z=A/np.sum(A,axis=1,keepdims=True)
    B=np.log(Z)
    Loss=-np.sum(B*Y) #Total loss over mini-batch
    

    Q=Y-Z
    grad=-X.T@Q #Gradient of loss wrt W over mini-batch

    W_new=W-lr*(grad+2*lam*W) #Update of W with regularization
    return Loss,W_new   



# %%
lr=0.0001 #learning rate
lam=0.1 #Regularization parameter
W=np.random.uniform(low=0.0, high=0.1, size=(X.shape[1],Y.shape[1]))#small weights initialization from Uniform distribution
batch_size=100

epoch=1000 #max number of epochs

num_batch=int(len(X)/batch_size) #Total number of batches
print('total number of batches:',num_batch)


best_error=100000 #Arbitrary large number, will be replaced by the lowest error during iterations
best_W=np.zeros((X.shape[1],Y.shape[1])) #Arbitrary zero weights, will be replaced by the best weights during iterations

train_errors=[]
train_accuracy=[]
test_errors=[]
test_accuracy=[]
train_losses=[]
test_losses=[]



for i in tqdm(range(epoch)):
    index=list(range(X.shape[0]))
    random.shuffle(index) #shuffle the index to get random batches for iterations in each epoch. 

    #weight update over mini-batches    
    for j in range(num_batch): #Runs code over all batches in one epoch
        a=index[j*batch_size:(j+1)*batch_size]
        x=X[a,:]
        y=Y[a,:]
    
        loss,W=param_update(x,y,W)
    
    tr_loss,_=param_update(X,Y,W)
    te_loss,_=param_update(X_test,Y_test,W)
    train_losses.append(tr_loss)
    test_losses.append(te_loss)
    train_acc,train_error=performance_metrics(Y,np.argmax(X@W,axis=1))
    test_acc, test_error=performance_metrics(Y_test,np.argmax(X_test@W,axis=1))

    
    train_errors.append(train_error)
    train_accuracy.append(train_acc)
    test_errors.append(test_error)
    test_accuracy.append(test_acc)
    if np.mean(test_error)<best_error:
        best_error=np.mean(test_error)
        best_W=W
        
        best_epoch=i   

     
      
  
  
      

# %%
print('Best Epoch: ',best_epoch)
print('Best Average Test Error: ',best_error)
print('Best errors for each digit:', test_errors[best_epoch])

# %%
best_test_acc, best_test_error=performance_metrics(Y_test,np.argmax(X_test@best_W,axis=1))
print('Best Test Accuracy: ',best_test_acc)
print('Best Test Error: ',best_test_error)
print('best average test accuracy: ',np.mean(best_test_acc))
print('best average test error: ',np.mean(best_test_error))

# %%
best_train_acc, best_train_error=performance_metrics(Y,np.argmax(X@best_W,axis=1))
print('Best Train Accuracy: ',best_train_acc)
print('Best Train Error: ',best_train_error)
print('best average train accuracy: ',np.mean(best_train_acc))
print('best average train error: ',np.mean(best_train_error))

# %%
train_errors=np.array(train_errors)
test_errors=np.array(test_errors)
train_accuracy=np.array(train_accuracy)
test_accuracy=np.array(test_accuracy)

for i in range(5):
    plt.figure()
    plt.plot(train_errors[:,i], label = "train error")
    plt.plot(test_errors[:,i], label = "test error")
    plt.legend()
    plt.xlabel('number of epochs')
    plt.ylabel('Error for digit ' + str(i+1))
    plt.show()

    plt.figure()
    plt.plot(train_accuracy[:,i], label = "train accuracy")
    plt.plot(test_accuracy[:,i], label = "test accuracy")
    plt.legend()
    plt.xlabel('number of epochs')
    plt.ylabel('Accuracy for digit ' + str(i+1))
    plt.show()

# %%
avg_er_train = np.sum(train_errors, 1)/5
avg_er_test = np.sum(test_errors, 1)/5
avg_acc_train = np.sum(train_accuracy, 1)/5
avg_acc_test = np.sum(test_accuracy, 1)/5

plt.figure()
plt.plot(avg_er_train, label = "overall train error")
plt.plot(avg_er_test, label ="overall test error")
plt.legend()
plt.xlabel('number of epochs')
plt.ylabel('training and test error')
plt.show()

plt.figure()
plt.plot(avg_acc_train, label = "overall train accuracy")
plt.plot(avg_acc_test, label ="overall test accuracy")
plt.legend()
plt.xlabel('number of epochs')
plt.ylabel('training and test accuracy')
plt.show()

# %%
#saving weights in a text file
#Shape of W is (785,5)
filehandler = open("multiclass_parameters.txt","wb")
pickle.dump(best_W, filehandler)
filehandler.close()

# %%
#Average loss per epoch
train_losses=np.array(train_losses)
test_losses=np.array(test_losses)
train_losses_avg=train_losses/len(tr_labels)
test_losses_avg=test_losses/len(te_labels)


# %%
#Plot Loss
plt.figure()
plt.plot(train_losses_avg, label = "train loss")
plt.plot(test_losses_avg, label = "test loss")
plt.legend()
plt.show()

# %%
#Plotting the learned weights
for i in range(5):
    image = best_W[0:784,i]
    image= image.reshape(28,28)
    plt.subplot(1,5,i+1)
    plt.imshow(image)
    plt.colorbar()
plt.show()

# %%
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

# %%
#Load the saved parameter files for checking everything is working fine
file=open("multiclass_parameters.txt","rb")
W_load=pickle.load(file)
print(W_load.shape)
file.close()

# %%
#Plotting the learned weights after loading
for i in range(5):
    image = W_load[0:784,i]
    image= image.reshape(28,28)
    plt.subplot(1,5,i+1)
    plt.imshow(image)
    plt.colorbar()
plt.show()




