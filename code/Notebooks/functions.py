
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras.preprocessing import image
import pathlib
import cv2
import imghdr
from sklearn.metrics import classification_report


def TestImages(img_directory, input_shape, model= None):
    preds = []
    false_images = []
    for file in os.listdir(img_directory):
        f = os.path.join(img_directory, file)
        test_image_og = image.load_img(f,
                                    target_size = input_shape)
        test_image = image.img_to_array(test_image_og)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        preds.append(result[0][0])
        if result[0][0] < 0.5:
            false_images.append(test_image_og)
    plt.figure(figsize=(20, 20))
    for count, i in enumerate(false_images):
        x = len(false_images)
        ax = plt.subplot(x, x, count + 1)
        plt.imshow(i)
        plt.axis("off")
    return preds

def CheckImagesReport(s_dir):
    def CheckImages( s_dir, ext_list):
        bad_images=[]
        bad_ext=[]
        s_list= os.listdir(s_dir)
        for klass in s_list:
            klass_path=os.path.join (s_dir, klass)
            print ('processing class directory ', klass)
            if os.path.isdir(klass_path):
                file_list=os.listdir(klass_path)
                for f in file_list:               
                    f_path=os.path.join (klass_path,f)
                    tip = imghdr.what(f_path)
                    if ext_list.count(tip) == 0:
                        bad_images.append(f_path)
                    if os.path.isfile(f_path):
                        try:
                            img=cv2.imread(f_path)
                            shape=img.shape
                        except:
                            print('file ', f_path, ' is not a valid image file')
                            bad_images.append(f_path)
                    else:
                        print('*** fatal error, you a sub directory ', f, ' in class directory ', klass)
            else:
                print ('*** WARNING*** you have files in ', s_dir, ' it should only contain sub directories')

        return bad_images, bad_ext  

    good_exts=['jpg', 'png', 'jpeg', 'gif', 'bmp' ] # list of acceptable extensions
    bad_file_list, bad_ext_list=CheckImages(s_dir, good_exts)
    if len(bad_file_list) !=0:
        print('improper image files are listed below')
        for i in range (len(bad_file_list)):
            print (bad_file_list[i])
    else:
        print(' no improper image files were found')

    return bad_file_list
    
def DeleteIncompatibleImages(bad_file_list):
    for i in bad_file_list:
        if os.path.exists(i):
            os.remove(i)
        else:
            print("The file does not exist")
            
def GetLabels(model, dataset):
    # Get predictions using dataset and store them in np arrays
    predictions = np.array([]).reshape(0,1)
    labels = np.array([]).reshape(0,1)

    if model:
        for x, y in dataset:
            predictions = np.concatenate([predictions, model.predict(x)])
            labels = np.concatenate([labels, y.numpy()])
        predictions_conf = [1 if x >= 0.5 else 0 for x in predictions.flatten().tolist()]
    else:
        for x, y in dataset:
            labels = np.concatenate([labels, y.numpy()])
        predictions_conf = []
        
    # prepare predictions and labels for Confusion Matrix
    labels_conf = labels.flatten().tolist()
    
    return predictions_conf, labels_conf

def ConfusionMatrix(model = None, dataset = None, save_fig = False, save_fig_location = None):
    
    labels_conf, predictions_conf = GetLabels(model, dataset)
    
    # Make confusion matrix
    class_names = dataset.class_names
    con_mat = tf.math.confusion_matrix(labels = labels_conf, predictions=predictions_conf).numpy()
    #con_mat_df = pd.DataFrame(con_mat_norm,
    con_mat_df = pd.DataFrame(con_mat,
                              index = class_names,
                              columns = class_names)
    
    # plot on seaborn
    figure = plt.figure(figsize = (8, 8))
    sns.heatmap(con_mat_df, annot = True, cmap = plt.cm.Reds, fmt='g')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    if save_fig and save_fig_location:
        figure.savefig(save_fig_location, dpi = 300)
        print(f"figure has been saved to: {save_fig_location}")       
    
    return con_mat_df, figure

def CalculateAccuracy(confusion_matrix):
    tn = confusion_matrix.iloc[0,0]
    fp = confusion_matrix.iloc[0,1]
    fn = confusion_matrix.iloc[1,0]
    tp = confusion_matrix.iloc[1,1]


    return (tp + tn)/(tp + fp + tn + fn)

def CalculateF1Score(precision, recall):
    return 2 * ((precision * recall)/(precision + recall))

def CreateMetricsReport(model = None, dataset = None, confusion_matrix = None):   
    dp = 5
    # Creates report for precision, recall, f1-score from sklearn
    loss, precision, recall, auc = model.evaluate(dataset)
    labels_conf, predictions_conf = GetLabels(model, dataset)
    class_report = classification_report(labels_conf, predictions_conf, digits = dp)
    print(class_report)
    print('Test loss :', round(loss, dp))
    print('Test auc :',round(auc, dp))
    print('Test accuracy: ', round(CalculateAccuracy(confusion_matrix), dp))
#     print('Test precision :', precision)
#     print('Test recall :', recall)
#     print('Test F1 Score: ', CalculateF1Score(precision, recall))
 
    