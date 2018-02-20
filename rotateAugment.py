import numpy as np 
import cv2
from skimage import io, transform
import shutil
import os
from distutils.dir_util import copy_tree
from pathlib import Path

class Augment:
    def __init__(self):
        self.rootPath = Path.cwd()
        self.directories = ['augImages', 'sanityCheck', 'data', 'badData', 
                            'goodData']
        self.dirDict = {}
        for i in self.directories:
            self.dirDict[i] = self.rootPath / i
        self.ROTATE_MIN = -33.3
        self.ROTATE_MAX = 33.3
        self.ROTATE_PER = 0.15
        
    def getImageFileList(self, path):
        '''RETURNS A LIST OF ALL FILES IN /DATA DIR THAT DO NOT END IN .TXT'''
        imageFileList = os.listdir(self.dirDict[path].as_posix())
        imageFileList = [i for i in imageFileList if i[-3:] != 'txt']
        return imageFileList
    
    def loadImage(self, fileName, path): 
        '''TAKES IN IMAGE FILENAME AND RETURNS IMAGE, OBJECT SIZE (HxW), 
        CLASSES AND LABELS'''
        image = cv2.imread((self.dirDict[path] / fileName).as_posix())
        sizeHW = [image.shape[0], image.shape[1]]
        textFile = fileName.split('.')[0]+'.txt'
        labels = (self.dirDict[path] / textFile).open()
        labels = [i.strip() for i in labels]
        labels = [i.split() for i in labels]
        labels = np.asarray(labels)
        classes = labels[:,0].astype(np.int)
        labels = labels[:,1:].astype(np.float)
        return image, sizeHW, classes, labels
    
    def drawBox(self, img, sizeHW, classes, labels, window):
        '''TAKES IN LABELS IN YOLO FORMAT AND CREATES ANNOTATED IMAGE'''
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i in range(len(classes)):
            x = (labels[i][0] - labels[i][2]/2) * sizeHW[1]
            y = (labels[i][1] - labels[i][3]/2) * sizeHW[0]
            w = labels[i][2] * sizeHW[1]
            h = labels[i][3] * sizeHW[0]
            cv2.rectangle(img,(int(x),int(y)),(int(x+w),
                           int(y+h)),(0,0,255),2)
            cv2.putText(img,str(int(classes[i])+1),(int(x + w/2),
                            int(y+h/2)), font, 1,(0,0,255),
                            1,cv2.LINE_AA)
        if window:
            cv2.imshow("annotated_img", img)
            key = cv2.waitKey(0)
            return key, img
        else:
            return img
    
    def checkData(self, path):
        '''LOOPS THROUGH DATA IN YOUR DATA FOLDER GIVING A PREVIEW OF LABELED
        IMAGE ALLOWING YOU TO REMOVE IMPROPERLY LABELED IMAGES IF MOVE IS TRUE
        IT WILL MOVE CHECKED DATA INTO SORTED FOLDERS IN ROOT DIRECTORY'''
        print('Press ESC to quit')
        filelist = self.getImageFileList(path)
        move = input('Do you want to seperate sorted data? (Y/N): ').lower()     
        if move == 'y':
            try:
                self.dirDict['goodData'].mkdir()
            except Exception as e:
                print(e.args[0])
            try:
                self.dirDict['badData'].mkdir()
            except Exception as e:
                print(e.args[0])
            print('Press Spacebar to move image into badData dir')
            print('Press any other key to keep data')
        
        for fileName in filelist:
            img, sizeHW, classes, labels, = self.loadImage(fileName, path)
            key,img = self.drawBox(img, sizeHW, classes, labels, window=True)
            image = self.dirDict[path] / fileName
            label = self.dirDict[path] / (fileName.split('.')[0]+'.txt')
            
            if key == 32 and move == 'y':
                print('Removing', fileName)
                image.replace(self.dirDict['badData'] / image.name)
                label.replace(self.dirDict['badData'] / label.name)
            elif key == 27:
                break
            elif move =='y':
                image.replace(self.dirDict['goodData'] / image.name)
                label.replace(self.dirDict['goodData'] / label.name)
        cv2.destroyAllWindows() 
                
    def convert_yolo_2_tf(self,label):
        '''Takes in text file of yolo coords and converts from: 
        [center x, center y, width height] in perentages to:
        [y_min, x_min, y_max, x_max] in percentages'''
        # CREATE NEW BLANK BOX IN FORMAT:   
        # [batch, number of bounding boxes, coords]
        numBoxes = len(label)
        boxes = np.zeros([1,numBoxes,4])
        # FILL IN NEW BOXES
        for i in range(numBoxes):
            boxes[:,i,0] = label[i][1]-label[i][3]/2
            boxes[:,i,1] = label[i][0]-label[i][2]/2
            boxes[:,i,2] = label[i][1]+label[i][3]/2
            boxes[:,i,3] = label[i][0]+label[i][2]/2
        # ENSURE VALUES ARE >= 0
        boxes[boxes<0] = 0
        return boxes
    
    def convert_tf_2_yolo(self, label):
        '''Takes in a list coordinates and converts the array
        [batch, number of bounding boxes, coords] 
        from: [y_min, x_min, y_max, x_max] in % 
        to: [center x, center y, width height] in %'''
        numBoxes = len(label[-1,:,:])
        boxes = np.zeros([numBoxes,4])
        for i in range(numBoxes):
            boxes[i][0] = (label[:,i,1] + label[:,i,3])/2
            boxes[i][1] = (label[:,i,0] + label[:,i,2])/2
            boxes[i][2] = (label[:,i,3] - label[:,i,1])
            boxes[i][3] = (label[:,i,2] - label[:,i,0])
        return boxes
    
    
    def rotateAllPoints(self, angle, tf_labels, sizeHW):
        '''THIS FUNCTION TAKES IN ANGLE, TF B.B LABELS AND THE SIZE OF THE IMAGE 
        AND WILL APPLY ROTATION TO THE LABELS AND RETURN LABELS IN TF FORMAT'''
        # TAKES IN TF FORMAT LABELS AND CREATES POINTS FOR EACH CORNER OF THE
        # BOUNDING BOX IN [X,Y] FORMAT
        boxes = tf_labels[-1,:,:]
        numBoxes = len(tf_labels[-1,:,:])
        all_points = []
        for i in range(numBoxes):
            points=[[boxes[i,1],boxes[i,0]],
                    [boxes[i,1],boxes[i,2]],
                    [boxes[i,3],boxes[i,0]],
                    [boxes[i,3],boxes[i,2]]]
            all_points.append(points)
        
        # CONVERT BOXES FROM IMG ARRAY IN PERCENTAGES FORM TO CARTESIAN PLANE
        all_points = np.asarray(all_points)
        all_points[:,:,1] = -((all_points[:,:,1]*sizeHW[0])-sizeHW[0]/2)
        all_points[:,:,0] = (all_points[:,:,0]*sizeHW[1])-sizeHW[1]/2
        
        # TAKES IN ALL POINTS ROTATES THEM
        rads = np.deg2rad(angle)
        rotation_matrix = np.array([[np.cos(rads), -np.sin(rads)],
                                    [np.sin(rads), np.cos(rads)]])
        new_points = []
        n = all_points.shape[0]
        for i in range(n):
            points = np.zeros([2])
            for ii in range(4):
                new_point = np.matmul(rotation_matrix,all_points[i,ii,:])
                points = np.vstack((points,new_point))
            new_points.append(points[1:,:])
        new_points = np.asarray(new_points)
        
        # CONVERT BACK TO IMG ARRAY FORM IN PERCENTAGES
        new_points[:,:,1] = (-(new_points[:,:,1])+sizeHW[0]/2)/sizeHW[0]
        new_points[:,:,0] = (new_points[:,:,0]+sizeHW[1]/2)/sizeHW[1]
        
        # CONVERT POINTS TO TF FORMAT
        all_boxes = np.zeros([4])
        for i in range(n):
            box = [np.min(new_points[i,:,1]),np.min(new_points[i,:,0]),
                   np.max(new_points[i,:,1]),np.max(new_points[i,:,0])]
            all_boxes = np.vstack((all_boxes,box))
        all_boxes = all_boxes[1:,:]
        
        # MAKE SURE ALL VALUES ARE >= 0 AND EXPAND DIMENSION FOR TF FORMAT
        all_boxes[all_boxes<0] = 0
        return np.expand_dims(all_boxes,0)
    
    def rotate(self, angle, image, label, sizeHW):
        '''TAKES IN ANGLE, IMAGE, YOLO FORMAT LABELS AND IMAGE SIZE AND 
        RETURNS ROTATED IMAGE AND ROTATED LABELS'''
        img = transform.rotate(image, angle, mode='edge')
        tf_labels = self.convert_yolo_2_tf(label)
        label = self.rotateAllPoints(angle, tf_labels, sizeHW)
        label = self.convert_tf_2_yolo(label)
        return img, label
    
    def rotateAugment(self, path):
        files = self.getImageFileList(path)
        try:
            self.ROTATE_MIN = int(input('What min rotation do you want? Default is {}: '.format(self.ROTATE_MIN)))
        except:
            print('Using default value of {}'.format(self.ROTATE_MIN))
        try:
            self.ROTATE_MAX = int(input('What max rotation do you want? Default is {}: '.format(self.ROTATE_MAX)))
        except:
            print('Using default value of {}'.format(self.ROTATE_MAX))
        try:
            self.ROTATE_PER = float(input('What percentage of rotation aug do you want? Default is {}: '.format(self.ROTATE_PER)))
        except:
            print('Using default value of {}'.format(self.ROTATE_PER))
        dataSanity = input('Do you want to sanity check your data? (Y/N) Default is (N): ').lower()
        
        try:
            self.dirDict['augImages'].mkdir()
        except Exception as error:
            print(error)
            
        if dataSanity == 'y':
            try:
                self.dirDict['sanityCheck'].mkdir()
            except Exception as error:
                print(error)
        
        for file in files:
            image, sizeHW, classes, labels = self.loadImage(file, path)
            boxes = self.convert_yolo_2_tf(labels)
            if len(boxes[boxes<=0.05]) == 0 and len(boxes[boxes>=0.95]) == 0:
                if np.random.rand() <= self.ROTATE_PER:
                    angle = np.random.randint(self.ROTATE_MIN, self.ROTATE_MAX)
                    rot_img, rot_coord = self.rotate(angle, image, labels, sizeHW)
                    filename = file.split('.')[0]
                    label = open((self.dirDict['augImages'] / (filename+'_aug.txt'))
                                 .as_posix(),'w')
                    for i in range(len(classes)):
                        label.write('{} {:.7f} {:.7f} {:.7f} {:.7f} \n'
                                    .format(classes[i], 
                                    rot_coord[i][0], rot_coord[i][1], 
                                    rot_coord[i][2], rot_coord[i][3]))
                    label.close()
                    
                    # CONVERT IMAGE TO RGB AND SAVE
                    io.imsave((self.dirDict['augImages'] / (filename+'_aug.jpg'))
                              .as_posix(), rot_img[...,::-1], quality=100)
                    
                    if dataSanity =='y':
                        img = self.drawBox(rot_img, sizeHW, classes, 
                                           rot_coord, window = False)
                        # BUG: SOME PIXEL VALUES GET SAVED AS 255
                        img[img>1] = 1
                        io.imsave((self.dirDict['sanityCheck'] / (filename+'.jpg'))
                                  .as_posix(), img[...,::-1], quality=100)
                        
    def moveFiles(self, oldPath, newPath):
        copy_tree(self.dirDict[oldPath].as_posix(), 
                  self.dirDict[newPath].as_posix())
        shutil.rmtree(self.dirDict[oldPath].as_posix())
        
    def makeTrainFile(self, path):
        dataTrainList = open((self.rootPath / 'dataset014_train.txt').as_posix(),'w')
        oldString = "/home/ubuntu/datasets/dataset014/data/"
        files = self.getImageFileList(path)
        for i in files:
            if i[-3::] != 'txt':
                dataTrainList.write(oldString+i+'\n')
        dataTrainList.close()
                        
def main():
    augment = Augment()
    
    while True:
        print('What would you like to do today?')
        print('--------------------------------')
        print('[1] - Check labeled data (yolo)')
        print('[2] - Create rotated-augment object detection data')
        print('[3] - Check augmented images')
        print('[4] - Move augmented images into data folder')
        print('[5] - Generate training file for ARVP darknet')
        print('[6] - Call it a day')
        choice = input(': ')
        
        if choice == '1':
            augment.checkData(path='data')
            
        if choice == '2':
            augment.rotateAugment(path = 'data')

        if choice == '3':
            if augment.dirDict['augImages'].is_dir():
                augment.checkData(path='augImages')
            else:
                print('\nNo Aug. Images Folder Found!\n')
        
        if choice == '4':
            augment.moveFiles(oldPath = 'augImages', newPath = 'data')
        
        if choice == '5':
            augment.makeTrainFile(path = 'data')
            
        if choice == '6':
            print("And a hell of a day it's been...")
            break
            
main()
