# model.py
import csv
import cv2
from os import listdir
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint

FLIP = 'FLIP:'

# Load data from one folder
def load_data(data_folder):
    results = []
    steerings = []
    with open(data_folder + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # center image
            center_image_path = data_folder + '/IMG/' + line[0].split('\\')[-1]
            line[0] = center_image_path
            left_image_path = data_folder + '/IMG/' + line[1].split('\\')[-1]
            line[1] = left_image_path
            right_image_path = data_folder + '/IMG/' + line[2].split('\\')[-1]
            line[2] = right_image_path
            results.append(line)
            steerings.append(float(line[3]))
    return results, steerings
            
# Loop all sub folders of the data folder, load all driving_log.csv and merge it together for the total data
def load_all_data():
    lines = []
    steerings = []
    for f in listdir('./data'):
        results, steers = load_data('./data/' + f)
        lines += results
        steerings +=  steers
    return lines, steerings

# Use the left and right camera images with the correction.
# If the center angle is bigger than 0.05 (the car is going on the curve), flip the images to increase the data on the curve street.    
def augment_data(lines, steerings, correction=0.22):
    image_paths = []
    angles = []
    for line, steering in zip(lines, steerings):
        # Insert center image path and steer
        center_angle = float(line[3])
        image_paths.append(line[0])
        angles.append(center_angle)
        
        # Left camera image
        left_angle = center_angle + correction
        image_paths.append(line[1])
        angles.append(left_angle)
        
        # Right camera image
        right_angle = center_angle - correction
        image_paths.append(line[2])
        angles.append(right_angle)
        
        # Flip the images if the camera does not capture the straight forward image (angle < 0.05
        if center_angle >= 0.05:
            # flip center image
            image_paths.append(FLIP + line[0])
            angles.append(center_angle*-1.0)
            
            # flip left camera image
            image_paths.append(FLIP + line[1])
            angles.append(left_angle*-1.0)
            
            # flip right camera image
            image_paths.append(FLIP + line[2])
            angles.append(right_angle*-1.0)
    return image_paths, angles

def show_hist(angles):    
    num_bins = 41
    avg_samples_per_bin = len(angles)/num_bins
    hist, bins = np.histogram(angles, num_bins)
    width = 0.5 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
    plt.show()

# Generate a batch data with batch size - reference to Udacity lession in order to reduce the processing memory with a lot of data
def generator(image_paths, angles, batch_size=128):
    num_samples = len(image_paths)
    while 1: # Loop forever so the generator never terminates
        shuffle(image_paths, angles)
        for offset in range(0, num_samples, batch_size):
            batch_samples = image_paths[offset:offset+batch_size]
            batch_angles = angles[offset:offset+batch_size]
      
            images = []
            res_angles = []
            for batch_sample, batch_angle in zip(batch_samples,batch_angles):
                names = batch_sample.split(':')
                if len(names) == 2:
                    name = names[1]
                    image = cv2.imread(name)
                    imgOut = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_flip = cv2.flip(imgOut,1)
                    images.append(image_flip)
                else:
                    name = batch_sample
                    image = cv2.imread(name)
                    imgOut = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(imgOut)            
                res_angles.append(batch_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(res_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def NVidea():
    
    ch, row, col = 3, 160, 320
    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(row, col, ch),
            output_shape=(row, col, ch)))
    model.add(Cropping2D(cropping=((60,20),(0,0))))
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    #model.add(Dropout(0.5))
    model.add(Dense(50))
    #model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def train_model(image_paths, angles, batch_size=128,nb_epoch=15):
    #train_samples, validation_samples = train_test_split(lines, test_size=0.2)
    train_samples, validation_samples, train_angles, validation_angles = train_test_split(image_paths, angles, test_size=0.2)
    #train_samples, train_angles, validation_samples, validation_angles = train_test_split(image_paths, angles, test_size=0.2)
    # compile and train the model using the generator function
    train_generator = generator(train_samples, train_angles, batch_size=batch_size)
    validation_generator = generator(validation_samples, validation_angles, batch_size=batch_size)
    ch, row, col = 3, 160, 320
    model = NVidea()
    model.compile(loss='mse', optimizer='adam')
    # Apply EarlyStopping
    early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='min')
    # Apply ModelCheckpoint in order to save the model after each epoch, we will choose the best model between epoches
    checkpoint = ModelCheckpoint('model{epoch:02d}.h5')
    model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=nb_epoch, callbacks=[early_stop, checkpoint])
    #model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=nb_epoch)
    model.save('model.h5')
        
lines, steerings = load_all_data()
print('Total record:', len(lines))


#angles = np.array(steerings)
#show_hist(angles)

# Because the distribution is very unbalanced (almost straight forward images with angle <0.5), flip images if angle > 0.05 only
image_paths, angles = augment_data(lines, steerings)
# Show distribution
#angles_ar = np.array(angles)
#show_hist(angles_ar)

#train_model(image_paths, angles)

