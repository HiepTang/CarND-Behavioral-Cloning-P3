# Behavior Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Files Submitted and Code Quality
### Submission includes all required files
My project includes the following files:
* model.py containing the script to load, preprocess data and create, train the model
* drive.py for driving car in autonomous mode - increased the desired speed to 30 mph
* model.h5 contains a trained neural network model
* writeup_report.md summaries the result
### Driving car in autonomous mode
I have modified the original drive.py to increase the desired speed to 30. The below command uses to drive the car in autonomous mode
```sh
python drive.py model.h5
```
### The data preprocessing, model creating and training - model.py
#### Data loading and preprocessing
I have collected many different data for training such as clockwise, counterclockwise, curve, recovery on track 1 and some on track 2, so I need to implement the data loading function in order to load all sub directory of data folder and merge it together. It helps me to test easier with different data approach by doing copy the captured data to the data folder.
```python
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
```
The augment_data() function loads the left, right and center camera images and correct the angle with the correction. It also try to flip the captured images on the curve street in order to balance the dataset distribution.
```python
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
```
The input images are normalized with the Lamda layer, it will also make sure the model will normalize input images when making predictions in drive.py
```python
ch, row, col = 3, 160, 320
model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,
input_shape=(row, col, ch),
output_shape=(row, col, ch)))
```
The Keras Cropping2D is used for cropping the input images in order to remove some unuseful information on the top and bottom such as tree, sky, hill and the hood of the car.
```python
model.add(Cropping2D(cropping=((60,20),(0,0))))
```
#### Generators
In order to reduce the memory when working with a lot of data, I followed the Udacity lession to implement the generator function as below.
```python
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
```
