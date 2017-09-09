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
I have collected many different data for training such as clockwise, counterclockwise, curve, recovery on track 1 and some on track 2, so I need to implement the data loading function in order to load all sub directory of data folder and merge it together.
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
