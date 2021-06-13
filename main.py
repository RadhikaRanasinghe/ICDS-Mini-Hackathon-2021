from KNN import *
from dataset_handling import *

# Making the test_data.csv to numeric.
create_Numeric("test")

# Making the test_data.csv to numeric.
create_Numeric("train")

# Creating test and train data split (1:9).
create_test_train()

# Create oversample datasets (ROS/ADASYN).
create_oversample()

# Create undersample dataset (RENN).
create_undersample()

# Running KNN.
run_KNN("CRAP_ROS")

# Creating submission file.
final_KNN()
