# Model Deployment Guide for AWS/Ubuntu Server

This guide provides step-by-step instructions to run our machine learning model on an AWS Ubuntu server and generate `Results.csv`.

## Prerequisites

- AWS account access
- SSH key pair (`your-key.pem`) stored in `~/.ssh/`
- Basic familiarity with terminal/command line

## Step-by-Step Instructions

### Step 1: Create AWS Instance
- Launch a new instance on Ronin/AWS with the following specifications:
  - **OS**: Ubuntu 22.04
  - **Instance Type**: t3.small (General Purpose)
  - **Storage**: 10 GB

### Step 2: Connect to Server
```bash
ssh -i ~/.ssh/your-key.pem ubuntu@<server-name>.nus.cloud
```

### Step 3: Install Python Package Manager
```bash
sudo apt install python3-pip
```

### Step 4: Confirm Installation
- When prompted `Do you want to continue? [Y/n]`, press `Y` and then `Enter`

### Step 5: Wait for Installation
- Wait for the installation to complete (press `Enter` if needed)

### Step 6: Install Required Python Packages
``` bash
pip install pickle5
pip install pandas
pip install -U scikit-learn
```

### Step 7: Download Model Files
``` bash
wget -O test_set.csv "https://raw.githubusercontent.com/yusukedt/dsa4262_group1/Lucas-branch/test_set.csv"
wget -O svm_train_2.pkl "https://raw.githubusercontent.com/yusukedt/dsa4262_group1/joshs-branch/svm_train_2.pkl"
wget https://raw.githubusercontent.com/yusukedt/dsa4262_group1/Lucas-branch/pkl_run.py
```

### Step 8: Make Script Executable
``` bash
chmod +x pkl_run.py
```

### Step 9: Run the Model
``` bash
python3 pkl_run.py test_set.csv svm_train_2.pkl
```
This will generate `test_result.csv` containing the model predictions.


### Step 10: Exit the server
``` bash
exit
```

### Step 11: Download the results
``` bash
scp -i ~/.ssh/your-key.pem ubuntu@<server-name>.nus.cloud:test_result.csv ~/Downloads/
```

### Step 12: Stop the Server
- Remember to stop the AWS instance through the management console to avoid unnecessary charges.

### Notes
- Replace <server-name> with your actual server name
- Ensure your SSH key has proper permissions (chmod 400 your-key.pem)
- The entire process typically takes 5-10 minutes to complete

Thank you for your hard work!
