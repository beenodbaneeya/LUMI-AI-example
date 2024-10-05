# LUMI-AI-example utilizing the LUMI-O/Setting up the container and running the program in LUMI
The purpose of this document is to explain how we can utilize the LUMI object storage by referrring to a specific use case.However, you should be familiar about the LUMI-O object store and what it offers.please refer to this document to learn more about the LUMI-O [`LUMI-O`](https://docs.lumi-supercomputer.eu/storage/lumio/) 


## Use Case: Training a Visual-Transformer model, where the dataset for training is stored using Object Storage in LUMI
The first step while you are working with the Object storage is to create the LUMI-O credentials.Please refer to this document to learn about how to create the LUMI-O credentials [`Create LUMI-O credentials`](https://docs.lumi-supercomputer.eu/storage/lumio/auth-lumidata-eu/).

Once you finish creating the credentials you will have things listed below with you which is sufficient for using the LUMI-O.
1. LUMI Project Number
2. Access Key
3. Secret key


This guide will first walk you through the process of making a dataset publicly available using LUMI-O and later give you details on how other users can access and use that dataset for their projects.Once you create youe LUMI credentials,you can configure the LUMI Object storage by following this process described below.


## Step 1: Configure LUMI Object Storage
1. **Log into LUMI**: Begin by logging into the LUMI system.
2. **Load the LUMI-O Module**: Load the `lumio` module using the following command:
    ```bash
    module load lumio
    ```
3. **Configure LUMI-O**: Once the module is successfully loaded, configure the object storage with the command:
    ```bash
    lumio-conf
    ```
    This command will prompt you to register your project number, access key, and secret key.You can get a brief overview here: [`Configuring the LUMI-O connection`](https://docs.lumi-supercomputer.eu/storage/lumio/)



## Making dataaset Publicly available


## Accessing the dataset and then using it for your project.
Once the dataset it made publicly available, you can access them for your project.Firt create the project directory where you want to copy the datasets.
You will need to know the project number if you are working on different project and also the path to the file that is made public on the other end.In our case, once we create the project directory inside LUMI, we can copy those dataset using this command.

command:
    ```bash
    rclone copyurl "https://project_number.lumidata.eu/imagenet/train_images.hdf5" /project/project_number/lumiai/train_images.hdf5
    ```

This command copy the files ```train_images.hdf5``` files inside our project directory specified by the path where we also need to define the filename where we want to copy the contents of those files.That is why both the source and destination has the file name called ```train_images.hdf5```

