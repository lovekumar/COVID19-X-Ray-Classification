from PIL import Image
import os


def run(spark, job_config_map):
    '''
    This is common function for classification (both binary as well as multi-class classification)

    :param spark: SparkSession object
    :param job_config_map: map for passing multiple parameters to the job
        It contains
            input_base_path  : location of the xray images
            output_base_path : output location for saving re_size images
            folders          : List of folder names to be re_sized
            resize_to        : output image size

    :return: None
    '''

    base_path = job_config_map['input_base_path']
    output_path = job_config_map['output_base_path']
    folders = job_config_map['folders']
    size = job_config_map['resize_to']

    for each_folder in folders:
        input_file_path = base_path + '/' + each_folder
        output_file_path = output_path + '/' + each_folder

        # List of files
        file_list = os.listdir(input_file_path)

        # Create output folders, if missing
        if not os.path.isdir(output_file_path):
            os.makedirs(output_file_path)

        # Resize each image and store it at output location
        for each_file in file_list:
            file_path = input_file_path + '/' + each_file
            resized_file_path = output_file_path + '/' + each_file
            im = Image.open(file_path)
            im_resize = im.resize((size, size), Image.ANTIALIAS)
            im_resize.save(resized_file_path)
