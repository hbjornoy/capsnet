import os
import shutil
from numpy.random import seed, shuffle
import re
from sys import platform

def listdir_nohidden(path):
    """create a shuffled list of all the folders/files in a given folder"""
    seed(1)
    l = []
    for f in os.listdir(path):
        if not f.startswith('.'):
            l.append(f)
    shuffle(l)
    return l

def mkdir(path):
    """create folder if it doesnt already exist"""
    if not os.path.isdir(path):
        os.mkdir(path)

def create_symbols_dataset(num_classes):
    if platform == "darwin":
        capsnet_path = "/Users/havardbjornoy/capsnet"
    else:
        capsnet_path = "/work/stud/haavabjo/capsnet"

    raw = os.path.join(capsnet_path, "omniglot_raw_data")
    lang = os.path.join(capsnet_path, "language_dataset")
    symb = os.path.join(capsnet_path, "symbol_dataset")
    training_set = os.path.join(symb, "training_set")
    testing_set = os.path.join(symb, "testing_set")

    mkdir(symb)
    mkdir(training_set)
    mkdir(testing_set)

    class_counter = 0

    # iterate through raw/images_background
    for set in listdir_nohidden(raw):

        for lang in listdir_nohidden(os.path.join(raw, set)):
            print(set + '/' + lang)

            for char in listdir_nohidden(os.path.join(raw, set, lang)):
                if class_counter == num_classes:
                    print("created_dataset with num_classes: ", class_counter)
                    return
                class_name = lang + '@' + re.sub("[^0-9]", "", char)
                mkdir(os.path.join(training_set, class_name))
                mkdir(os.path.join(testing_set, class_name))
                class_counter += 1

                for index, instance in enumerate(listdir_nohidden(os.path.join(raw, set, lang, char))):
                    if index < 12:
                        shutil.copy(os.path.join(raw, set, lang, char, instance),
                                    os.path.join(training_set, class_name))
                    else:
                        shutil.copy(os.path.join(raw, set, lang, char, instance), os.path.join(testing_set, class_name))
                        if index > 19:  # check if imbalanced classes. all should be 20 instances
                            print(set + '/' + lang + '/' + char + " ---- " + index)


if __name__ == "__main__":
    if platform == "darwin":
        capsnet_path = "/Users/havardbjornoy/capsnet"
    else:
        capsnet_path = "/work/stud/haavabjo/capsnet"

    #print(capsnet_path)
    raw = os.path.join(capsnet_path, "omniglot_raw_data")
    lang = os.path.join(capsnet_path, "language_dataset")
    symb = os.path.join(capsnet_path, "symbol_dataset")
    training_set = os.path.join(symb, "training_set")
    testing_set = os.path.join(symb, "testing_set")


    # create symbol_dataset_folder
    mkdir(symb)
    mkdir(training_set)
    mkdir(testing_set)

    # iterate through raw/images_background
    for set in listdir_nohidden(raw):
        for lang in listdir_nohidden(os.path.join(raw, set)):
            print(set+'/'+lang)
            for char in listdir_nohidden(os.path.join(raw,set,lang)):
                class_name = lang+'@'+ re.sub("[^0-9]", "", char)
                mkdir(os.path.join(training_set, class_name))
                mkdir(os.path.join(testing_set, class_name))
                for index, instance in enumerate(listdir_nohidden(os.path.join(raw,set,lang,char))):
                    if index < 12:
                        shutil.copy(os.path.join(raw,set,lang,char,instance), os.path.join(training_set, class_name))
                    else:
                        shutil.copy(os.path.join(raw, set, lang, char, instance), os.path.join(testing_set, class_name))
                        if index > 19: #check if imbalanced classes. all should be 20 instances
                            print(set+'/'+lang+'/'+char+" ---- "+index)

