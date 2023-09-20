import os, time
import csv
import enum
import string
import arguments.setting as setting
import datastructure.dataset as dataset
import  learning.train_new as train_new

# from learning.construct_features import construct_features
from progress.bar import IncrementalBar
# import learning.train
# import learning.test
import getopt, sys

def run(data, setting, log_dir, train=False, features_only=False, runs_start=0, runs=10, draw_map=False):
    """ Run model training and testing

    Parameters
    ----------
    data : Dataset
        Dataset to use for model training/testing
    setting : Setting
        Setting as specified by class
    train : bool
        True if want to train models
    features_only : bool
        True if only want to encode features
    runs_stat : int
        First run of Monte-Carlo cross-validation to run
    runs : int
        Last run of Monte-Carlo cross-validation to run
    draw_map : bool
        True if want to save attention maps
    """
    if runs_start >= runs:
        return 
    import numpy as np

    print(np.shape(data.train_set))
    print("_-----")
    # Create features
    # construct_features(data.get_train_set(), setting)
    # construct_features(data.get_validation_set(), setting)
    # construct_features(data.get_test_set(), setting)

    if features_only:
        return

    bar = IncrementalBar('Running Monte Carlo ', max=runs)

    balanced_accuracies = []
    sensitivities = []
    specificities = []
    # Iterate Monte-carlo
    for k in range(runs_start, runs):
        # @MPR
        fold_log_dir = os.path.join(log_dir, f'fold_{k}')
        os.makedirs(fold_log_dir)
        # Set split
        data.set_fold(k)
        # Train model
        # @MPR
        if train:
            train_new.train(data.get_train_set(), data.get_validation_set(), k, fold_log_dir, setting)
        """
        # Test model
        balanced_accuracy, sensitivity, specificity = learning.test.test(data.get_test_set(), k, setting, draw_map=draw_map)
        balanced_accuracies.append(balanced_accuracy)
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        bar.next()

    bar.finish()
    # Save results
    for patients in data.get_test_set():
        for p in patients:
            results_folder = setting.get_data_setting().get_results_folder()
            p.save_predicted_scores(results_folder)
            if draw_map:
                p.save_map()
    
    balanced_accuracies = np.array(balanced_accuracies)
    sensitivities = np.array(sensitivities)
    specificities = np.array(specificities)
    print(np.mean(balanced_accuracies))
    print(1.96*np.std(balanced_accuracies))
    print(np.mean(sensitivities))
    print(1.96*np.std(sensitivities))
    print(np.mean(specificities))
    print(1.96*np.std(specificities))

    print(sensitivities)
    print(specificities)
    combined = np.abs(sensitivities-specificities)
    print(np.mean(combined))
    print(1.96*np.std(combined))
        """

# @MPR
def run_train(data_directories, csv_file, working_directory, qupath_project_file):
    """ Set up setting and dataset and run training/testing
    """
    # @MPR
    s = setting.Setting(data_directories, csv_file, working_directory, qupath_project_file)

    data = dataset.Dataset(s)
    # @MPR
    log_dir = os.path.join(working_directory, 'Runs', f'{s.get_network_setting().get_run_identifier()}-{time.strftime("%y-%m-%d-%H-%M-%S")}')
    os.makedirs(log_dir, exist_ok = True)
    
    run(data, s, log_dir, train=True, features_only=False, runs_start=0,runs=s.get_network_setting().get_runs(), draw_map=True)



def main(argv):
    try:
        # @MPR
        opts, args = getopt.getopt(argv, "hd:c:w:q:", ["data_directory=","csv_file=","working_directory=", "qupath_project_file="])
    except getopt.GetoptError:
        # @MPR
        print('main.py -d <data_directory> -c <csv_file> -w <working_directory> -q <qupath_project_file>')
        sys.exit(2)
    opts_vals = [o[0] for o in opts]
    if not('-d' in opts_vals or '--data_directory' in opts_vals):
        print('Specify -d or --data_directory')
        sys.exit(2)
    if not('-c' in opts_vals or '--csv_file' in opts_vals):
        print('Specify -c or --csv_file')
        sys.exit(2)
    if not('-w' in opts_vals or '--working_directory' in opts_vals):
        print('Specify -w or --working_directory')
        sys.exit(2)
    # @MPR
    if not('-q' in opts_vals or '--qupath_project_file' in opts_vals):
        print('Specify -q or --qupath_project_file')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
             # @MPR
             print('main.py -d <data_directory> -c <csv_file> -w <working_directory> -q <qupath_project_file>')
        elif opt in ('-d', '--data_directory'):
            data_directory = arg.strip('[]').split(',')
        elif opt in ('-c', '--csv_file'):
            if type(arg) == str and arg.endswith('.csv'):
                csv_file = arg
            else:
                print("Wrong data type for -c or --csv_file should be path to .csv")
                sys.exit(2)
        elif opt in ('-w', '--working_directiory'):
            if type(arg) == str:
                working_directory = arg
            else:
                print("Wrong data type for -w or --working_directory should be string")
                sys.exit(2)
        # @MPR
        elif opt in ('-q', '--qupath_project:file'):
            if type(arg) == str and arg.endswith('.qpproj'):
                qupath_project_file = arg
            else:
                print("Wrong data type for -q or --qupath_project_file should be path to .qpproj")
                sys.exit(2)

    # @MPR
    run_train(data_directory, csv_file, working_directory, qupath_project_file)

if __name__=="__main__":
    #run_train()
    main(sys.argv[1:])

