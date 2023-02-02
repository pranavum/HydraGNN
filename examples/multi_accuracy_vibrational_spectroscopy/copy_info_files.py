import os
import shutil

def datasets_load(source_dirpath, destination_dirpath):

    for subdir, dirs, files in os.walk(source_dirpath):
        for dir in dirs:
            # collect information about molecular structure and chemical composition
            try:
                smilestr_files_list = [f for f in os.listdir(source_dirpath + '/' + dir + '/') if f.endswith('.dat')]
                if len(smilestr_files_list) == 1:
                    smilestr_file = source_dirpath + '/' + dir + '/' + smilestr_files_list[0]
                    shutil.copy(smilestr_file, destination_dirpath + '/' + dir + '/')
            # file not found -> exit here
            except IOError:
                pass


if __name__ == "__main__":
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    source_dirpath = os.path.join(dirpwd, "dataset/QM8-HQ")
    destination_dirpath = os.path.join(dirpwd, "dataset/QM8-LQ")
    datasets_load(source_dirpath, destination_dirpath)