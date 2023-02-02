import os
import matplotlib.pyplot as plt

def datasets_load(source_dirpath1, source_dirpath2):

    for subdir, dirs, files in os.walk(source_dirpath1):
        for dir in dirs:
            # collect information about molecular structure and chemical composition

            # check that the spectrum has been calculated. Otherwise, ingore the molecule case
            spectrum_filename_list1 = [f for f in os.listdir(source_dirpath1 + '/' + dir + '/') if
                                      f.endswith('.csv')]

            # check that the spectrum has been calculated. Otherwise, ingore the molecule case
            spectrum_filename_list2 = [f for f in os.listdir(source_dirpath2 + '/' + dir + '/') if
                                       f.endswith('.csv')]

            try:
                """
                assert len(
                    spectrum_filename_list1) > 0, "spectrum file missing from directory: " + source_dirpath1 + '/' + dir + '/'
                assert len(
                    spectrum_filename_list1) < 2, "too many spectrum files within directory: " + source_dirpath1 + '/' + dir + '/'
                assert len(
                    spectrum_filename_list2) > 0, "spectrum file missing from directory: " + source_dirpath2 + '/' + dir + '/'
                assert len(
                    spectrum_filename_list2) < 2, "too many spectrum files within directory: " + source_dirpath2 + '/' + dir + '/'
                """
                spectrum_filename1 = source_dirpath1 + '/' + dir + '/' + spectrum_filename_list1[0]
                spectrum_energies1 = []
                with open(spectrum_filename1, "r") as input_file:
                    count_line = 0
                    for line in input_file:
                        if 500 <= count_line <= 1000:
                            spectrum_energies1.append(float(line.strip().split(',')[1]))
                        elif count_line > 505:
                            break
                        count_line = count_line + 1

                spectrum_filename2 = source_dirpath2 + '/' + dir + '/' + spectrum_filename_list2[0]
                spectrum_energies2 = []
                with open(spectrum_filename2, "r") as input_file:
                    count_line = 0
                    for line in input_file:
                        if 500 <= count_line <= 1000:
                            spectrum_energies2.append(float(line.strip().split(',')[1]))
                        elif count_line > 505:
                            break
                        count_line = count_line + 1

                fig, ax = plt.subplots()
                ax.plot(spectrum_energies1, color="blue", linestyle='solid')
                ax.plot(spectrum_energies2, color="red", linestyle='solid')
                plt.ylim([-0.2, max(spectrum_energies1) + 0.2])
                plt.title("Molecule ID: "+f"{dir}")
                plt.tight_layout()
                plt.draw()
                plt.savefig("logs/"+f"_sample_{dir}.png")
                plt.close(fig)

            # file not found -> exit here
            except:
                pass


if __name__ == "__main__":
    dirpwd = os.path.dirname(os.path.abspath(__file__))
    source_dirpath1 = os.path.join(dirpwd, "dataset/QM8-LQ")
    source_dirpath2 = os.path.join(dirpwd, "dataset/QM8-HQ")
    datasets_load(source_dirpath1, source_dirpath2)