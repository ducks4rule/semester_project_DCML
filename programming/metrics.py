import subprocess


def calculate_metrics(number: int = 0, sqtt_number: int = 1,
                      with_sps_metric: bool = False, directory: str = 'predictions/',
                      file_name: str = 'test',
                      with_inversion: bool = True):
# copy the prediction file and the ground truth file to the evaluation folder
    with_sps_metric = str(int(with_sps_metric))
    number = str(1)
    if with_inversion:
        name = 'w_inv_sqtt_' + str(sqtt_number)
    else:
        name = 'wo_inv_sqtt_' + str(sqtt_number)
    subprocess.run(['./get_the_metrics.sh',
                    directory + 'n_gram_prediction_' + file_name + '.pkl',
                    directory + 'n_gram_ground_truth_' + file_name + '.pkl',
                    name,
                    number,
                    with_sps_metric])

