import clustering
import extract_features
import numpy as np
from operator import itemgetter
from scipy.cluster.vq import vq
import sys
import os


def pca(data, n_components):
    sample_sum = data.shape[0]
    dim = data.shape[1]
    mean = np.mean(data, axis=0)
    norm = data - mean

    covariance = (1 / sample_sum) * np.dot(norm.swapaxes(0, 1), norm)
    ew, ev = np.linalg.eig(covariance)

    eigen_matrix = []
    for i in range(dim):
        eigen_matrix.append((np.abs(ew[i]), ev[:, i]))
    eigen_matrix.sort(key=itemgetter(0), reverse=True)

    new_samples = np.array([sample[1] for sample in eigen_matrix[:n_components]])
    result = np.dot(norm, new_samples.swapaxes(0, 1))
    return result


def data_processing(file_path):
    feats = np.zeros((1, 39))
    pre_emphasis = 0.97

    wav_sig, sample_rate = extract_features.load_file(file_path)
    print("Loading successfully")
    print("Now truncating the first 30 seconds only")
    wav_sig = wav_sig[:int(30 * sample_rate)]

    frames = extract_features.enframe(wav_sig, 1024, 1024, pre_emphasis)  # nf*256
    feat = extract_features.delta_delta_mfcc(frames, 1024, sample_rate)  # nf*39
    feats = np.vstack((feats, feat))
    feats = feats[1:, :]

    new_mfccs = pca(feats, 8)
    return new_mfccs


def train(file_path):
    new_mfccs = data_processing(file_path)
    print("Start clustering...")
    centroids = clustering.run_kmeans(new_mfccs, 160, new_mfccs.shape[1])
    print("Calculating the space vector to find the closest path.")
    _, d = vq(new_mfccs, centroids)
    distort = d.sum() / new_mfccs.shape[0]
    print("Average distort is", distort)
    return centroids, distort


def get_distort(model, song):
    song_mfcc = data_processing(song)
    print("Get the MFCC value with shape: ", song_mfcc.shape)
    _, distort = vq(song_mfcc, model)
    result = distort.sum() / song_mfcc.shape[0]
    return result


if __name__ == '__main__':
    flag = sys.argv[1]
    target_file_list = sys.argv[2:]

    if flag == "-r" or flag == "--run":     # run with existing trained model
        npy_path = target_file_list[0].replace("/", "\\").replace("\\\\", "\\").replace("\\", "\\\\")

        start_index = npy_path.rfind("\\") + 1
        temp = npy_path[start_index:]
        model_genre = npy_path[start_index:start_index + temp.find(".")]

        model_cb = np.load(npy_path)
        dis_path = npy_path.replace(".npy", ".txt")

        with open(dis_path, 'r', encoding='utf-8') as f:
            distort = float(f.readline())
        standard = distort * 2.35
        # standard = distort + 8.5      # for testing
        print("Model loading successfully")

        p = target_file_list[1].replace("/", "\\").replace("\\\\", "\\").replace("\\", "\\\\")
        file_list = os.listdir(p)

        correction_count = 0
        for f in file_list:
            print("Processing the song: " + f)
            current_genre = f[:f.find(".")]
            song_path = p + "\\" + f
            threshold = get_distort(model_cb, song_path)
            print("Current threshold: ", threshold)
            if threshold <= standard:
                print("By prediction, this song is a", model_genre)
                if current_genre == model_genre:
                    correction_count = correction_count + 1
                    print("Correct prediction.")
                else:
                    print("Incorrect prediction. Current genre should be", current_genre)
            else:
                print("By prediction, these two songs does not belong to the same genre")
                if current_genre == model_genre:
                    print("Incorrect prediction. Two songs both belongs to", current_genre)
                else:
                    correction_count = correction_count + 1
                    print("Correct prediction. The model is", model_genre, "while the current song is", current_genre)
            print("====================================\n")
        print("The test is done. Among all the", len(file_list), "songs, the accuracy is", (correction_count / len(file_list)))

    elif flag == "-t" or flag == "--train":     # train the model and save
        target = target_file_list[0].replace("/", "\\").replace("\\\\", "\\").replace("\\", "\\\\")
        print("Loading training sample: " + target)
        print("Please wait...This training procedure may take long time depends on the target sample song")

        code_book, distort = train(target)

        npy_path = target[:target.rfind(".")] + ".npy"
        dis_path = target[:target.rfind(".")] + ".txt"
        print("Done...Saving the model to " + npy_path + " ...")
        np.save(npy_path, code_book)
        with open(dis_path, 'w+', encoding='utf-8') as f:
            f.write(str(distort))

        print("Training complete. The model is stored in:", npy_path, "\nAnd the sum of distort is stored in:", dis_path)

    elif flag == "-tr":     # run without existing trained model
        target = target_file_list[0].replace("/", "\\").replace("\\\\", "\\").replace("\\", "\\\\")

        start_index = target.rfind("\\") + 1
        temp = target[start_index:]
        model_genre = target[start_index:start_index + temp.find(".")]

        print("Loading training sample: " + target)
        print("Please wait...This training procedure may take long time depends on the target sample song")

        code_book, distort = train(target)

        npy_path = target[:target.rfind(".")] + ".npy"
        dis_path = target[:target.rfind(".")] + ".txt"
        print("Done...Saving the model to " + npy_path + " ...")
        np.save(npy_path, code_book)
        with open(dis_path, 'w+', encoding='utf-8') as f:
            f.write(str(distort))

        standard = distort * 2.35
        # standard = distort + 8.5      # for testing
        print("Training complete. The model is stored in:", npy_path, "\nAnd the sum of distort is stored in:", dis_path)

        p = target_file_list[1].replace("/", "\\").replace("\\\\", "\\").replace("\\", "\\\\")
        file_list = os.listdir(p)

        correction_count = 0
        for f in file_list:
            print("Processing the song: " + f)
            current_genre = f[:f.find(".")]
            song_path = p + "\\" + f
            threshold = get_distort(code_book, song_path)
            print("Current threshold: ", threshold)
            if threshold <= standard:
                print("This song is a", model_genre)
                if current_genre == model_genre:
                    correction_count = correction_count + 1
                    print("Correct prediction.")
                else:
                    print("Incorrect prediction. Current genre should be", current_genre)
            else:
                print("Two songs does not belong to the same genre")
                if current_genre == model_genre:
                    print("Incorrect prediction. Two songs both belongs to", current_genre)
                else:
                    correction_count = correction_count + 1
                    print("Correct prediction. The model is", model_genre, "while the current song is", current_genre)
            print("====================================\n")
        print("The test is done. Among all the", len(file_list), "songs, the accuracy is", (correction_count / len(file_list)))
    else:
        print("Missing or invalid flag parameter")
