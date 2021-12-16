# coding=utf-8
import os
import genre
import csv


result_filename = "result.csv"
file_list = os.listdir("dataset\\CyclicBlues")

for i in range(1, 100):
    current_song = "blues.000" + str(i).zfill(2) + ".wav"
    cmd = "dataset\\CyclicBlues\\" + current_song
    print("Current training:", cmd)
    code_book, distort = genre.train(cmd)
    standard = distort * 2.35
    correction_count = 0

    for f in file_list:
        print("Processing the song: " + f)
        song_path = "dataset\\CyclicBlues\\" + f
        threshold = genre.get_distort(code_book, song_path)
        if threshold <= standard:
            correction_count = correction_count + 1
        print("====================================\n")

    a = (correction_count - 1) / (len(file_list) - 1)
    temp = [current_song, a]
    result = open(result_filename, 'a+', newline='')
    w = csv.writer(result, dialect='excel')
    w.writerow(temp)
    result.close()