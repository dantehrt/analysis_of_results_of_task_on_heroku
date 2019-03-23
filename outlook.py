import itertools
import matplotlib.pyplot as plt
import numpy as np
import time

def list_generator(number_of_A, number_of_B):
    factor_list = []
    for i in range(number_of_A):
        factor_list.append('A' + str(i+1))
    for i in range(number_of_B):
        factor_list.append('B' + str(i+1))

    return factor_list


def caluculate(factor_list):
    permutations = list(itertools.permutations(factor_list))
    print('順列数：' + str(len(permutations)))

    change_times_list = [0] * len(permutations)
    max_A_continued_times_list = [0] * len(permutations)
    max_B_continued_times_list = [0] * len(permutations)

    for i, permutation in enumerate(permutations):
        change_times = -1
        A_continued_times = 1
        B_continued_times = 1
        max_A_continued_times = 1
        max_B_continued_times = 1
        last_string = ''
        for factor in permutation:
            if factor[0:1] == last_string[0:1]:
                if 'A' in factor:
                    A_continued_times += 1
                    max_A_continued_times = A_continued_times if A_continued_times > max_A_continued_times else max_A_continued_times
                else:
                    B_continued_times += 1
                    max_B_continued_times = B_continued_times if B_continued_times > max_B_continued_times else max_B_continued_times
            else:
                A_continued_times = 1
                B_continued_times = 1
                change_times += 1
            last_string = factor

        change_times_list[i] = change_times
        max_A_continued_times_list[i] = max_A_continued_times
        max_B_continued_times_list[i] = max_B_continued_times
    return permutations, change_times_list, max_A_continued_times_list, max_B_continued_times_list


def sort_permutations(permutations, change_times_list, max_A_continued_times_list, max_B_continued_times_list, sort_by):
    sort_index = 0
    if sort_by == 'permutations':
        sort_index = 0
    elif sort_by == 'change_times_list':
        sort_index = 1
    elif sort_by == 'max_A_continued_times_list':
        sort_index = 2
    elif sort_by == 'max_B_continued_times_list':
        sort_index = 3

    temp_list = np.array([permutations, change_times_list, max_A_continued_times_list, max_B_continued_times_list])
    sorted_temp_list = temp_list[:, temp_list[sort_index, :].argsort()]

    return sorted_temp_list[0], sorted_temp_list[1], sorted_temp_list[2], sorted_temp_list[3]


def draw_graph(change_times_list, max_A_continued_times_list, max_B_continued_times_list):
    width = 0.3
    x = np.arange(len(change_times_list))
    plt.bar(x, change_times_list, width=width)
    plt.bar(x + width, max_A_continued_times_list, width=width)
    plt.bar(x + width * 2, max_B_continued_times_list, width=width)
    # plt.xticks(x + width*2 / 3)
    plt.show()


def draw_frequency_distribution(change_times_list, number_of_A, number_of_B):
    list_len = change_times_list.max()
    frequency_distribution = np.zeros(list_len)
    average_length_of_A = np.zeros(list_len)
    average_length_of_B = np.zeros(list_len)
    for index in range(list_len):
        change_times = index + 1
        frequency_distribution[index] = np.sum(change_times_list == change_times)
        if change_times % 2 == 0:
            average_length_of_A[index] = number_of_A * ((1 / (change_times / 2) + 1 / (change_times / 2 + 1)) / 2)
            average_length_of_B[index] = number_of_B * ((1 / (change_times / 2) + 1 / (change_times / 2 + 1)) / 2)
        else:
            average_length_of_A[index] = number_of_A * (1 / ((change_times + 1) / 2))
            average_length_of_B[index] = number_of_B * (1 / ((change_times + 1) / 2))

    x = np.arange(list_len) + 1
    width = 0.3
    plt.xlabel('change times')
    plt.ylabel('frequency distribution')
    plt.bar(x, frequency_distribution, width=width,label='Dam height (m)')
    plt.twinx()
    plt.ylabel('average length')
    plt.bar([0], [0], width=0, label='Frequency') # 凡例作成のためのダミー
    plt.bar(x + width, average_length_of_A, color='r', width=width, label='Average length of A')
    plt.bar(x + width * 2, average_length_of_B, color='g', width=width, label='Average length of B')
    plt.legend(shadow=True, loc='upper right')
    plt.show()




if __name__ == '__main__':
    start = time.time()
    number_of_A = 5
    number_of_B = 5
    factor_list = list_generator(number_of_A, number_of_B)
    elapsed_time = time.time() - start
    print("list_generator:{0}".format(elapsed_time) + "[sec]")

    start = time.time()
    permutations, change_times_list, max_A_continued_times_list, max_B_continued_times_list = caluculate(factor_list)
    elapsed_time = time.time() - start
    print("caluculate:{0}".format(elapsed_time) + "[sec]")

    start = time.time()
    permutations, change_times_list, max_A_continued_times_list, max_B_continued_times_list \
        = sort_permutations(permutations, change_times_list, max_A_continued_times_list, max_B_continued_times_list, 'change_times_list')
    elapsed_time = time.time() - start
    print("sort_permutations:{0}".format(elapsed_time) + "[sec]")

    # start = time.time()
    # draw_graph(change_times_list, max_A_continued_times_list, max_B_continued_times_list)
    # elapsed_time = time.time() - start
    # print("draw_graph:{0}".format(elapsed_time) + "[sec]")

    start = time.time()
    draw_frequency_distribution(change_times_list, number_of_A, number_of_B)
    elapsed_time = time.time() - start
    print("draw_frequency_distribution:{0}".format(elapsed_time) + "[sec]")

