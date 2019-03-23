import pandas as pd
import ast
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import datetime
import os
from scipy import stats

import data_processor, view_images

plt.style.use('ggplot')


class pycolor:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    END = '\033[0m'
    BOLD = '\038[1m'
    UNDERLINE = '\033[4m'
    INVISIBLE = '\033[08m'
    REVERCE = '\033[07m'


# タグタスクにおける比較
def compare_when_tag_task(answer, target):
    answer_input = ''
    target_input = ''
    number_of_correct = 0
    for i in range(1, 6):
        column_name = 'answer_text_box' + str(i)
        answer_input += '"' + answer[column_name] + '" '
        target_input += '"' + target[column_name] + '" '
        if answer[column_name] == target[column_name].lower():
            number_of_correct += 1

    return number_of_correct, answer_input, target_input


# バウンディングボックスタスクにおける比較
# 二つのdfを受け取る
count = 0


def compare_when_bounding_task(answer, target, debug=False):
    global count
    sorted_bounding_box_dic_list_of_answer = data_processor.make_sorted_bounding_box_dic_list(
        answer['answer_annotation_data'])
    number_of_bounding_boxes = len(sorted_bounding_box_dic_list_of_answer)

    if (not target.isnull()['answer_annotation_data']
            and not target['answer_annotation_data'] == '[]'):
        sorted_bounding_box_dic_list_of_target = data_processor.make_sorted_bounding_box_dic_list(
            target['answer_annotation_data'])
        image_url = target['input_image_url']

        # for文はバウンディングボックスの個数回処理される．
        # 精度が最も高かったものをmax_accuracy_rate_listに入れる
        # ソートして実際のバウンディングボックス個，上から取り出す．
        max_accuracy_rate_list = []

        # ターゲットのバウンディングボックスを一つずつ取り出す．
        for sorted_bounding_box_dic_of_target in sorted_bounding_box_dic_list_of_target:
            # 答えの中で最も適したものを求める
            accuracy_rate_list = []
            for sorted_bounding_box_dic_of_answer in sorted_bounding_box_dic_list_of_answer:
                accuracy_rate = calculate_accuracy_of_bounding_box(sorted_bounding_box_dic_of_answer,
                                                                   sorted_bounding_box_dic_of_target)
                accuracy_rate_list.append(accuracy_rate)
                # view_images.from_answer_dic_and_target_dic(sorted_bounding_box_dic_of_answer, sorted_bounding_box_dic_of_target, image_url)
            max_accuracy_rate_list.append(max(accuracy_rate_list))

        sum_of_accuracy_rate = sum(sorted(max_accuracy_rate_list, reverse=True)[:number_of_bounding_boxes])
        average_of_accuracy_rate = sum_of_accuracy_rate / number_of_bounding_boxes
        # if image_url == '054':
        #     if 0.5 < average_of_accuracy_rate < 0.6:
        #         print('average_of_accuracy_rate :' + str(average_of_accuracy_rate))
        #         view_images.from_answer_and_target_box_dic_list(sorted_bounding_box_dic_list_of_answer,
        #                                                         sorted_bounding_box_dic_list_of_target, image_url)
        #     if 0.9 < average_of_accuracy_rate :
        #         print('average_of_accuracy_rate :' + str(average_of_accuracy_rate))
        #         view_images.from_answer_and_target_box_dic_list(sorted_bounding_box_dic_list_of_answer,
        #                                                         sorted_bounding_box_dic_list_of_target, image_url)
        if debug:
            print(image_url)
            print(max_accuracy_rate_list)
            print('sum_of_accuracy_rate     :' + str(sum_of_accuracy_rate))
            print('number_of_bounding_boxes :' + str(number_of_bounding_boxes))
            print('average_of_accuracy_rate :' + str(average_of_accuracy_rate))
            if count % 10 == 0:
                view_images.from_answer_and_target_box_dic_list(sorted_bounding_box_dic_list_of_answer,
                                                                sorted_bounding_box_dic_list_of_target, image_url)
    else:
        average_of_accuracy_rate = 0
    count += 1
    return average_of_accuracy_rate


# タスク条件毎の，ワーカ一人当たりの平均タスク完了数
def calculate_average_number_of_completed_tasks(directory_path):
    # worker_idとワーカの結果の辞書
    worker_dic = data_processor.extract_dic_of_worker_who_complete_task(directory_path)

    mturk_worker_information = pd.read_csv(directory_path + '/mturk.workerinformation.csv',
                                           dtype={'input_image_url': 'str', 'answer_annotation_data': 'str'})
    task_conditions = list(set(worker_dic.values()))
    task_conditions.sort()

    for i in range(1, 21):
        # それぞれの条件下での合計
        sum_of_each_conditions = {}
        # それぞれの条件下での合計（100以上は100に補正）
        corrected_sum_of_each_conditions = {}
        # それぞれの条件下で該当するワーカ数（平均を取る際の分母）
        times_of_each_conditions = {}
        # 100を超えた回数
        times_of_over_100 = {}
        for task_condition in task_conditions:
            sum_of_each_conditions[task_condition] = 0
            corrected_sum_of_each_conditions[task_condition] = 0
            times_of_each_conditions[task_condition] = 0
            times_of_over_100[task_condition] = 0

        print(i)
        for worker_id, task_condition in worker_dic.items():
            number_of_completed_tasks = mturk_worker_information.query(
                'worker_id == "{}" and task_condition == "{}"'.format(worker_id, task_condition)).iloc[0][
                'number_of_completed_tasks']
            if number_of_completed_tasks >= i:
                # 100以上は100に補正
                if number_of_completed_tasks >= 100:
                    # print('task_condition:' + str(task_condition) + '  number_of_completed_tasks:' + str(
                    #     number_of_completed_tasks))
                    corrected_sum_of_each_conditions[task_condition] = corrected_sum_of_each_conditions[
                                                                           task_condition] + 100
                    times_of_over_100[task_condition] = times_of_over_100[task_condition] + 1
                else:
                    corrected_sum_of_each_conditions[task_condition] = corrected_sum_of_each_conditions[
                                                                           task_condition] + number_of_completed_tasks
                sum_of_each_conditions[task_condition] = sum_of_each_conditions[
                                                             task_condition] + number_of_completed_tasks
                times_of_each_conditions[task_condition] = times_of_each_conditions[task_condition] + 1

        # 条件毎の平均値
        average_of_each_conditions = {}
        # 補正された，条件毎の平均値
        corrected_average_of_each_conditions = {}
        for k, v in sum_of_each_conditions.items():
            average_of_each_conditions[k] = v / times_of_each_conditions[k]

        for k, v in corrected_sum_of_each_conditions.items():
            corrected_average_of_each_conditions[k] = v / times_of_each_conditions[k]

        print(sum_of_each_conditions)
        print(corrected_sum_of_each_conditions)
        print(times_of_each_conditions)
        print(average_of_each_conditions)
        print(corrected_average_of_each_conditions)
        print(times_of_over_100)

        plt.bar(average_of_each_conditions.keys(), average_of_each_conditions.values(), color='#E24A33')
        plt.bar(corrected_average_of_each_conditions.keys(), corrected_average_of_each_conditions.values(),
                color='#338ABD')
        plt.savefig('graphs/Figure' + str(i) + '.jpg')


# mturk.outputsから指定ワーカのデータを表示する（確認用）
def show_result_from_mturk_outputs(directory_path, worker_id):
    mturk_outputs = pd.read_csv(directory_path + '/mturk.output.csv',
                                dtype={'input_image_url': 'str', 'answer_annotation_data': 'str'})
    targets = mturk_outputs.query('worker_id == "{}"'.format(worker_id)).sort_values('submit_time')
    answers_of_tagging_task = pd.read_csv(directory_path + '/answers_of_tagging_task.csv',
                                          dtype={'input_image_url': 'str', 'answer_annotation_data': 'str'})
    answers_of_bounding_task = pd.read_csv(directory_path + '/answers_of_bounding_task.csv',
                                           dtype={'input_image_url': 'str', 'answer_annotation_data': 'str'})
    average_accuracy_rate_of_tagging, average_accuracy_rate_of_bounding = calculate_accuracy_of_df(targets,
                                                                                                   answers_of_tagging_task,
                                                                                                   answers_of_bounding_task)
    print('Annotation accuracy:{}'.format(average_accuracy_rate_of_tagging))
    print('BOunding accuracy:{}'.format(average_accuracy_rate_of_bounding))
    for index, targets_row in targets.iterrows():
        image_url = targets_row['input_image_url']
        accept_time = targets_row['accept_time']
        submit_time = targets_row['submit_time']
        work_time = targets_row['work_time']
        print('image_url:' + image_url)
        print('accept_time:' + accept_time)
        print('submit_time:' + submit_time)
        print('work_time:' + str(work_time))
        if targets_row['input_task_type'] == 't':
            answer = ''
            for i in range(1, 6):
                column_name = 'answer_text_box' + str(i)
                answer += '"' + targets_row[column_name] + '" '
            print(answer)
        elif targets_row['input_task_type'] == 'b':
            if (not targets_row.isnull()['answer_annotation_data']
                    and not targets_row['answer_annotation_data'] == '[]'):
                sorted_bounding_box_dic_list = data_processor.make_sorted_bounding_box_dic_list(
                    targets_row['answer_annotation_data'])
                view_images.from_bounding_box_dic_list(image_url, sorted_bounding_box_dic_list)


# バウンディングタスク，アノテーションタスクの平均時間を求める
def calculate_average_time_of_each_task(directory_path):
    query = query_of_completed_worker(directory_path)
    mturk_outputs = pd.read_csv(directory_path + '/mturk.output.csv',
                                dtype={'input_image_url': 'str', 'answer_annotation_data': 'str'}).query(query)
    bounding_tasks_outputs = mturk_outputs.query('input_task_type == "b"').sort_values('work_time')
    corrected_bounding_tasks_outputs = mturk_outputs.query(
        'input_task_type == "b" and 5000 < work_time < 300000').sort_values(
        'work_time')
    tagging_tasks_outputs = mturk_outputs.query('input_task_type == "t"').sort_values('work_time')
    corrected_tagging_tasks_outputs = mturk_outputs.query(
        'input_task_type == "t" and 5000 < work_time < 300000').sort_values(
        'work_time')
    print(bounding_tasks_outputs.describe())
    print(corrected_bounding_tasks_outputs.describe())
    print(tagging_tasks_outputs.describe())
    print(corrected_tagging_tasks_outputs.describe())

    bounding_tasks_outputs['work_time'].hist(bins=30, alpha=0.8, label='Bounding task')
    tagging_tasks_outputs['work_time'].hist(bins=30, alpha=0.8, label='Annotation task')
    plt.xlabel('work_time(ms)')
    plt.ylabel('number_of_tasks')
    plt.legend()
    plt.savefig('graphs/average_time_of_each_task.jpg')
    plt.figure()

    corrected_bounding_tasks_outputs['work_time'].hist(bins=30, alpha=0.8, label='Bounding task')
    corrected_tagging_tasks_outputs['work_time'].hist(bins=30, alpha=0.8, label='Annotation task')
    plt.xlabel('work_time(ms)')
    plt.ylabel('number_of_tasks')
    plt.legend()
    plt.savefig('graphs/corrected_average_time_of_each_task.jpg')


# ワーカがいつ辞めたかをグラフ化
def show_times_when_worker_exit(filepath):
    mturk_workerinformation = pd.read_csv(filepath, dtype={'input_image_url': 'str', 'answer_annotation_data': 'str'})
    mturk_workerinformation_A = mturk_workerinformation.query('task_condition == "A"').sort_values(
        'number_of_completed_tasks')
    mturk_workerinformation_B = mturk_workerinformation.query('task_condition == "B"').sort_values(
        'number_of_completed_tasks')
    mturk_workerinformation_C = mturk_workerinformation.query('task_condition == "C"').sort_values(
        'number_of_completed_tasks')
    mturk_workerinformation_D = mturk_workerinformation.query('task_condition == "D"').sort_values(
        'number_of_completed_tasks')
    mturk_workerinformation_E = mturk_workerinformation.query('task_condition == "E"').sort_values(
        'number_of_completed_tasks')
    mturk_workerinformation_F = mturk_workerinformation.query('task_condition == "F"').sort_values(
        'number_of_completed_tasks')

    # mturk_workerinformation_A['number_of_completed_tasks'].hist(bins=120, alpha=0.8, label='A', align='right')
    mturk_workerinformation_B['number_of_completed_tasks'].hist(bins=120, alpha=0.8, label='B', align='right')
    plt.hist([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], bins=120, alpha=0.8, label='B', align='right')
    # mturk_workerinformation_C['number_of_completed_tasks'].hist(bins=120, alpha=0.8, label='C', align='right')
    # mturk_workerinformation_D['number_of_completed_tasks'].hist(bins=120, alpha=0.8, label='D', align='right')
    # mturk_workerinformation_E['number_of_completed_tasks'].hist(bins=100, alpha=0.8, label='E', align='right')
    # mturk_workerinformation_F['number_of_completed_tasks'].hist(bins=100, alpha=0.8, label='F', align='right')

    # plt.xlabel('work_time(ms)')
    # plt.ylabel('number_of_tasks')
    plt.legend()
    plt.show()


# tagging_taskの正解を作成する関数
def make_answer_of_tagging_task(directory_path):
    query = query_of_completed_worker(directory_path)
    mturk_outputs = pd.read_csv(directory_path + '/mturk.output.csv',
                                dtype={'input_image_url': 'str', 'answer_annotation_data': 'str'}).query(query).query(
        'input_task_type == "t"')

    df_answer = pd.DataFrame(index=[], columns=mturk_outputs.columns)
    for index in range(1, 101):
        targets = mturk_outputs.query('input_image_url == "{:03}"'.format(index))
        s_answer = pd.Series(index=mturk_outputs.columns, name=index)
        s_answer['input_task_type'] = 't'
        s_answer['input_image_url'] = '{:03}'.format(index)
        s_answer['answer_text_box1'] = targets['answer_text_box1'].value_counts().index[0]
        s_answer['answer_text_box2'] = targets['answer_text_box2'].value_counts().index[0]
        s_answer['answer_text_box3'] = targets['answer_text_box3'].value_counts().index[0]
        s_answer['answer_text_box4'] = targets['answer_text_box4'].value_counts().index[0]
        s_answer['answer_text_box5'] = targets['answer_text_box5'].value_counts().index[0]
        df_answer = df_answer.append(s_answer)
    df_answer.to_csv(directory_path + '/answers_of_tagging_task.csv', index=False)


# tagging taskの結果を確認
def show_tagging_task_results(directory_path):
    answers = pd.read_csv(directory_path + '/answers_of_tagging_task.csv',
                          dtype={'input_image_url': 'str', 'answer_annotation_data': 'str'})
    for index, answer in answers.iterrows():
        text = ''
        for i in range(1, 6):
            text += str(i) + '"' + answer['answer_text_box' + str(i)] + '" '
        print('image_url:' + answer['input_image_url'])
        print(text)
        im = Image.open('./resources/images/tagging_images/' + answer['input_image_url'] + '.jpg')
        plt.imshow(im)
        plt.show()


# bounding taskの結果を確認
# is_eachは一つずつ見るかどうか
def show_bounding_task_results(directory_path, is_each):
    answers = pd.read_csv(directory_path + '/answers_of_bounding_task.csv',
                          dtype={'input_image_url': 'str', 'answer_annotation_data': 'str'})
    for index, answer in answers.iterrows():
        if (not answer.isnull()['answer_annotation_data']
                and not answer['answer_annotation_data'] == '[]'):
            sorted_bounding_box_dic_list = data_processor.make_sorted_bounding_box_dic_list(
                answer['answer_annotation_data'])
            print(answer['input_image_url'] + ':' + str(len(sorted_bounding_box_dic_list)))
            if is_each:
                for sorted_bounding_box_dic in sorted_bounding_box_dic_list:
                    if answer['input_image_url'] == '084':
                        view_images.from_answer_dic_and_target_dic(sorted_bounding_box_dic, sorted_bounding_box_dic,
                                                                   answer['input_image_url'])
            else:
                view_images.from_bounding_box_dic_list(answer['input_image_url'], sorted_bounding_box_dic_list)


# 矩形が重なった部分の面積を求める
# 重なり部分の面積 / 大きい方の四角形の面積を精度とする
def calculate_accuracy_of_bounding_box(bounding_box_dic_of_answer, bounding_box_dic_of_target, debug=False):
    answer_left = bounding_box_dic_of_answer['left']
    answer_top = bounding_box_dic_of_answer['top']
    answer_width = bounding_box_dic_of_answer['width']
    answer_height = bounding_box_dic_of_answer['height']
    answer_right = answer_left + answer_width
    answer_under = answer_top + answer_height
    answer_area = answer_width * answer_height

    target_left = bounding_box_dic_of_target['left']
    target_top = bounding_box_dic_of_target['top']
    target_width = bounding_box_dic_of_target['width']
    target_height = bounding_box_dic_of_target['height']
    target_right = target_left + target_width
    target_under = target_top + target_height
    target_area = target_width * target_height

    if answer_left > target_right or answer_right < target_left or answer_under < target_top or answer_top > answer_under:
        accuracy_rate = 0
    else:
        overlapping_left = max(answer_left, target_left)
        overlapping_top = max(answer_top, target_top)
        overlapping_right = min(answer_left + answer_width, target_left + target_width)
        overlapping_under = min(answer_top + answer_height, target_top + target_height)
        overlapping_area = max((overlapping_right - overlapping_left) * (overlapping_under - overlapping_top), 0)
        larger_area = max(answer_area, target_area)
        accuracy_rate = overlapping_area / larger_area

        if debug:
            print('answer_left  :' + str(answer_left))
            print('answer_top   :' + str(answer_top))
            print('answer_width :' + str(answer_width))
            print('answer_height:' + str(answer_height))
            print('answer_area  :' + str(answer_area))
            print('target_left  :' + str(target_left))
            print('target_top   :' + str(target_top))
            print('target_width :' + str(target_width))
            print('target_height:' + str(target_height))
            print('target_area  :' + str(target_area))
            print('overlapping_left :' + str(overlapping_left))
            print('overlapping_top  :' + str(overlapping_top))
            print('overlapping_right:' + str(overlapping_right))
            print('overlapping_under:' + str(overlapping_under))
            print('overlapping_area :' + str(overlapping_area))
            print('larger_area  :' + str(larger_area))
            print('accuracy_rate:' + str(accuracy_rate))

    return accuracy_rate


# 精度を求める関数
def calculate_accuracy(directory_path):
    query = query_of_completed_worker(directory_path)
    mturk_outputs = pd.read_csv(directory_path + '/mturk.output.csv',
                                dtype={'input_image_url': 'str', 'answer_annotation_data': 'str'}).query(query)

    answers_of_tagging_task = pd.read_csv(directory_path + '/answers_of_tagging_task.csv',
                                          dtype={'input_image_url': 'str', 'answer_annotation_data': 'str'})
    answers_of_bounding_task = pd.read_csv(directory_path + '/answers_of_bounding_task.csv',
                                           dtype={'input_image_url': 'str', 'answer_annotation_data': 'str'})

    task_conditions = ['A', 'B', 'C', 'D', 'E', 'F']
    bounding_accuracy_rate = []
    corrected_bounding_accuracy_rate = []
    tagging_accuracy_rate = []
    corrected_tagging_accuracy_rate = []

    # 条件毎の精度のリストを持つ
    bounding_accuracy_rate_dic = {}
    tagging_accuracy_rate_dic = {}
    # タスク条件毎に処理
    for task_condition in task_conditions:
        mturk_outputs_each_condition = mturk_outputs.query('input_task_condition == "{}"'.format(task_condition))
        count_of_tagging_task = 0
        corrected_count_of_tagging_task = 0
        count_of_bounding_task = 0
        corrected_count_of_bounding_task = 0
        sum_of_accuracy_rate_of_tagging = 0
        corrected_sum_of_accuracy_rate_of_tagging = 0
        sum_of_accuracy_rate_of_bounding = 0
        corrected_sum_of_accuracy_rate_of_bounding = 0
        bounding_accuracy_rate_of_each_condition_list = []
        tagging_accuracy_rate_of_each_condition_list = []
        for index, mturk_output in mturk_outputs_each_condition.iterrows():
            if mturk_output['input_task_type'] == 't':
                answer = \
                    answers_of_tagging_task.query(
                        'input_image_url == "{}"'.format(mturk_output['input_image_url'])).iloc[0]
                number_of_correct, answer_input, target_input = compare_when_tag_task(answer, mturk_output)
                accuracy_rate = number_of_correct / 5
                sum_of_accuracy_rate_of_tagging += accuracy_rate
                count_of_tagging_task += 1
                if accuracy_rate > 0.5:
                    corrected_sum_of_accuracy_rate_of_tagging += accuracy_rate
                    tagging_accuracy_rate_of_each_condition_list.append(accuracy_rate)
                    corrected_count_of_tagging_task += 1
            else:
                answer = \
                    answers_of_bounding_task.query(
                        'input_image_url == "{}"'.format(mturk_output['input_image_url'])).iloc[0]
                accuracy_rate = compare_when_bounding_task(answer, mturk_output)

                sum_of_accuracy_rate_of_bounding += accuracy_rate
                count_of_bounding_task += 1
                if accuracy_rate > 0.5:
                    corrected_sum_of_accuracy_rate_of_bounding += accuracy_rate
                    bounding_accuracy_rate_of_each_condition_list.append(accuracy_rate)
                    corrected_count_of_bounding_task += 1

        print(task_condition)
        average_accuracy_rate_of_tagging = sum_of_accuracy_rate_of_tagging / count_of_tagging_task if count_of_tagging_task else 0
        corrected_average_accuracy_rate_of_tagging = corrected_sum_of_accuracy_rate_of_tagging / corrected_count_of_tagging_task if corrected_count_of_tagging_task else 0
        average_accuracy_rate_of_bounding = sum_of_accuracy_rate_of_bounding / count_of_bounding_task if count_of_bounding_task else 0
        corrected_average_accuracy_rate_of_bounding = corrected_sum_of_accuracy_rate_of_bounding / corrected_count_of_bounding_task if corrected_count_of_bounding_task else 0
        print('Bounding         :' + str(average_accuracy_rate_of_bounding))
        print('Bounding(c)      :' + str(corrected_average_accuracy_rate_of_bounding))
        print('Annotation       :' + str(average_accuracy_rate_of_tagging))
        print('Annotation(c)    :' + str(corrected_average_accuracy_rate_of_tagging))
        bounding_accuracy_rate.append(average_accuracy_rate_of_bounding)
        corrected_bounding_accuracy_rate.append(corrected_average_accuracy_rate_of_bounding)
        tagging_accuracy_rate.append(average_accuracy_rate_of_tagging)
        corrected_tagging_accuracy_rate.append(corrected_average_accuracy_rate_of_tagging)
        bounding_accuracy_rate_dic[task_condition] = bounding_accuracy_rate_of_each_condition_list
        tagging_accuracy_rate_dic[task_condition] = tagging_accuracy_rate_of_each_condition_list

    print(bounding_accuracy_rate)
    print(corrected_bounding_accuracy_rate)
    print(tagging_accuracy_rate)
    print(corrected_tagging_accuracy_rate)
    # print(bounding_accuracy_rate_dic)
    # print(tagging_accuracy_rate_dic)
    plt.bar([1, 2, 3, 4, 5, 6], corrected_bounding_accuracy_rate, color='#E24A33', width=0.4,
            label='Bounding task(corrected)')
    plt.bar([1, 2, 3, 4, 5, 6], bounding_accuracy_rate, color='#338ABD', width=0.4, label='Bounding task')
    plt.bar([1.4, 2.4, 3.4, 4.4, 5.4, 6.4], corrected_tagging_accuracy_rate, color='pink', width=0.4,
            label='Annotation task(corrected)')
    plt.bar([1.4, 2.4, 3.4, 4.4, 5.4, 6.4], tagging_accuracy_rate, color='#00E3ED', width=0.4, label='Annotation task')
    plt.xticks([1.2, 2.2, 3.2, 4.2, 5.2, 6.2], task_conditions)
    plt.legend(loc='lower left')
    plt.savefig('graphs/accuracy_rate')

    return bounding_accuracy_rate_dic, tagging_accuracy_rate_dic


# 渡されたdfに対して精度を求める関数
def calculate_accuracy_of_df(targets, answers_of_tagging_task, answers_of_bounding_task):
    count_of_tagging_task = 0
    count_of_bounding_task = 0

    sum_of_accuracy_rate_of_tagging = 0
    sum_of_accuracy_rate_of_bounding = 0
    for index, mturk_output in targets.iterrows():
        if mturk_output['input_task_type'] == 't':
            answer = \
                answers_of_tagging_task.query(
                    'input_image_url == "{}"'.format(mturk_output['input_image_url'])).iloc[0]
            number_of_correct, answer_input, target_input = compare_when_tag_task(answer, mturk_output)
            accuracy_rate = number_of_correct / 5
            sum_of_accuracy_rate_of_tagging += accuracy_rate
            count_of_tagging_task += 1
        else:
            answer = \
                answers_of_bounding_task.query(
                    'input_image_url == "{}"'.format(mturk_output['input_image_url'])).iloc[0]
            accuracy_rate = compare_when_bounding_task(answer, mturk_output)

            sum_of_accuracy_rate_of_bounding += accuracy_rate
            count_of_bounding_task += 1

    average_accuracy_rate_of_tagging = sum_of_accuracy_rate_of_tagging / count_of_tagging_task if count_of_tagging_task else 0
    average_accuracy_rate_of_bounding = sum_of_accuracy_rate_of_bounding / count_of_bounding_task if count_of_bounding_task else 0

    return average_accuracy_rate_of_tagging, average_accuracy_rate_of_bounding


# タスク完了数のヒストグラムを描画
def show_hist_of_number_of_completed_tasks(directory_path, is_all, is_ef):
    query = query_of_completed_worker(directory_path)

    mturk_worker_information = pd.read_csv(directory_path + '/mturk.workerinformation.csv',
                                           dtype={'input_image_url': 'str', 'answer_annotation_data': 'str'}).query(
        query)

    if is_all:
        print(mturk_worker_information['number_of_completed_tasks'].describe())
        mturk_worker_information['number_of_completed_tasks'].hist(bins=24)
    elif is_ef:
        bounding_information = mturk_worker_information.query('task_condition == "E"')
        annotation_information = mturk_worker_information.query('task_condition == "F"')

        print(bounding_information['number_of_completed_tasks'].describe())
        print(annotation_information['number_of_completed_tasks'].describe())

        bounding_information['number_of_completed_tasks'].hist(bins=10, alpha=0.5, label='Bounding')
        annotation_information['number_of_completed_tasks'].hist(bins=10, alpha=0.5, label='Annotation')
    else:
        switching_information = mturk_worker_information.query('task_condition == "A"')

        print(switching_information['number_of_completed_tasks'].map(lambda x: x-1).describe())

        switching_information['number_of_completed_tasks'].hist(bins=10, alpha=0.5, label='Switching')

    plt.legend()
    plt.show()


# タスク完了数の分散分析を行う
def analysis_of_variance(directory_path):
    print('analysis')
    query = query_of_completed_worker(directory_path)

    mturk_worker_information = pd.read_csv(directory_path + '/mturk.workerinformation.csv',
                                           dtype={'input_image_url': 'str', 'answer_annotation_data': 'str'}).query(
        query)

    mturk_worker_information = mturk_worker_information.query('number_of_completed_tasks > 11')

    result_A = mturk_worker_information.query('task_condition == "A"')['number_of_completed_tasks'].reset_index(
        drop=True)
    result_B = mturk_worker_information.query('task_condition == "B"')['number_of_completed_tasks'].reset_index(
        drop=True)
    result_C = mturk_worker_information.query('task_condition == "C"')['number_of_completed_tasks'].reset_index(
        drop=True)
    result_D = mturk_worker_information.query('task_condition == "D"')['number_of_completed_tasks'].reset_index(
        drop=True)
    result_E = mturk_worker_information.query('task_condition == "E"')['number_of_completed_tasks'].reset_index(
        drop=True)
    result_F = mturk_worker_information.query('task_condition == "F"')['number_of_completed_tasks'].reset_index(
        drop=True)

    result_A.name = 'A'
    result_B.name = 'B'
    result_C.name = 'C'
    result_D.name = 'D'
    result_E.name = 'E'
    result_F.name = 'F'
    # print('result_A')
    # print(result_A)
    # print('result_B')
    # print(result_B)
    # print('result_C')
    # print(result_C)
    # print('result_D')
    # print(result_D)
    # print('result_E')
    # print(result_E)
    # print('result_F')
    # print(result_F)

    # df_h = pd.concat([result_A, result_B, result_C, result_D, result_E, result_F], axis=1)
    # print(df_h)
    # df_h.to_csv('table.csv', index=False)

    print(stats.f_oneway(result_A, result_B, result_C, result_D, result_E, result_F))


# タスク精度の分散分析を行う
def analysis_of_variance_of_accuracy(directory_path):
    bounding_accuracy_rate_dic, tagging_accuracy_rate_dic = calculate_accuracy(directory_path)

    bounding_A = pd.Series(bounding_accuracy_rate_dic['A'], name='A')
    print(bounding_A.describe())
    bounding_B = pd.Series(bounding_accuracy_rate_dic['B'], name='B')
    print(bounding_B.describe())
    bounding_C = pd.Series(bounding_accuracy_rate_dic['C'], name='C')
    print(bounding_C.describe())
    bounding_D = pd.Series(bounding_accuracy_rate_dic['D'], name='D')
    print(bounding_D.describe())
    bounding_E = pd.Series(bounding_accuracy_rate_dic['E'], name='E')
    print(bounding_E.describe())

    tagging_A = pd.Series(tagging_accuracy_rate_dic['A'], name='A')
    print(tagging_A.describe())
    tagging_B = pd.Series(tagging_accuracy_rate_dic['B'], name='B')
    print(tagging_B.describe())
    tagging_C = pd.Series(tagging_accuracy_rate_dic['C'], name='C')
    print(tagging_C.describe())
    tagging_D = pd.Series(tagging_accuracy_rate_dic['D'], name='D')
    print(tagging_D.describe())
    tagging_F = pd.Series(tagging_accuracy_rate_dic['F'], name='F')
    print(tagging_F.describe())

    print(stats.f_oneway(bounding_A, bounding_B,
                         bounding_C, bounding_D,
                         bounding_E))
    print(stats.f_oneway(tagging_A, tagging_B,
                         tagging_C, tagging_D,
                         tagging_F))
    bounding_df = pd.concat([bounding_A, bounding_B, bounding_C, bounding_D, bounding_E], axis=1)
    tagging_df = pd.concat([tagging_A, tagging_B, tagging_C, tagging_D, tagging_F], axis=1)

    bounding_df.to_csv('bounding.csv', index=False)
    tagging_df.to_csv('tagging.csv', index=False)


# スイッチタスクを抽出する．
# task_conditionとtask_orderのペアを返す
# (task_condition, task_order)
def extract_switch_tasks(directory_path):
    mturk_intputs = pd.read_csv(directory_path + '/mturk.input.csv').sort_values('id')
    task_conditions = mturk_intputs['task_condition'].unique()

    switch_tasks_pairs = []
    for task_condition in task_conditions:
        each_condition_mturk_intputs = mturk_intputs.query('task_condition == "{}"'.format(task_condition))

        previous_task_type = each_condition_mturk_intputs['task_type'].iloc[0]
        for index, mturk_intput in each_condition_mturk_intputs.iterrows():
            if previous_task_type != mturk_intput['task_type']:
                switch_tasks_pairs.append((task_condition, mturk_intput['task_order']))
            previous_task_type = mturk_intput['task_type']

    return switch_tasks_pairs


# スイッチコストを計算する．
def evaluate_switch_cost(directory_path):
    switch_tasks_pairs = extract_switch_tasks(directory_path)

    # HITを完了したユーザのデータのみを抽出
    worker_ids = data_processor.extract_worker_who_complete_task(directory_path)

    # 質の悪いワーカ(関数を作成して求めた)
    bad_workers = ['A10IMBRVQEXAF4', 'A122RVIUXTKC9I', 'A162MP4ZCKL5M9', 'A18P7QD3PUA4YM', 'A191V8LNTTLHSA',
                   'A19VERTZEKA3HH', 'A19XD2OZXHREZ', 'A1DKO0W1JQDJVA', 'A1DVTB61111IIU', 'A1HYA9IORBCUW6',
                   'A1I5Y52TDMIOT6', 'A1KILWESE344I6', 'A1LTNU795MKL2X', 'A1QPS126EE16WP', 'A1SNC8UL8YFRH5',
                   'A1SU9OWAXZZMM', 'A1TMP6T8VCEJDR', 'A1ZPOPMHVMC56I', 'A209ROKVKHDRM7', 'A20HWQ5XFW3WSC',
                   'A22A52CE826927', 'A236X6JLGMZ9V9', 'A23FIUOUMR9Y2L', 'A24H043G3L9O8G', 'A27G47ZW5METJY',
                   'A27J8EOAZMU0RM', 'A27RKO0WSSLC3Y', 'A28EPNOB37T7AO', 'A2A73VNVENIN7M', 'A2CI15LSIGRMLI',
                   'A2D4X82TOMF0F9', 'A2H321P1NSEW35', 'A2HV8VJ3BWPO8W', 'A2JA19C4Y5JZD3', 'A2L72UWIG81HPT',
                   'A2P1KI42CJVNIA', 'A2PMUFPD9HV5CL', 'A2PQV1A2D4CG52', 'A2TUUIV61CR0C7', 'A34HSPNUYSHSR',
                   'A36GOXSROXWG3B', 'A371HJLJ4JGQWA', 'A3FUQIL43AOA9H', 'A3HBLK46AT4IR2', 'A3N9RE9CMZMYTH',
                   'A3O8YXPG1HGUH0', 'A3P0LQCNIXWOVJ', 'A3R7VBZJYCEIMZ', 'A3S392MV6HWTTE', 'A3S7CS4LVZL5ZK',
                   'A3TNWLKK5PAQF9', 'A3UQX3NQ7HMS62', 'A5STK3PEW58OV', 'A8MGW12LQD1C2', 'A8P2CCZS7ZBQP',
                   'AA5M80LZX2Y7E', 'AD1ILDUXZHASF', 'ADX0Z39YKQY99', 'AECIP5QWNEUJJ', 'AF1M43ALWWKIT', 'AF7EW2KB066NQ',
                   'AMCJQTBCGX9R5', 'AN738HXJ1TH8N', 'ANBTNIM08969Z', 'ARP946TJY4INO', 'ATIGCUKIPNWG0', 'ATL3WW9MDX62M',
                   'AYUUIIW2UK75I']

    # 質の悪いワーカを除去
    worker_ids = list(set(worker_ids) - set(bad_workers))

    query = ''
    for worker_id in worker_ids:
        query += 'worker_id == "{}"'.format(worker_id) + ' or '
    query = query[:-4]
    mturk_outputs = pd.read_csv(directory_path + '/mturk.output.csv',
                                dtype={'input_image_url': 'str', 'answer_annotation_data': 'str'}).query(query)

    # 作業時間で抽出
    mturk_outputs = mturk_outputs.query('5000 < work_time < 60000')

    # A_mturk_outputs = mturk_outputs.query('input_task_condition == "A" and input_task_type == "b"')
    # tA_mturk_outputs = mturk_outputs.query('input_task_condition == "A" and input_task_type == "t"')
    # B_mturk_outputs = mturk_outputs.query('input_task_condition == "B" and input_task_type == "b"')
    # tB_mturk_outputs = mturk_outputs.query('input_task_condition == "B" and input_task_type == "t"')
    # C_mturk_outputs = mturk_outputs.query('input_task_condition == "C" and input_task_type == "b"')
    # tC_mturk_outputs = mturk_outputs.query('input_task_condition == "C" and input_task_type == "t"')
    # D_mturk_outputs = mturk_outputs.query('input_task_condition == "D" and input_task_type == "b"')
    # tD_mturk_outputs = mturk_outputs.query('input_task_condition == "D" and input_task_type == "t"')
    # E_mturk_outputs = mturk_outputs.query('input_task_condition == "E" and input_task_type == "b"')
    # tE_mturk_outputs = mturk_outputs.query('input_task_condition == "E" and input_task_type == "t"')
    # F_mturk_outputs = mturk_outputs.query('input_task_condition == "F" and input_task_type == "b"')
    # tF_mturk_outputs = mturk_outputs.query('input_task_condition == "F" and input_task_type == "t"')
    # print(A_mturk_outputs.describe())
    # # print(tA_mturk_outputs.describe())
    # print(B_mturk_outputs.describe())
    # # print(tB_mturk_outputs.describe())
    # print(C_mturk_outputs.describe())
    # # print(tC_mturk_outputs.describe())
    # print(D_mturk_outputs.describe())
    # # print(tD_mturk_outputs.describe())
    # print(E_mturk_outputs.describe())
    # # print(tF_mturk_outputs.describe())
    # A_mturk_outputs['work_time'].hist(bins=10, alpha=0.5, density=True, label='A')
    # # tA_mturk_outputs['work_time'].hist(bins=10, alpha=0.5, density=True, label='Bounding')
    # B_mturk_outputs['work_time'].hist(bins=10, alpha=0.5, density=True, label='B')
    # # tB_mturk_outputs['work_time'].hist(bins=10, alpha=0.5, density=True, label='Bounding')
    # C_mturk_outputs['work_time'].hist(bins=10, alpha=0.5, density=True, label='C')
    # # tC_mturk_outputs['work_time'].hist(bins=10, alpha=0.5, density=True, label='Bounding')
    # # D_mturk_outputs['work_time'].hist(bins=10, alpha=0.5, density=True, label='D')
    # # tD_mturk_outputs['work_time'].hist(bins=10, alpha=0.5, density=True, label='Bounding')
    # E_mturk_outputs['work_time'].hist(bins=10, alpha=0.5, density=True, label='E')
    # # tF_mturk_outputs['work_time'].hist(bins=10, alpha=0.5, density=True, label='Bounding')
    # plt.legend()
    # plt.show()

    switch_query = ''
    for switch_tasks_pair in switch_tasks_pairs:
        switch_query += '(input_task_condition == "{}" and input_task_order == {})'.format(switch_tasks_pair[0],
                                                                                           switch_tasks_pair[
                                                                                               1]) + ' or '
    switch_query = switch_query[:-4]

    no_switch_query = ''
    for switch_tasks_pair in switch_tasks_pairs:
        no_switch_query += '(input_task_condition != "{}" or input_task_order != {})'.format(switch_tasks_pair[0],
                                                                                             switch_tasks_pair[
                                                                                                 1]) + ' and '
    no_switch_query = no_switch_query[:-5]

    switch_outputs = mturk_outputs.query(switch_query)
    no_switch_outputs = mturk_outputs.query(no_switch_query)

    tagging_switch_outputs = switch_outputs.query('input_task_type == "t"')
    tagging_no_switch_outputs = no_switch_outputs.query('input_task_type == "t"')
    bounding_switch_outputs = switch_outputs.query('input_task_type == "b"')
    bounding_no_switch_outputs = no_switch_outputs.query('input_task_type == "b"')
    print(tagging_switch_outputs.describe())
    print(tagging_no_switch_outputs.describe())
    print(bounding_switch_outputs.describe())
    print(bounding_no_switch_outputs.describe())


# 質の悪いワーカを抽出
def extract_bad_workers(directory_path):
    answers_of_tagging_task = pd.read_csv(directory_path + '/answers_of_tagging_task.csv',
                                          dtype={'input_image_url': 'str', 'answer_annotation_data': 'str'})
    answers_of_bounding_task = pd.read_csv(directory_path + '/answers_of_bounding_task.csv',
                                           dtype={'input_image_url': 'str', 'answer_annotation_data': 'str'})

    # HITを完了したユーザのデータのみを抽出
    worker_ids = data_processor.extract_worker_who_complete_task(directory_path)
    bad_workers = []
    for worker_id in worker_ids:
        query = 'worker_id == "{}"'.format(worker_id)
        mturk_outputs = pd.read_csv(directory_path + '/mturk.output.csv',
                                    dtype={'input_image_url': 'str', 'answer_annotation_data': 'str'}).query(query)
        average_accuracy_rate_of_tagging, average_accuracy_rate_of_bounding = calculate_accuracy_of_df(mturk_outputs,
                                                                                                       answers_of_tagging_task,
                                                                                                       answers_of_bounding_task)
        if (average_accuracy_rate_of_tagging != 0 and average_accuracy_rate_of_tagging < 0.8) or (
                average_accuracy_rate_of_bounding != 0 and average_accuracy_rate_of_bounding < 0.9):
            bad_workers.append(worker_id)

    print(bad_workers)


# HITを完了したユーザのデータのみを抽出
def query_of_completed_worker(directory_path):
    worker_ids = data_processor.extract_worker_who_complete_task(directory_path)
    query = ''
    for worker_id in worker_ids:
        query += 'worker_id == "{}"'.format(worker_id) + ' or '
    query = query[:-4]

    return query


# ワーカが停止する確率fを求める
def show_f_of_workers_exit(directory_path, do_complemented):
    query = query_of_completed_worker(directory_path)

    mturk_worker_information = pd.read_csv(directory_path + '/mturk.workerinformation.csv',
                                           dtype={'input_image_url': 'str', 'answer_annotation_data': 'str'}).query(
        query)
    bounding_information = mturk_worker_information.query('9 < number_of_completed_tasks <= 90')
    annotation_information = mturk_worker_information.query('13 < number_of_completed_tasks <= 90')

    bounding_number_of_completed_tasks_list = \
        bounding_information.query('task_condition == "E"').sort_values('number_of_completed_tasks')[
            'number_of_completed_tasks']
    annotation_number_of_completed_tasks_list = \
        annotation_information.query('task_condition == "F"').sort_values('number_of_completed_tasks')[
            'number_of_completed_tasks']

    bounding_probability_tuple_list = []
    annotation_probability_tuple_list = []

    count = 1
    for bounding_number_of_completed_tasks in bounding_number_of_completed_tasks_list:
        probability = count / len(bounding_number_of_completed_tasks_list)
        bounding_probability_tuple_list.append((bounding_number_of_completed_tasks, probability))
        count += 1

    count = 1
    for annotation_number_of_completed_tasks in annotation_number_of_completed_tasks_list:
        probability = count / len(annotation_number_of_completed_tasks_list)
        annotation_probability_tuple_list.append((annotation_number_of_completed_tasks, probability))
        count += 1

    print('bounding_probability_tuple_list:{}'.format(bounding_probability_tuple_list))
    print('annotation_probability_tuple_list:{}'.format(annotation_probability_tuple_list))
    if do_complemented:
        # 確率を線形に補完する
        # 同じタスク数の人が何人かいる場合，最大の確率を取る．
        bounding_probability_tuple_list = [(k, v) for (k, v) in bounding_probability_tuple_list if
                                           v == max([t for (x, t) in bounding_probability_tuple_list if x == k])]
        bounding_probability_list = [0]
        previous_position = 0
        previous_value = 0
        for (key, value) in bounding_probability_tuple_list:
            position = key
            for index in range(previous_position + 1, position + 1):
                probability = previous_value + (value - previous_value) / (position - previous_position) * (
                        index - previous_position)
                bounding_probability_list.append(probability)
            previous_position = position
            previous_value = value
        for index in range(previous_position, 100):
            bounding_probability_list.append(1)

        # 確率を線形に補完する
        annotation_probability_tuple_list = [(k, v) for (k, v) in annotation_probability_tuple_list if
                                             v == max([t for (x, t) in annotation_probability_tuple_list if x == k])]
        annotation_probability_list = [0]
        previous_position = 0
        previous_value = 0
        for (key, value) in annotation_probability_tuple_list:
            position = key
            for index in range(previous_position + 1, position + 1):
                probability = previous_value + (value - previous_value) / (position - previous_position) * (
                        index - previous_position)
                annotation_probability_list.append(probability)
            previous_position = position
            previous_value = value
        for index in range(previous_position + 1, 101):
            annotation_probability_list.append(1)

        print(bounding_probability_list)
        print(annotation_probability_list)

        plt.plot([s for s in range(101)], bounding_probability_list, marker='o', label='Bounding')
        plt.plot([s for s in range(101)], annotation_probability_list, marker='o', label='Annotation')
        plt.legend()
        plt.show()
    else:
        plt.plot([x[0] for x in bounding_probability_tuple_list], [x[1] for x in bounding_probability_tuple_list],
                 marker='o',
                 label='Bounding')
        plt.plot([x[0] for x in annotation_probability_tuple_list], [x[1] for x in annotation_probability_tuple_list],
                 marker='o',
                 label='Annotation')
        plt.xlabel('number_of_repetition')
        plt.ylabel('probability_of_stop')
        plt.legend()
        plt.show()


# ワーカが停止する確率gを求める
def show_g_of_workers_exit(directory_path, do_complemented):
    query = query_of_completed_worker(directory_path)

    mturk_worker_information = pd.read_csv(directory_path + '/mturk.workerinformation.csv',
                                           dtype={'input_image_url': 'str', 'answer_annotation_data': 'str'}).query(
        query)
    mturk_worker_information = mturk_worker_information.query('11 < number_of_completed_tasks <= 90')

    task_conditions = ['A']
    switching_times_list = []
    for task_condition in task_conditions:
        number_of_completed_tasks_list = list(
            mturk_worker_information.query('task_condition == "{}"'.format(task_condition)).sort_values(
                'number_of_completed_tasks')['number_of_completed_tasks'])

        for number_of_completed_tasks in number_of_completed_tasks_list:
            if task_condition == 'A':
                switching_times = number_of_completed_tasks - 1
            elif task_condition == 'B':
                switching_times = number_of_completed_tasks // 5
            elif task_condition == 'C':
                switching_times = number_of_completed_tasks // 10
            elif task_condition == 'D':
                switching_times = number_of_completed_tasks // 20
            else:
                switching_times = -999999
            switching_times_list.append(switching_times)
    switching_times_list.sort()

    probability_tuple_list = []

    count = 1
    for switching_times in switching_times_list:
        probability = count / len(switching_times_list)
        probability_tuple_list.append((switching_times, probability))
        count += 1

    if do_complemented:
        # 同じタスク数の人が何人かいる場合，最大の確率を取る．
        probability_tuple_list = [(k, v) for (k, v) in probability_tuple_list if
                                  v == max([t for (x, t) in probability_tuple_list if x == k])]

        # # 確率を線形に補完する
        probability_list = [0]
        previous_position = 0
        previous_p = 0
        # n:タスク数，p:確率
        for (n, p) in probability_tuple_list:
            position = n
            for index in range(previous_position + 1, position + 1):
                probability = previous_p + (p - previous_p) / (position - previous_position) * (
                        index - previous_position)
                probability_list.append(probability)
            previous_position = position
            previous_p = p
        print(probability_list)

        plt.plot([s for s in range(len(probability_list))], probability_list, marker='o', label='Bounding')
        plt.legend()
        plt.show()
    else:
        print(probability_tuple_list)
        plt.plot([x[0] for x in probability_tuple_list], [x[1] for x in probability_tuple_list], marker='o')
        plt.xlabel('number_of_switching')
        plt.ylabel('probability_of_stop')
        plt.show()


if __name__ == '__main__':
    # show_detail('resources/evaluated_results_directory/evaluated_results_of_' + '30ZX6P7VF8VKTCO1QI1AWI4JHLZ2JX' + '.csv')
    # # 真面目なワーカ
    # show_detail('evaluated_results_of_first_experiment_directory/evaluated_results_of_A17T285YLOZKZQ.csv')
    # show_detail('resources/experiment2/evaluated_results_directory/evaluated_results.csv', 'experiment2')
    # show_detail('resources/main_experiment_results/evaluated_results_directory/evaluated_results_of_okino.csv')
    # calculate_average_time()
    # extract_results_of_multiples_of_10('resources/main_experiment(bounding_box)', 'A13BWRE4H0ADIQ')
    # calculate_average_number_of_completed_tasks('resources/main_experiment_mixed')
    # calculate_average_number_of_completed_tasks('resources/main_experiment')
    # calculate_average_number_of_completed_tasks('resources/main_experiment(bounding_box)')

    # # あるワーカの結果を確認
    # show_result_from_mturk_outputs('resources/main_experiment_additional', 'A122RVIUXTKC9I')

    # # タスク条件毎の，ワーカ一人当たりの平均タスク完了数
    # calculate_average_number_of_completed_tasks('resources/main_experiment_additional')
    # # タスク毎の平均時間を算出
    # calculate_average_time_of_each_task('resources/main_experiment_additional')
    # ワーカがいつ辞めたかをグラフ化
    # show_times_when_worker_exit('resources/main_experiment_additional/mturk.workerinformation.csv')
    # tagging_taskの正解を作成する関数
    # make_answer_of_tagging_task('resources/main_experiment_additional')
    # tagging taskの結果を確認
    # show_tagging_task_results('resources/main_experiment_additional')
    # # bounding taskの結果を確認
    # show_bounding_task_results('resources/main_experiment_additional', False)

    # 精度を求める
    calculate_accuracy('resources/main_experiment_additional')
    # # タスク完了数の分布を表示
    # show_hist_of_number_of_completed_tasks('resources/main_experiment_additional', is_all=False, is_ef=False)
    # # 分散分析を行う
    # analysis_of_variance('resources/main_experiment_additional')
    # スイッチコストを求める
    # evaluate_switch_cost('resources/main_experiment_additional')
    # extract_bad_workers('resources/main_experiment_additional')
    # # ワーカの停止確率fを求める
    # show_f_of_workers_exit('resources/main_experiment_additional', True)
    # # ワーカの停止確率gを求める
    # show_g_of_workers_exit('resources/main_experiment_additional', True)

    # # 制度に関する分散分析
    # analysis_of_variance_of_accuracy('resources/main_experiment_additional')
