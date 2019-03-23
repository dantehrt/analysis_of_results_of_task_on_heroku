# -*- coding: utf-8 -*-

import pandas as pd
import ast
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

import pprint
import data_processor

def view_image_from_bounding_box_dic_list(task_number, bounding_box_dic_list):
    # Load the image from the HIT
    img = Image.open('./resources/images/bounding_images/' + str(task_number).zfill(3) + '.jpg')
    im = np.array(img, dtype=np.uint8)
    # Create figure, axes, and display the image
    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for bounding_box_dic in bounding_box_dic_list:
        rect = patches.Rectangle((bounding_box_dic['left'], bounding_box_dic['top']),
                                 bounding_box_dic['width'], bounding_box_dic['height'], linewidth=1,
                                 edgecolor='#32cd32', facecolor='none')
        ax.add_patch(rect)

    plt.show()


def draw_images_for_tagging_task(worker_id, task_number, bounding_box_dic_list):
    im = Image.open('./resources/images/bounding_images/' + str(task_number).zfill(3) + '.jpg')
    draw = ImageDraw.Draw(im)
    font = ImageFont.truetype('/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc', 30)
    for index, bounding_box_dic in enumerate(bounding_box_dic_list):
        draw.rectangle((bounding_box_dic['left'], bounding_box_dic['top'],
                        bounding_box_dic['left'] + bounding_box_dic['width'],
                        bounding_box_dic['top'] + bounding_box_dic['height']), fill=None, outline=(0, 255, 0))
        draw.text((bounding_box_dic['left']+4, bounding_box_dic['top']+2), str(index + 1), font=font, fill=(0, 222, 0))

    directory_path = 'drawed_images/' + worker_id + '/'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    im.save(directory_path + str(task_number).zfill(3) + '.jpg', quality=95)


# 回答から平均となるバウンディングボックスを作成
def draw_average_bounding_boxes(directory_path):
    worker_ids = data_processor.extract_worker_who_complete_task(directory_path)
    mturk_outputs = pd.read_csv(directory_path + '/mturk.output.csv',
                                dtype={'input_image_url': 'str', 'answer_annotation_data': 'str'})

    dic_of_worker = data_processor.dictionary_of_worker_id_and_worker_outputs(mturk_outputs, worker_ids)
    for task_order in range(1, 101):
        # bounding_boxの個数（個数が同じものが最も多いのが妥当な数だと判断）
        box_count_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # 辞書（key=worker_id, value=左上からの距離でソートされたbounding_boxの辞書のリスト）
        dic_of_sorted_bounding_box_dic_list = {}

        for worker_id, results_df in dic_of_worker.items():
            # 複数回同じタスクをしているワーカーがいるので，submit_timeが古い順に並び替え
            target_df = results_df.query('input_image_url == "{:03}"'.format(task_order)).sort_values('submit_time')
            if len(target_df):
                # 一番上を取得
                target_series = target_df.iloc[0]
                # annotation_dataがnullでなく，空でもない場合
                if not target_series.isnull()['answer_annotation_data'] and not target_series[
                                                                                    'answer_annotation_data'] == '[]':
                    sorted_bounding_box_dic_list = data_processor.make_sorted_bounding_box_dic_list(target_series['answer_annotation_data'])
                    if task_order == 10:
                        view_image_from_bounding_box_dic_list(task_order, sorted_bounding_box_dic_list)
                    dic_of_sorted_bounding_box_dic_list[worker_id] = sorted_bounding_box_dic_list
                    box_count_list[len(sorted_bounding_box_dic_list)] += 1
        if task_order == 9:
            print(box_count_list)
        # 平均を取る前に一旦sumをだす
        sum_bounding_box_dic_list = []
        # 平均を取ったbounding_boxの辞書のリスト
        average_bounding_box_dic_list = []
        # task_orderにおけるboxの個数
        box_count = box_count_list.index(max(box_count_list))
        # 事前に辞書のリストを用意
        for i in range(box_count):
            sum_bounding_box_dic_list.append({'top': 0, 'left': 0, 'width': 0, 'height': 0})
            average_bounding_box_dic_list.append({'top': 0, 'left': 0, 'width': 0, 'height': 0})

        # ボックスの個数が妥当な回答の数
        reasonable_parameter = 0
        # bounding_boxの個数とソートされたbounding_boxの辞書のリストが出来たので回す
        for worker_id, sorted_bounding_box_dic_list in dic_of_sorted_bounding_box_dic_list.items():
            # boxの個数が妥当な場合
            if len(sorted_bounding_box_dic_list) == box_count:
                reasonable_parameter += 1
                for index, sorted_bounding_box_dic in enumerate(sorted_bounding_box_dic_list):
                    sum_bounding_box_dic_list[index]['top'] = sum_bounding_box_dic_list[index]['top'] + \
                                                              sorted_bounding_box_dic['top']
                    sum_bounding_box_dic_list[index]['left'] = sum_bounding_box_dic_list[index]['left'] + \
                                                               sorted_bounding_box_dic['left']
                    sum_bounding_box_dic_list[index]['width'] = sum_bounding_box_dic_list[index]['width'] + \
                                                                sorted_bounding_box_dic['width']
                    sum_bounding_box_dic_list[index]['height'] = sum_bounding_box_dic_list[index]['height'] + \
                                                                 sorted_bounding_box_dic['height']

        for i in range(box_count):
            average_bounding_box_dic_list[i]['top'] = sum_bounding_box_dic_list[i]['top'] / reasonable_parameter
            average_bounding_box_dic_list[i]['left'] = sum_bounding_box_dic_list[i]['left'] / reasonable_parameter
            average_bounding_box_dic_list[i]['width'] = sum_bounding_box_dic_list[i]['width'] / reasonable_parameter
            average_bounding_box_dic_list[i]['height'] = sum_bounding_box_dic_list[i]['height'] / reasonable_parameter

        print(task_order)
        view_image_from_bounding_box_dic_list(task_order, average_bounding_box_dic_list)


def extract_bounding_box_dic_list(worker_id, file_path):
    mturk_outputs = pd.read_csv(file_path,
                                dtype={'input_image_url': 'str', 'answer_annotation_data': 'str'})
    worker_outputs = mturk_outputs.query('worker_id == "{}"'.format(worker_id)).sort_values('submit_time')

    for task_order in range(1, 101):
        target_df = worker_outputs.query('input_task_order == {}'.format(task_order))
        if len(target_df):
            # 一番上を取得
            target_series = target_df.iloc[0]
            # annotation_dataがnullでなく，空でもない場合
            if not target_series.isnull()['answer_annotation_data'] and not target_series[
                                                                                'answer_annotation_data'] == '[]':
                sorted_bounding_box_dic_list = data_processor.make_sorted_bounding_box_dic_list(
                    target_series['answer_annotation_data'])
                draw_images_for_tagging_task(worker_id, task_order, sorted_bounding_box_dic_list)

    worker_outputs.to_csv('drawed_images/' + worker_id + '/evaluated_results.csv', index=False)


if __name__ == '__main__':
    # extract_worker_who_complete_task('resources/main_experiment(bounding_box)')
    # extract_bounding_box_dic_list('answer', 'resources/main_experiment(bounding_box)/bounding_box_answer.csv')
    # extract_bounding_box_dic_list('A3KSAP865D3L7D', 'resources/main_experiment(bounding_box)')
    # extract_bounding_box_dic_list('A30NZAZ04OUGQD', 'resources/main_experiment(bounding_box)')
    # extract_bounding_box_dic_list('A13BWRE4H0ADIQ', 'resources/main_experiment(bounding_box)')
    # extract_bounding_box_dic_list('A3TEVNU2YYO1VH', 'resources/main_experiment(bounding_box)')
    draw_average_bounding_boxes('resources/main_experiment_additional')