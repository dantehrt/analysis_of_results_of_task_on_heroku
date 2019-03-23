# -*- coding: utf-8 -*-

import pandas as pd
import ast
import os

import analyser


# tag_taskの時に比較をする
def compare_when_tag_task(answer, result):
    number_of_correct = 0
    evaluated_result = pd.Series(result)
    for i in range(1, 6):
        column_name = 'answer_text_box' + str(i)
        if answer[column_name] == result[column_name].lower():
            number_of_correct += 1
    if number_of_correct == 5:
        evaluated_result['approve'] = 'x'
        is_correct = True
    else:
        evaluated_result['reject'] = 'Sorry. Your answer is incorrect.'
        is_correct = False

    return evaluated_result, is_correct


# bounding_taskの時に比較をする
def compare_when_bounding_task(answer, result):
    evaluated_result = pd.Series(result)

    # 何も入力されていない時
    if result['answer_annotation_data'] == '{}':
        evaluated_result['reject'] = 'Sorry. Your answer is incorrect.'
        is_correct = False
    else:
        result_bounding_box_list = list(map(lambda x: x + '}', result['answer_annotation_data'][1:-2].split('},')))
        answer_bounding_box_list = list(map(lambda x: x + '}', answer['answer_annotation_data'][1:-2].split('},')))
        # bounding_boxの数が少ない時（何かに付けていない）
        number_of_bounding_boxes = len(answer_bounding_box_list)
        if len(result_bounding_box_list) < number_of_bounding_boxes:
            evaluated_result['reject'] = 'Sorry. Your answer is incorrect.'
            is_correct = False
        else:
            number_of_correct = 0
            for result_bounding_box_str in result_bounding_box_list:
                # 辞書として使えるように変更
                result_bounding_box = ast.literal_eval(result_bounding_box_str)
                # ５つの答え全てと比較する
                for answer_bounding_box_str in answer_bounding_box_list:
                    answer_bounding_box = ast.literal_eval(answer_bounding_box_str)
                    if (result_bounding_box['left'] > answer_bounding_box['left'] - 10
                            and result_bounding_box['left'] < answer_bounding_box['left'] + 10
                    and result_bounding_box['top'] > answer_bounding_box['top'] - 10
                            and result_bounding_box['top'] < answer_bounding_box['top'] + 10
                    and result_bounding_box['width'] > answer_bounding_box['width'] - 15
                            and result_bounding_box['width'] < answer_bounding_box['width'] + 15
                    and result_bounding_box['height'] > answer_bounding_box['height'] - 15
                            and result_bounding_box['height'] < answer_bounding_box['height'] + 15):
                        number_of_correct += 1
            # １つ間違いまで許容
            if number_of_correct >= number_of_bounding_boxes - 1:
                evaluated_result['approve'] = 'x'
                is_correct = True
            else:
                evaluated_result['reject'] = 'Sorry. Your answer is incorrect.'
                is_correct = False

    return evaluated_result, is_correct


# 答えとターゲットのcsv(をpandasに変換したもの)を受け取り，評価をつけたDataFrameを返す
def evaluate_targets(answers, targets):
    approval = 0 # 承認数
    rejection = 0 # reject数
    df = pd.DataFrame()
    for index, targets_row in targets.iterrows():
        image_url = targets_row['input_image_url']
        answer_row = answers[answers['input_image_url'] == image_url].iloc[0]
        if targets_row['input_task_type'] == 't':
            evaluated_targets, is_correct = compare_when_tag_task(answer_row, targets_row)
        elif targets_row['input_task_type'] == 'b':
            evaluated_targets, is_correct = compare_when_bounding_task(answer_row, targets_row)
            # print(is_correct)
            # analyser.view_image_with_bounding_box(answer_row, targets_row)
        else:
            print('何かがおかしいよ')
            is_correct = False
            evaluated_targets = None

        if is_correct:
            approval += 1
        else:
            rejection += 1
        df = df.append(evaluated_targets)

    return df, approval, rejection


# workerごとの結果を抽出し，辞書化して返す
def extract_results_of_each_worker_id(directory_path):
    results = pd.read_csv(directory_path + '/output.csv', dtype={'input_image_url':'str'}).sort_values('worker_id')

    dic = {}
    worker_ids = results['worker_id'].unique()
    for worker_id in worker_ids:
        df_of_worker_id = results[results['worker_id'] == worker_id].sort_values('submit_time')
        dic[worker_id] = df_of_worker_id

    return dic


# 全ての結果に評価をつけて保存
def evaluate_all_results(directory_path):
    answers = pd.read_csv(directory_path + '/answer.csv', dtype={'input_image_url':'str'})
    results = pd.read_csv(directory_path + '/output.csv', dtype={'input_image_url':'str'})

    df, approval, rejection = evaluate_targets(answers, results)

    print('approve：' + str(approval))
    print('reject：' + str(rejection))
    df.to_csv(directory_path + '/evaluated_results_directory/evaluated_results.csv', index=False)


# ワーカごとの結果に評価を付けて保存
def evaluate_results_of_each_worker_id(directory_path):
    answers = pd.read_csv(directory_path + '/answer.csv', dtype={'input_image_url':'str'})
    results_of_each_worker_id = extract_results_of_each_worker_id(directory_path)

    for worker_id, results in results_of_each_worker_id.items():
        df, approval, rejection = evaluate_targets(answers, results)

        if not os.path.exists(directory_path + '/evaluated_results_directory'):
            os.makedirs(directory_path + '/evaluated_results_directory')
        df.to_csv(directory_path + '/evaluated_results_directory/evaluated_results_of_' + worker_id + '.csv', index=False)


if __name__ == '__main__':
    # evaluate_all_results('experiment2')
    # evaluate_results_of_each_worker_id('resources/main_experiment_results')
    # # 不真面目なワーカ
    # analyser.show_detail('resources/evaluated_results_directory/evaluated_results_of_' + '30ZX6P7VF8VKTCO1QI1AWI4JHLZ2JX' + '.csv')
    # # 真面目なワーカ
    # analyser.show_detail('evaluated_results_of_first_experiment_directory/evaluated_results_of_A17T285YLOZKZQ.csv')
    # analyser.show_detail('resources/experiment2/evaluated_results_directory/evaluated_results.csv', 'experiment2')
    analyser.show_detail('resources/main_experiment_results/evaluated_results_directory/evaluated_results_of_okino.csv')
    # analyser.calculate_average_time()