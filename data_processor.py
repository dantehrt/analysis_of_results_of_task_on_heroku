import pandas as pd
import ast

# worker_idとworkerのoutputsの辞書を作成
def dictionary_of_worker_id_and_worker_outputs(outputs_df, worker_ids):
    dic_of_worker = {}
    for worker_id in worker_ids:
        outputs_of_each_worker = outputs_df.query('worker_id == "{}"'.format(worker_id)).sort_values('submit_time')
        dic_of_worker[worker_id] = outputs_of_each_worker

    return dic_of_worker


# タスクを完了した（MTurkでsubmitした）人のデータをリストにして返す
def extract_worker_who_complete_task(directory_path):
    mturk_batch_results = pd.read_csv(directory_path + '/batch_results.csv',
                                      dtype={'input_image_url': 'str', 'answer_annotation_data': 'str'})
    worker_ids = mturk_batch_results['WorkerId'].sort_values().unique()

    return worker_ids

# タスクを完了した（MTurkでsubmitした）人のデータを辞書にして返す
def extract_dic_of_worker_who_complete_task(directory_path):
    mturk_batch_results = pd.read_csv(directory_path + '/batch_results.csv',
                                      dtype={'input_image_url': 'str', 'answer_annotation_data': 'str'})
    worker_dic = {}
    for index, mturk_batch_result in mturk_batch_results.iterrows():
        worker_dic[mturk_batch_result['WorkerId']] = mturk_batch_result['Input.task_condition']

    return worker_dic

# answer_annotation_dataを受け取って，bounding_boxの辞書のリストを左上からの距離でソートして返す
def make_sorted_bounding_box_dic_list(answer_annotation_data):
    # annotation_dataを辞書の文字列のリストに変換
    bounding_box_str_list = list(
        map(lambda x: x + '}', answer_annotation_data[1:-2].split('},')))
    # 辞書のリストに変換
    bounding_box_dic_list = []
    for bounding_box_str in bounding_box_str_list:
        bounding_box_dic = ast.literal_eval(bounding_box_str)
        bounding_box_dic_list.append(bounding_box_dic)
    # bounding_boxを左上からの距離でソート
    sorted_bounding_box_dic_list = sorted(bounding_box_dic_list,
                                          key=lambda x: x['top'] ** 2 + x['left'] ** 2)

    return sorted_bounding_box_dic_list