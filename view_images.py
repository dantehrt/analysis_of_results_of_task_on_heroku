import ast
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# 回答と評価対象をを一行ずつ受け取り画像を描画
def from_answers_and_results(answer, result):
    # Load the image from the HIT
    img = Image.open('./resources/images/bounding_images/' + result['input_image_url'] + '.jpg')
    im = np.array(img, dtype=np.uint8)
    # Create figure, axes, and display the image
    fig, ax = plt.subplots(1)
    ax.imshow(im)

    # 何も入力されていない時
    if result['answer_annotation_data'] == '{}':
        pass
    else:
        result_bounding_box_list = list(map(lambda x: x + '}', result['answer_annotation_data'][1:-2].split('},')))
        for result_bounding_box_str in result_bounding_box_list:
            result_bounding_box = ast.literal_eval(result_bounding_box_str)
            # print(str(result_bounding_box))

            rect = patches.Rectangle((result_bounding_box['left'], result_bounding_box['top']),
                                     result_bounding_box['width'], result_bounding_box['height'], linewidth=1,
                                     edgecolor='#32cd32', facecolor='none')
            ax.add_patch(rect)

    answer_bounding_box_list = list(map(lambda x: x + '}', answer['answer_annotation_data'][1:-2].split('},')))
    for answer_bounding_box_str in answer_bounding_box_list:
        answer_bounding_box = ast.literal_eval(answer_bounding_box_str)

        rect = patches.Rectangle((answer_bounding_box['left'], answer_bounding_box['top']),
                                 answer_bounding_box['width'], answer_bounding_box['height'], linewidth=1,
                                 edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    plt.show()


# 回答と評価対象のバウンディングボックスの辞書を一つずつ受け取って描画
def from_answer_dic_and_target_dic(bounding_box_dic_of_answer, bounding_box_dic_of_target, image_url):
    # Load the image from the HIT
    img = Image.open('./resources/images/bounding_images/' + image_url + '.jpg')
    im = np.array(img, dtype=np.uint8)
    # Create figure, axes, and display the image
    fig, ax = plt.subplots(1)
    ax.imshow(im)

    rect = patches.Rectangle((bounding_box_dic_of_target['left'], bounding_box_dic_of_target['top']),
                                 bounding_box_dic_of_target['width'], bounding_box_dic_of_target['height'], linewidth=1,
                                 edgecolor='#32cd32', facecolor='none')
    ax.add_patch(rect)

    rect = patches.Rectangle((bounding_box_dic_of_answer['left'], bounding_box_dic_of_answer['top']),
                             bounding_box_dic_of_answer['width'], bounding_box_dic_of_answer['height'], linewidth=1,
                             edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    plt.show()
    
# 二つのバウンディングボックスの辞書のリストを受け取って描画する．
def from_answer_and_target_box_dic_list(answer_bounding_box_dic_list, target_bounding_box_dic_list, image_url):
    # Load the image from the HIT
    img = Image.open('./resources/images/bounding_images/' + image_url + '.jpg')
    im = np.array(img, dtype=np.uint8)
    # Create figure, axes, and display the image
    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for target_bounding_box_dic in target_bounding_box_dic_list:
        rect = patches.Rectangle((target_bounding_box_dic['left'], target_bounding_box_dic['top']),
                                     target_bounding_box_dic['width'], target_bounding_box_dic['height'], linewidth=1,
                                     edgecolor='#32cd32', facecolor='none')
        ax.add_patch(rect)

    for answer_bounding_box_dic in answer_bounding_box_dic_list:
        rect = patches.Rectangle((answer_bounding_box_dic['left'], answer_bounding_box_dic['top']),
                                 answer_bounding_box_dic['width'], answer_bounding_box_dic['height'], linewidth=1,
                                 edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    plt.show()


# バウンディングボックスの辞書のリストを受け取って描画する関数
def from_bounding_box_dic_list(image_url, bounding_box_dic_list):
    # Load the image from the HIT
    img = Image.open('./resources/images/bounding_images/' + image_url + '.jpg')
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
