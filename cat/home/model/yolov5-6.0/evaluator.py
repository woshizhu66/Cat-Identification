import copy
import os
import time
import pandas as pd

pd.set_option('display.max_columns', 500)
pd.set_option('max_colwidth', 500)


def evaluator_v1(img_dir, class_num, class_name, func, threshold_IOU=0.5, threshold_conf=0.25, ap_num=0):
    """
    @param img_dir: e,g, img_dir = '../model/img_test/' .
            you have to put .jpg(/.JPG /.png /.PNG) files and their labeled .txt files into this path both.
            each img must have its own .jpg(or .JPG) file and txt file in this path.
            .jpg(or .JPG) file and txt file for every img should have the same name with different suffix.
            an example: 001.jpg  001.txt 。the labeled .txt files must look like this:
                 1 130.500000 309.500000 211.000000 217.000000
                 0 488.000000 252.500000 186.000000 225.000000
    @param class_num: e,g, class_num = 2. In this case your label should be 0 or 1 only
    @param class_name:  e,g, class_name = ('person', 'helmet').
            In this case person should be labeled with 0 and helmet should be 1
    @param func: your predict function. The input of your func is the path of a img
             and the ret of your func should be like:
             r = [('helmet', '98.84', (131.29471998948318, 310.70161232581506, 222.4404628460224, 224.30275770334094)),
                 ('person', '99.61', (480.44410118689905, 251.42230070554294, 199.1521101731521, 242.10511537698602))]
    @param threshold_IOU: IOU bigger than this threshold_1 means detected, otherwise miss detected!
             If more than one detection boxes are detected for a ground truth box, detection boxes with highest
             IOU will be used!
    @param threshold_conf: boxes with probability smaller than threshold_2 will be ignored!
    @param ap_num: =0 means Interpolating all points, = 11means 11-point interpolation
    @return: nothing
    """
    print('================================= evaluator start to run =================================')
    print('=================================check your input parameters below:')
    print('u are using img_dir(where u put img and txt) =  : ', img_dir)
    print('u are using ', class_num, ' class(es), class name are :', class_name)
    print('u are choosing threshold_IOU  = ', threshold_IOU)
    print('u are choosing threshold_conf  = ', threshold_conf)
    print('u are choosing ap_num = ', ap_num)

    # define the ret structure
    list_rua = ['all', 'all-weighting']
    col_class = list_rua + list(class_name)
    clo_GT = [0] * len(col_class)
    clo_TP = [0] * len(col_class)
    clo_FP = [0] * len(col_class)
    clo_FN = [0] * len(col_class)
    clo_precision = [0] * len(col_class)
    clo_recall = [0] * len(col_class)
    col_ap = [0] * len(col_class)
    clo_M_IOU = [0] * len(col_class)
    clo_time = [0] * len(col_class)
    clo_img_num = [0] * len(col_class)

    # check environment
    files_img_dir = os.listdir(img_dir)
    if not files_img_dir:
        print('warning! no files in ', img_dir,
              ',evaluator do nothing and quit! (u may consider to put img and .txt into the dir)')
        return -1
    if len(files_img_dir) % 2 != 0:
        print('warning! unbalanced img and txt detected in ', img_dir,
              'evaluator do nothing and quit!(should be 1 img - 1 txt)')
        return -1

    dict_GT = {}  # 存放所有GT key=文件名， value= [(c x y w h),(c x y w h)]
    dict_predict = {}  # 存放所有predict key=文件名， value= [(c c (x y w h)),(c c (x y w h))]
    dict_predict_TP = {}  # 存放所有predict ,同时记录这个框是不是TP key=文件名， value= [(c c (x y w h) True),(c c (x y w h) False)]
    time_cost = 0  # 对所有的图片进行一次预测，一共花了多久

    files_img_dir.sort()
    count_pics = len(files_img_dir) / 2
    # 解析 txt ,预测 img
    print('start detecting...')
    current_p_detect = 1
    for name in files_img_dir:
        img_path = os.path.join(img_dir, name)
        if name.endswith('txt'):
            with open(img_path, 'r') as f1:
                lines = f1.readlines()  # read each line
            dict_GT[name.replace('.txt', '')] = lines
        elif name.endswith('jpg') or name.endswith('JPG') or name.endswith('PNG') or name.endswith('png'):
            print("detecting img :", current_p_detect, "/", int(count_pics))
            current_p_detect += 1
            t1 = time.time()
            # image = cv2.imread(img_path, -1)
            t1 = time.time()
            r = func(img_path)
            time_cost += (time.time() - t1)
            name_s = name.replace('.jpg', '').replace('.JPG', '').replace('.png', '').replace('.PNG', '')
            if not len(r) > 0:
                dict_predict[name_s] = []
            else:
                dict_predict[name_s] = r
        else:
            print('warning! non txt/img file detected in ', img_dir,
                  'evaluator do nothing and quit!(supported img format includes:'
                  'jpg JPG png PNG)')
            return -1

    # # 计算 TP FP FN IOU 等
    key_set = dict_GT.keys()  # img_dir 里有多少张图 ,all img names
    # loop classes and img to get results
    # # for each class
    count_i = 0
    test = []
    for class_i in range(class_num):
        print('===============>going to check : ', class_name[class_i])
        GT_class_i = 0  # 分 class, 不分img, all GT
        TP_class_i = 0  # 分 class, 不分img, all TP
        FP_class_i = 0  # 分 class, 不分img, all FP
        FN_class_i = 0  # 分 class, 不分img, all FN
        TP_sum_iou_class_i = 0  # 分 class, 不分img, TP IOU之和
        s_sum_redundant_class_i = 0  # 分 class, 不分img,冗余框面积
        num_sum_allGT_all_img = 0  # 分class,不分 img  name 所有GT框数量
        gt_hit_iou = []  # 所有TP的 IOU ,分 img ,class ,conf
        # # for each pic
        for key in key_set:
            k = 0
            print("merits compute process .. img name is :", key)
            dict_predict_TP[key] = []
            gt_i_img_class = []  # 拿到每张图的所有GT框，分class,放在 gt_i_img_class
            dit_txt_i_clean = []
            for i in dict_GT[key]:
                if i.isspace():
                    continue
                ic = i.replace('\n', '').split(" ")
                dit_txt_i_clean.append(ic)
            for i in dit_txt_i_clean:
                if int(i[0]) == class_i:
                    gt_i_img_class.append(list(i[1:]))

            predict_i_img_class = []  # 拿到每张图的所有predict框，分class,分conf,放在 gt_i_img_class
            for i in dict_predict[key]:
                if i[0] == class_name[class_i] and float(i[1]) >= threshold_conf * 100:
                    predict_i_img_class.append(i)

            gt_i_img_class_flot = []  # 每张图的所有GT框, str 格式 转 float
            # # str to float
            for gtb in gt_i_img_class:
                gtb = list(map(float, gtb))
                gt_i_img_class_flot.append(gtb)

            num_gt_i_img_class = len(gt_i_img_class_flot)
            num_sum_allGT_all_img += num_gt_i_img_class
            redundant_set = []  # 所有冗余 ,分 img ,class ,conf
            gt_hit = []  # 所有TP,正确命中的框 ,分 img ,class ,conf

            # # for each predict box
            for one_predict in predict_i_img_class:
                dict_predict_TP_i = copy.deepcopy(one_predict)
                dict_predict_TP_i = list(dict_predict_TP_i)
                bb = one_predict[2]
                intersection_boxes = find_intersection(bb, gt_i_img_class_flot)  # 计算交叉框
                if not len(intersection_boxes) > 0:  # 无交叉框 ,则冗余
                    print('img ', key, ' class:', class_name[class_i],
                          ' has a predict box redundant (冗余)  due to no intersection! ')
                    redundant_set.append(bb)
                    dict_predict_TP_i.append(False)
                    dict_predict_TP[key].append(dict_predict_TP_i)  # index 3 ,加了是否是GT
                    continue
                intersection_boxes_iou = []
                for itc in intersection_boxes:  # 有交叉框 ,选最大的IOU
                    intersection_boxes_iou.append(calc_iou2(bb, itc))
                if max(intersection_boxes_iou) < threshold_IOU:
                    print('img ', key, ' class:', class_name[class_i],
                          ' has a predict box redundant (冗余) due to small IOU! ')
                    redundant_set.append(bb)  # 有交叉框 ,最大的IOU 小于阈值， 冗余
                    dict_predict_TP_i.append(False)
                    dict_predict_TP[key].append(dict_predict_TP_i)  # index 3 ,加了是否是GT
                    continue

                pos = intersection_boxes_iou.index(max(intersection_boxes_iou))
                gt_hit.append(intersection_boxes[pos])  # 记录这个TP
                dict_predict_TP_i.append(True)
                dict_predict_TP[key].append(dict_predict_TP_i)  # index 3 ,加了是否是GT
                gt_hit_iou.append(max(intersection_boxes_iou))  # 记录这个有TP的 IOU
                gt_i_img_class_flot.remove(intersection_boxes[pos])  # 去掉这个已经有TP的 GT

            # loop 完一张图的 all predictions 后，统计这张图的信息，存在全部图（分class）的变量中
            GT_class_i += num_gt_i_img_class
            TP_class_i += len(gt_hit)
            FP_class_i += len(redundant_set)
            FN_class_i += (num_gt_i_img_class - len(gt_hit))

            TP_sum_iou_class_i = sum(gt_hit_iou)
            print("img:", key, "class:", class_i,
                  "     GT(金标签) =", num_gt_i_img_class,
                  " and TP(正确) =", len(gt_hit),
                  " and FP(冗余) =", len(redundant_set),
                  " and FN(漏检) =", (num_gt_i_img_class - len(gt_hit)))
            s_sum_redundant_class_i = calc_area(redundant_set)  # 冗余框面积,暂未使用

        # # 对于某个类，所有的 img loop 完成之后
        col_class[count_i + 2] = class_name[class_i]
        clo_GT[count_i + 2] = GT_class_i
        clo_TP[count_i + 2] = TP_class_i
        clo_FP[count_i + 2] = FP_class_i
        clo_FN[count_i + 2] = FN_class_i
        if TP_class_i != 0:
            clo_M_IOU[count_i + 2] = TP_sum_iou_class_i / TP_class_i  # 每个TP的 mean IOU
        if (TP_class_i + FP_class_i) != 0:
            clo_precision[count_i + 2] = TP_class_i / (TP_class_i + FP_class_i)
        if (TP_class_i + FN_class_i) != 0:
            clo_recall[count_i + 2] = TP_class_i / (TP_class_i + FN_class_i)
        # ====================================================
        # # compute AP
        # # 构造prediction df ,按照 conf 降序,对每一个 prediction 计算当前 pre rec ，形成 两个list,两个list通过指定的N点分割计算得到AP
        # 构造prediction df
        conf_all_predict = [0] * (TP_class_i + FP_class_i)  # conf
        name_all_predict = [0] * (TP_class_i + FP_class_i)  # 文件名 ,包括某个指定class的全部 img name
        box_all_predict = [0] * (TP_class_i + FP_class_i)  # box 信息
        is_TP_all_predict = [0] * (TP_class_i + FP_class_i)  # 这个box是否是TP
        prediction_df = pd.DataFrame({'conf': conf_all_predict,
                                      'name': name_all_predict,
                                      'box': box_all_predict,
                                      'is_tp': is_TP_all_predict})  # 得到所有prediction （分class）,且维护每一个框属于哪张图,conf是多少
        current_i_predict = 0
        for key in key_set:
            for i in dict_predict_TP[key]:
                if i[0] == class_name[class_i] and float(i[1]) >= threshold_conf * 100:
                    prediction_df.iloc[current_i_predict, 0] = float(i[1])
                    prediction_df.iloc[current_i_predict, 1] = str(key)
                    prediction_df.iloc[current_i_predict, 2] = str(i[2])
                    prediction_df.iloc[current_i_predict, 3] = i[3]
                    current_i_predict += 1
        # 按照 conf 降序排列
        prediction_df = prediction_df.sort_values(by="conf", ascending=False)
        # 对每一个 prediction 计算当前 pre rec
        precision_list = [0]
        recall_list = [0]
        sum_TP = 0
        num_current_prediction = 0
        for i in range(TP_class_i + FP_class_i):
            df_img_name = prediction_df.iloc[i, 1]
            df_box = prediction_df.iloc[i, 2]
            df_conf = prediction_df.iloc[i, 0]
            df_is_TP = prediction_df.iloc[i, 3]  # 这个prediction是TP吗？
            num_current_prediction += 1
            if df_is_TP is True:
                sum_TP += 1
            precision_list.append(round(sum_TP / num_current_prediction, 8) if num_current_prediction != 0 else 0)
            recall_list.append(round(sum_TP / num_sum_allGT_all_img, 8) if num_sum_allGT_all_img != 0 else 0)
        precision_list.append(0)
        recall_list.append(1)
        print("==============recall_list:", recall_list)
        print("==============precision_list:", precision_list)
        col_ap[count_i + 2] = _compute_ap2(recall_list, precision_list, ap_num)
        count_i += 1

    # compute avg and make the ret dataframe
    ret_df = pd.DataFrame({'class': col_class,
                           'GT': clo_GT,
                           'TP': clo_TP,
                           'FP': clo_FP,
                           'FN': clo_FN,
                           'M_IOU': clo_M_IOU,
                           'precision': clo_precision,
                           'recall': clo_recall,
                           'ap': col_ap,
                           'time': clo_time,
                           'img_num': clo_img_num
                           })
    ret_df.iloc[0, 1] = ret_df.iloc[range(2, ret_df.shape[0]), 1].sum()
    ret_df.iloc[0, 2] = ret_df.iloc[range(2, ret_df.shape[0]), 2].sum()
    ret_df.iloc[0, 3] = ret_df.iloc[range(2, ret_df.shape[0]), 3].sum()
    ret_df.iloc[0, 4] = ret_df.iloc[range(2, ret_df.shape[0]), 4].sum()
    ret_df.iloc[0, 5] = ret_df.iloc[range(2, ret_df.shape[0]), 5].mean()
    ret_df.iloc[0, 6] = ret_df.iloc[range(2, ret_df.shape[0]), 6].mean()
    ret_df.iloc[0, 7] = ret_df.iloc[range(2, ret_df.shape[0]), 7].mean()
    ret_df.iloc[0, 8] = ret_df.iloc[range(2, ret_df.shape[0]), 8].mean()
    ret_df.iloc[0, 9] = time_cost
    ret_df.iloc[0, 10] = int(count_pics)
    sum_GT = ret_df.iloc[range(2, ret_df.shape[0]), 1].sum()
    ret_df.iloc[1, 1] = ret_df.iloc[range(2, ret_df.shape[0]), 1].sum()
    ret_df.iloc[1, 2] = ret_df.iloc[range(2, ret_df.shape[0]), 2].sum()
    ret_df.iloc[1, 3] = ret_df.iloc[range(2, ret_df.shape[0]), 3].sum()
    ret_df.iloc[1, 4] = ret_df.iloc[range(2, ret_df.shape[0]), 4].sum()
    w_col = ret_df.iloc[range(2, ret_df.shape[0]), 5] * (ret_df.iloc[range(2, ret_df.shape[0]), 1] / sum_GT)
    ret_df.iloc[1, 5] = w_col.sum()
    w_col = ret_df.iloc[range(2, ret_df.shape[0]), 6] * (ret_df.iloc[range(2, ret_df.shape[0]), 1] / sum_GT)
    ret_df.iloc[1, 6] = w_col.sum()
    w_col = ret_df.iloc[range(2, ret_df.shape[0]), 7] * (ret_df.iloc[range(2, ret_df.shape[0]), 1] / sum_GT)
    ret_df.iloc[1, 7] = w_col.sum()
    w_col = ret_df.iloc[range(2, ret_df.shape[0]), 8] * (ret_df.iloc[range(2, ret_df.shape[0]), 1] / sum_GT)
    ret_df.iloc[1, 8] = w_col.sum()
    ret_df.iloc[1, 9] = time_cost
    ret_df.iloc[1, 10] = int(count_pics)
    print('=============================================================================')
    print('========================evaluator final result below ========================')
    print('class-all : 所有类的汇总，IOU/precision/recall/AP取多类的平均')
    print('class-all_weight : 所有类的汇总，按GT数量进行加权平均')
    print('GT : 手动标注出的')
    print('TP : 手动标注且被模型正确预测中的')
    print('FP : 未被手动标注但被模型预测成的，（冗余）')
    print('FN : 手动标注但未被模型预测成的，（漏检）')
    print('M_IOU : TP的IOU均值')
    print('precision : 精度')
    print('recall : 召回')
    print('AP : ap,默认是全概率分割，可选11等分割或任意其他粒度')
    print(ret_df)
    print('=============================================================================')
    print('========================evaluator finished!==================================')
    return ret_df


def find_intersection(ground_truth, bound_box_set):
    """
    ground_truth_set = [(x,y,w,h,),(x,y,w,h,)]
    bound_box_set = [(x,y,w,h,),(x,y,w,h,)]
    如果两个矩形相交，那么矩形A B的中心点和矩形的边长是有一定关系的:两个中心点间的距离肯定小于(等)AB边长和的一半,(去掉等于的场景)。
    """
    ret = []
    if len(ground_truth) < 4:
        return ret
    for bb in bound_box_set:
        diff_x = abs(ground_truth[0] - bb[0])
        diff_y = abs(ground_truth[1] - bb[1])
        add_x = ground_truth[2] + bb[2]
        add_y = ground_truth[3] + bb[3]
        if (diff_x < add_x / 2) & (diff_y < add_y / 2):
            ret.append(bb)
    return ret


def make_box(gtb):
    x = gtb[0]
    y = gtb[1]
    w = gtb[2]
    h = gtb[3]
    box = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
    return box


def calc_iou2(gtb, bb):
    boxA = make_box(gtb)
    boxB = make_box(bb)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # print("xA={},yA={},xB={},yB={}".format(xA,yA,xB,yB))
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-16)
    # return the intersection over union value
    # print('IOU = ', iou)
    return iou


def calc_area(nms):
    ret = 0
    for noms in nms:
        ret += noms[2] * noms[3]
    return ret


def _compute_ap2(recall, precision, ap_number=0):
    """
    :param recall:     list ,有多少个prediction boxes 就有多长 ,升序排列
    :param precision:  list ,有多少个prediction boxes 就有多长 ,和 recall 对应
    :param ap_number:  等分方式，默认全等分  From 2010 on, the method of computing AP by the PASCAL VOC challenge has changed. Currently,
    the interpolation performed by PASCAL VOC challenge uses all data points, rather than interpolating only 11 equally spaced points
    as stated in their paper.
    :return: AP
    """
    ap = 0
    if ap_number == 11:
        # The 11-point interpolation tries to summarize the shape of the Precision x Recall curve by averaging
        # the precision at a set of eleven equally spaced recall levels [0, 0.1, 0.2, ... , 1]:
        for j in [i / 10 for i in range(11) if True]:  # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            index = 0
            for r_i in recall:
                if r_i >= j:
                    break
                index += 1
            if index == len(recall):
                pre_max_right_side = 0
            else:
                pre_max_right_side = max(precision[index:])
            curr = 1 / ap_number * pre_max_right_side
            ap += curr
    elif ap_number == 0:  # 全分割
        r_set = recall
        for i in range(len(r_set)):
            pre_max_right_side = max(precision[i:])
            if i == 0:
                ap += (r_set[i] - 0) * pre_max_right_side
            else:
                ap += (r_set[i] - r_set[i - 1]) * pre_max_right_side
    return ap