#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import os
import sys
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from utils.np_ioueval import iouEval


DISTANCES = [(1e-8, 300.0),
             (1e-8, 10.0),
             (10.0, 20.0),
             (20.0, 30.0),
             (30.0, 40.0),
             (40.0, 50.0),
             (50.0, 60.0),
             (60.0, 70.0),
             (70.0, 80.0),
             (80.0, 300.0),]

def get_args():
    parser = argparse.ArgumentParser("./evaluate_semantics.py")
    parser.add_argument(
        '--eval_type', '-e',
        default='all',
        type=str, choices=['all', 'sub'],
        help='Eval ALL or just eval Subsample')
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default='/home/chx/Work/SemanticPOSS_dataset/sequences',
        help='Dataset dir. No Default',
    )
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        default='/home/chx/Work/PT-RandLA-Net/poss_result/test-baaf-random',
        help='Prediction dir. Same organization as dataset, but predictions in'
        'each sequences "prediction" directory. No Default. If no option is set'
        ' we look for the labels in the same directory as dataset'
    )
    parser.add_argument(
        '--sequences', '-s',
        nargs="+",
        default=["03"],
        help='evaluated sequences',
    )

    parser.add_argument(
        '--datacfg', '-dc',
        type=str,
        required=False,
        default="utils/semantic-poss.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        required=False,
        default=None,
        help='Limit to the first "--limit" points of each scan. Useful for'
        ' evaluating single scan from aggregated pointcloud.'
        ' Defaults to %(default)s',
    )
    FLAGS = parser.parse_args()

    # fill in real predictions dir
    if FLAGS.predictions is None:
        FLAGS.predictions = FLAGS.dataset
    return FLAGS


def load_label(data_root, sequences, sub_dir_name, ext):
    label_names = []
    for sequence in sequences:
        sequence = '{0:02d}'.format(int(sequence))
        label_paths = os.path.join(data_root, str(sequence), sub_dir_name)
        # populate the label names
        seq_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(label_paths)) for f in fn if f".{ext}" in f]
        seq_label_names.sort()
        label_names.extend(seq_label_names)
    return label_names


if __name__ == '__main__':

    FLAGS = get_args()
    # print summary of what we will do
    print("*" * 80)
    print("INTERFACE:")
    print("Eval What:", FLAGS.eval_type)
    print("Data: ", FLAGS.dataset)
    print("Predictions: ", FLAGS.predictions)
    print("Sequences: ", FLAGS.sequences)
    print("Config: ", FLAGS.datacfg)
    print("Limit: ", FLAGS.limit)
    print("*" * 80)

    print("Opening data config file %s" % FLAGS.datacfg)
    DATA = yaml.safe_load(open(FLAGS.datacfg, 'r'))

    # get number of interest classes, and the label mappings
    class_strings = DATA["labels"]
    class_ignore = DATA["learning_ignore"]
    class_inv_remap = DATA["learning_map_inv"]
    nr_classes = len(class_inv_remap)
    data_config = FLAGS.datacfg
    DATA = yaml.safe_load(open(data_config, 'r'))
    remap_dict = DATA["learning_map"]
    max_key = max(remap_dict.keys())
    remap_lut = np.zeros((max_key + 100), dtype=np.int32)
    remap_lut[list(remap_dict.keys())] = list(remap_dict.values())
    
    # create evaluator
    ignore = []
    for cl, ign in class_ignore.items():
        if ign:
            x_cl = int(cl)
            ignore.append(x_cl)
            print("Ignoring xentropy class ", x_cl, " in IoU evaluation")
    # create evaluator

    # create evaluator
    evaluators = []
    for i in range(len(DISTANCES)):
        evaluators.append(iouEval(nr_classes, ignore))
        evaluators[i].reset()


    # get label paths
    if FLAGS.eval_type == "sub":
        label_names = load_label(FLAGS.dataset, FLAGS.sequences, "labels", "npy")
    else:
        label_names = load_label(FLAGS.dataset, FLAGS.sequences, "labels", "label")

    py_names = load_label(FLAGS.dataset, FLAGS.sequences, "velodyne", "bin")
    # py_names = py_names[0:len(py_names):4]

    # get predictions paths
    pred_names = load_label(FLAGS.predictions, FLAGS.sequences, "predictions", "label")

    # label_names = label_names[0:len(label_names):4]

    print(len(label_names), len(pred_names))
    assert(len(label_names) == len(pred_names))


    print("Evaluating sequences")
    N = len(label_names)
    # open each file, get the tensor, and make the iou comparison
    for i in tqdm(range(N)):
        label_file = label_names[i]
        pred_file = pred_names[i]
        points_file = py_names[i]


        # open label
        if FLAGS.eval_type == "sub":
            label = np.load(label_file)
            label = label.reshape((-1))  # reshape to vector
        else:
            label = np.fromfile(label_file, dtype=np.int32)
            label = label.reshape((-1))
            sem_label = label & 0xFFFF  # semantic label in lower half
            inst_label = label >> 16  # instance id in upper half
            assert ((sem_label + (inst_label << 16) == label).all())
            label = remap_lut[sem_label]
            
        if FLAGS.limit is not None:
            label = label[:FLAGS.limit]  # limit to desired length
        # open prediction

        pred = np.fromfile(pred_file, dtype=np.int32)
        pred = pred.reshape((-1))    # reshape to vector
        pred = remap_lut[pred]
        if FLAGS.limit is not None:
            pred = pred[:FLAGS.limit]  # limit to desired length
        # add single scan to evaluation

        xyzr = np.fromfile(points_file, dtype=np.float32)
        xyzr = xyzr.reshape((-1, 4))
        points = xyzr[:, 0:3]
        if FLAGS.limit is not None:
            points = points[:FLAGS.limit]  # limit to desired length
        depth = np.linalg.norm(points, 2, axis=1)
        # print(depth.shape, pred.shape, label.shape)
        # evaluate for all distances
        for idx in range(len(DISTANCES)):
            # select by range
            lrange = DISTANCES[idx][0]
            hrange = DISTANCES[idx][1]
            mask = np.logical_and(depth > lrange, depth <= hrange)

            # mask by distance
            # mask_depth = depth[mask]
            # print("mask range, ", mask_depth.max(), mask_depth.min())
            mask_label = label[mask]
            mask_pred = pred[mask]

            # add single scan to evaluation
            evaluators[idx].addBatch(mask_pred, mask_label)

    # print for all ranges
    print("*" * 80)
    for idx in range(len(DISTANCES)):
        # when I am done, print the evaluation
        m_accuracy = evaluators[idx].getacc()
        m_jaccard, class_jaccard = evaluators[idx].getIoU()

        # print for spreadsheet
        sys.stdout.write('range {lrange}m to {hrange}m,'.format(lrange=DISTANCES[idx][0],
                                                                hrange=DISTANCES[idx][1]))
        for i, jacc in enumerate(class_jaccard):
            if i not in ignore:
                sys.stdout.write('{jacc:.3f}'.format(jacc=jacc.item()))
                sys.stdout.write(",")
        sys.stdout.write('{jacc:.3f}'.format(jacc=m_jaccard.item()))
        sys.stdout.write(",")
        sys.stdout.write('{acc:.3f}'.format(acc=m_accuracy.item()))
        sys.stdout.write('\n')
        sys.stdout.flush()

    # if FLAGS.codalab:
    #     results = {}
    #     for idx in range(len(DISTANCES)):
    #       # make string for distance
    #       d_str = str(DISTANCES[idx][-1])+"m_"
    #
    #       # get values for this distance range
    #       m_accuracy = evaluators[idx].getacc()
    #       m_jaccard, class_jaccard = evaluators[idx].getIoU()
    #
    #       # put in dictionary
    #       results[d_str+"accuracy_mean"] = float(m_accuracy)
    #       results[d_str+"iou_mean"] = float(m_jaccard)
    #       for i, jacc in enumerate(class_jaccard):
    #         if i not in ignore:
    #           results[d_str+"iou_"+class_strings[class_inv_remap[i]]] = float(jacc)
    #       # save to file
    #       with open('segmentation_scores_distance.txt', 'w') as yaml_file:
    #         yaml.dump(results, yaml_file, default_flow_style=False)