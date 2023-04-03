import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stardist.matching import matching_dataset


class MidogEvaluator:
    def __init__(self, predictions_path, gold_path, submission_matchings_path, pixel_sizes_path, output_path):
        super().__init__()

        self.predictions_path = predictions_path
        self.gold_path = gold_path
        self.submission_matchings_path = submission_matchings_path
        self.pixel_sizes_path = pixel_sizes_path
        self.output_path = output_path

        self.eval_thresholds = np.arange(.5, 1., .05)
        self.evaluate_by_image = True
        self.box_extent_w_px = 25
        self.box_extent_h_px = 25

        self.resolution = {}
        self.gold = {}

        self.debug = False

    def load_pixel_resolutions(self):
        # Pixel resolution
        with open(self.pixel_sizes_path, 'r') as f:
            mpp = f.readlines()
            for l in mpp:
                print(l)
                s = l.split(', ')
                xy = (float(s[1]), float(s[2].strip()))
                self.resolution[Path(s[0]).stem] = (float(xy[0]), float(xy[1]))

    def load_gold(self):
        # Gold
        with open(self.gold_path, 'r') as f:
            self.gold = json.load(f)

    def load_matching(self, matching_path):
        # Matchings
        with open(matching_path, 'r') as f:
            matching = json.load(f)
            image_names = matching.keys()
            for image_name in image_names:
                print('\tImage:', image_name)
                gt_ids = matching[image_name][0]
                pred_ids = matching[image_name][1]
                print('\t\t', gt_ids, '-->', pred_ids)
            
            return matching

    def load_prediction(self, team_name):
        # Predictions
        pred_file_path = self.predictions_path/('%s' % team_name)
        with open(pred_file_path, 'r') as pred_file:
            pred = json.load(pred_file)
        return pred

    def evaluate_one_sub(self, submission_matching_path):
        team_name = '%s.json' % submission_matching_path.stem[:-len('_matching')]
        team_id = submission_matching_path.stem
        
        dataset_scores = []
        dataset_matchings = []

        matching = self.load_matching(submission_matching_path)
        pred = self.load_prediction(team_name)

        # Process each image
        for idx, matching_im in enumerate(matching.keys()):
            

            print('Processing image:', idx, '->', matching_im)
            microns_per_pixel = self.resolution[Path(matching_im).stem]
            mmx, mmy = (1./(microns_per_pixel[0]/1000.), 1./(microns_per_pixel[1]/1000.))

            # Extract the coordinates AND convert to pixel space
            gold_boxes = [[float(elem[0])*mmx, float(elem[1])*mmy] for elem in self.gold[matching_im]]
            pred_boxes = [[float(elem['point'][0])*mmx, float(elem['point'][1])*mmy] for elem in pred[matching_im]]
            matched_ids = [[m[0], m[1]] for m in zip(matching[matching_im][0], matching[matching_im][1])]
            
            image_scores = self.fill_score_matrix(gold_boxes, pred_boxes, matched_ids, self.box_extent_w_px, self.box_extent_h_px)
            image_matchings = [matching[matching_im][0], matching[matching_im][1]]

            dataset_scores.append(image_scores)
            dataset_matchings.append(image_matchings)

            if self.debug:
                print(gold_boxes)
                print()
                print(self.gold[matching_im])
                print()
                print(pred_boxes)
                print()
                print(pred[matching_im])
                print()
                print(matched_ids)
        
        dataset_result = pd.concat([pd.DataFrame.from_dict(
            matching_dataset(dataset_scores, dataset_matchings, thresh=t, show_progress=False, by_image=self.evaluate_by_image, scores_input=True) for t in self.eval_thresholds)])

        dataset_result.to_csv(self.output_path / ('%s.csv' % team_id), index=False)

    def evaluate(self):
        self.load_gold()
        self.load_pixel_resolutions()

        '''
        Data formats:

        gold: {name: [ [x, y, unused]* ]}
        pred: {name: [ {'point': [x, y, unused]}* ]}
        matching: {name: [[id*], [id*]]}
        '''

        # Matcings & Preds
        for submission_matching in self.submission_matchings_path.iterdir():
            with open(submission_matching, 'r') as f:
                self.evaluate_one_sub(submission_matching)
                
    def fill_score_matrix(self, gold, pred, matching, bew, beh):
        n_gold = len(gold)
        n_pred = len(pred)

        score_matrix = np.zeros((n_gold, n_pred))
        
        for idx_gold, idx_pred in matching:
            box_gold = gold[idx_gold]
            box_pred = pred[idx_pred]

            iou = IoU.iou_box(box_gold[0], box_gold[1], box_pred[0], box_pred[1], bew, beh)
            score_matrix[idx_gold, idx_pred] = iou
        
        return score_matrix

class IoU:

    @staticmethod
    def intersect(a_xmin, a_xmax, a_ymin, a_ymax, 
                b_xmin, b_xmax, b_ymin, b_ymax):  # returns None if rectangles don't intersect
        dx = min(a_xmax, b_xmax) - max(a_xmin, b_xmin)
        dy = min(a_ymax, b_ymax) - max(a_ymin, b_ymin)
        if (dx>=0) and (dy>=0):
            return dx*dy
        else:
            return 0.

    @staticmethod
    def union(a_xmin, a_xmax, a_ymin, a_ymax, 
                b_xmin, b_xmax, b_ymin, b_ymax, I):
        
        U = (a_xmax - a_xmin)*(a_ymax - a_ymin) + (b_xmax - b_xmin)*(b_ymax - b_ymin) - I
        return U

    @staticmethod
    def iou_box(ax, ay, bx, by, bew, beh):
        # bw: width / 2 (extent x)
        # bh: height / 2 (extent y)

        a_xmin = ax - bew
        a_xmax = ax + bew
        a_ymin = ay - beh
        a_ymax = ay + beh
        
        b_xmin = bx - bew
        b_xmax = bx + bew
        b_ymin = by - beh
        b_ymax = by + beh

        I = IoU.intersect(
            a_xmin, a_xmax, a_ymin, a_ymax,
            b_xmin, b_xmax, b_ymin, b_ymax)
        
        U = IoU.union(
            a_xmin, a_xmax, a_ymin, a_ymax,
            b_xmin, b_xmax, b_ymin, b_ymax, I)

        return I / U

def test_IoU():
    i1 = IoU.intersect(a_xmin=10, a_xmax=20, a_ymin=10, a_ymax=20, 
            b_xmin=15, b_xmax=25, b_ymin=15, b_ymax=25)
    print('Intersect test:', i1==25)

    i2 = IoU.intersect(a_xmin=10, a_xmax=20, a_ymin=10, a_ymax=20, 
            b_xmin=20, b_xmax=30, b_ymin=20, b_ymax=30)
    print('Intersect test:', i2==0)

    u3 = IoU.union(a_xmin=10, a_xmax=20, a_ymin=10, a_ymax=20, 
            b_xmin=15, b_xmax=25, b_ymin=15, b_ymax=25, I=25)
    print('Union test:', u3==175)

    iou4 = IoU.iou_box(ax=15, ay=15, bx=20, by=20, beh=5, bew=5)
    print('IoU test:', iou4==25/175)

    u5 = IoU.union(a_xmin=10, a_xmax=20, a_ymin=10, a_ymax=20, 
                b_xmin=20, b_xmax=30, b_ymin=20, b_ymax=30, I=0)
    print('Union test:', u5==200)

    u6 = IoU.union(a_xmin=10, a_xmax=20, a_ymin=10, a_ymax=20, 
                b_xmin=10, b_xmax=20, b_ymin=10, b_ymax=20, I=100)
    print('Union test:', u6==100)

def main():
    parser = argparse.ArgumentParser(description='Process midog.')
    parser.add_argument('--gold-path', type=str, required=True)
    parser.add_argument('--submissions-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--matching-path', type=str, required=True)
    parser.add_argument('--pixel-sizes', type=str, required=True)
    args = parser.parse_args()
    
    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    mev = MidogEvaluator(
        predictions_path=Path(args.submissions_path),
        gold_path=Path(args.gold_path),
        submission_matchings_path=Path(args.matching_path),
        pixel_sizes_path=Path(args.pixel_sizes),
        output_path=output_path
    )

    mev.evaluate()

if __name__ == '__main__':
    main()