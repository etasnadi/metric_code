import argparse
import math
import os
import zipfile
import shutil
import traceback
from pathlib import Path
from collections import defaultdict

import imageio
import numpy as np
import pandas as pd

from stardist.matching import matching_dataset, matching


imread_func = imageio.v2.imread

class DsbLikeEvaluator:
    def __init__(self, submissions_path, gold_path, output_path, by_image, step):
        super().__init__()
        self.submissions_path = submissions_path
        self.gold_path = gold_path
        self.output_path = output_path
        
        # image_id -> mask where mask is a numpy array of shape (h, w)
        self.gold_masks = {}

        # Set to true when image level stats are needed.
        self.provide_detailed_results = False
        self.eval_thresholds = np.arange(.5, 1., step)
        self.evaluate_by_image = by_image

        print('Evaluating...')
        print('\tby_image: %s' % self.evaluate_by_image)
        print('\teval_thresholds: %.2f' % step)

    def load_gold_masks(self):
        '''
        Loads the chain codes from the gold_path csv file into the gold_masks member.
        '''
        csv = pd.read_csv(self.gold_path)
        chain_codes, image_sizes = self.parse_gold_csv(csv)
        self.gold_masks = self.parse_chain_codes(chain_codes, image_sizes)

    def load_gold_masks_cached(self, gold_path):
        '''
        Loads the image_id -> mask map from the masks folder.
        '''
        masks = {}
        for mask_path in gold_path.iterdir():
            mask = imread_func(mask_path)
            masks[mask_path.stem] = mask
        return masks

    def parse_chain_codes(self, chain_codes, image_sizes):
        '''
        Converts the chain codes to masks:
        @arg chain_codes: image_id -> [chain_code*]
        @arg image_sizes: image_id -> (h, w)
        '''
        gold_masks = {}
        for imid in chain_codes.keys():
            print('Processing: %s...' % imid)
            image_chain_codes = chain_codes[imid]
            size = image_sizes[imid][0]
            assert len(set(image_sizes[imid])) == 1, ValueError("All sizes should be equal for an image entry.")
            mask = self.chain_codes_to_mask(image_chain_codes, size)
            gold_masks[imid] = mask
        return gold_masks

    def chain_codes_to_mask(self, chain_codes, mask_shape):
        '''
        Draws the list of chain_codes to an empty canvas of shape mask_shape.

        @arg chain_codes: [chain_code*]
        @arg mask_shape: (h, w)
        '''
        label = np.zeros(mask_shape, np.uint16)
        for index, chain_code in enumerate(chain_codes):
            try:
                decoded = self.chain_code_to_mask(chain_code, mask_shape)
                if decoded is not None:
                    label[decoded == 255] = index + 1
            except Exception as e:
                print("Can't parse chain code, skipping object... Original chain code: %s" % chain_code)
                print(index, chain_code)
                print(traceback.format_exc())

        return label

    def decode_submission_masks(self, submission_chain_codes):
        submission_masks = {}
        for imid in self.gold_masks.keys():
            if imid in submission_chain_codes:
                image_chain_codes = submission_chain_codes[imid]
                mask_shape = self.gold_masks[imid].shape
                label = self.chain_codes_to_mask(image_chain_codes, mask_shape)
                if label is not None:
                    submission_masks[imid] = label
                else:
                    submission_masks[imid] = np.zeros_like(self.gold_masks[imid])
            else:
                submission_masks[imid] = np.zeros_like(self.gold_masks[imid])
        return submission_masks


    def load_sumbission_masks(self, submmission_zip_path):
        try:
            submission_content = self.load_submission_csv(submmission_zip_path)
            submission_chain_codes = self.parse_submission_csv(submission_content, self.gold_masks.keys())
        except Exception as e:
            print('Parsing of the submission failed: %s!' % submmission_zip_path, flush=True)
            print(str(e))
            traceback.print_exc()
            return
        
        return self.decode_submission_masks(submission_chain_codes)

    def evaluate(self):
        self.load_gold_masks()
        for sub_path in self.submissions_path.iterdir():
            self.evaluate_one_sub(sub_path)

    def evaluate_one_sub(self, sub_path):
        sub_masks = self.load_sumbission_masks(sub_path)
        
        image_ids = self.gold_masks.keys()
        pred_masks_list = [sub_masks[imid] for imid in image_ids]
        gold_masks_list = [self.gold_masks[imid] for imid in image_ids]

        if self.provide_detailed_results:
            for idx, (y_true, y_pred, image_id) in enumerate(zip(gold_masks_list, pred_masks_list, image_ids)):
                image_result = pd.concat([pd.DataFrame.from_dict(matching(y_true, y_pred, thresh=t) for t in self.eval_thresholds)])
                image_result['ImageID'] = image_id
                if idx == 0:
                    dataset_result = image_result
                else:
                    dataset_result = pd.concat([dataset_result, image_result])
        else:
            dataset_result = pd.concat([pd.DataFrame.from_dict(
                matching_dataset(gold_masks_list, pred_masks_list, thresh=t, show_progress=False, by_image=self.evaluate_by_image) for t in self.eval_thresholds)])
        
        print("Done %s." % sub_path.name)
        result_csv_file = self.output_path / ('%s.csv' % sub_path.stem)
        dataset_result.to_csv(result_csv_file, index=False)

    def save_masks(self, masks, output_path):
        '''
        @arg masks: mask_name -> mask.
        '''

        for mask_name, mask in masks.items():
            imageio.imwrite(output_path / ('%s.tif' % mask_name), mask)


class Dsb2018Evaluator(DsbLikeEvaluator):
    '''
    Some evaluation code is extracted mainly from Caicedo's notebooks for maximum compatiblity.
    '''

    def __init__(self, submissions_path, gold_path, output_path, by_image, step):
        super().__init__(submissions_path, gold_path, output_path, by_image, step)
        self.tmpdir = '/tmp'

    def parse_gold_csv(self, csv_data):
        image_chain_codes = defaultdict(list)
        image_sizes = defaultdict(list)

        for index, mask_object in enumerate(csv_data.itertuples()):
            if mask_object.Usage == 'Private':
                assert mask_object.EncodedPixels != '1 1', \
                    ValueError("Chain code does not exist for '%s'!" % mask_object.ImageId)
                image_chain_codes[mask_object.ImageId].append(mask_object.EncodedPixels)
                image_size = (int(mask_object.Height), int(mask_object.Width))
                image_sizes[mask_object.ImageId].append(image_size)
            elif mask_object.Usage == 'Ignored':
                assert mask_object.EncodedPixels == '1 1', \
                    ValueError("Ignored image but chain code exists: '%s' %d." % mask_object.EncodedPixels)
                pass
            else:
                raise ValueError("Unknown image type for image!" % mask_object.ImageId)

        return image_chain_codes, image_sizes
    
    def load_submission_csv(self, submission_path):
        return self.archive_to_csv(submission_path)

    def archive_to_csv(self, pathname):
        '''
        Caicedo's code to read the submission from the submitted zip file.
        Extracts a zip into a tmp dir, then reads the csv file found in the submission and returns.
        @arg pathname: the path of the submission .zip file.
        '''

        # Should be refactored later.

        _, directory = os.path.split(pathname)

        directory, _ = os.path.splitext(directory)

        directory = os.path.join(str(self.tmpdir), "DSB2018", directory)
        try:
            with zipfile.ZipFile(pathname, "r") as stream:
                names = stream.namelist()

                # Test whether the archive has _exactly_ one name:
                assert len(names) == 1

                name = names.pop()

                pathname = os.path.join(directory, name)

                #         try:
                stream.extractall(directory)

                # Test whether the archive was _successfully_ extracted:
                assert os.path.exists(pathname)

                scores = pd.read_csv(pathname)

                shutil.rmtree(directory, ignore_errors=True)
        except Exception as e:
            shutil.rmtree(directory, ignore_errors=True)
            raise e
        #         finally:
        #             directory = os.path.join("/tmp", "DSB2018")

        #             shutil.rmtree(directory)

        return scores

    def parse_submission_csv(self, csv, interested_list=None):
        '''
        Reads the image-id to chain codes mapping from the submission CSV.

        @arg interested_list: if set return the record only if it is in the interested list w.r its image-id.
        @return: [chain_code*]
        '''

        image_chain_codes = defaultdict(list)
        for index, mask_object in enumerate(csv.itertuples()):
            imid = mask_object.ImageId
            if interested_list is not None:
                if imid in interested_list:
                    image_chain_codes[imid].append(mask_object.EncodedPixels)
            else:
                image_chain_codes[imid].append(mask_object.EncodedPixels)

        return image_chain_codes

    def chain_code_to_mask(self, chain_code, mask_shape, color=255):
        '''
        Caicedo's code to decode a chain code.
        Draws a single chain code to an emtpy mask (therefore a binary mask is created).

        @arg chain_code: [int*]
        @mask_shape: (y, x)
        '''
        r, c = mask_shape

        if str(chain_code) == 'nan':
            return None

        chain_code = chain_code.replace("[", "").replace("]", "").replace(",", " ")
        chain_code = [int(instance) for instance in chain_code.split(" ") if instance != '']

        image = np.zeros(r * c, dtype=np.uint8)

        for index, size in np.array(chain_code).reshape(-1, 2):
            index -= 1

            image[index:index + size] = color

        return image.reshape(c, r).transpose()


class SartoriusEvaluator(DsbLikeEvaluator):
    def __init__(self, submissions_path, gold_path, output_path, by_image, step):
        super().__init__(submissions_path, gold_path, output_path, by_image, step)

    def parse_gold_csv(self, mask_objects):
        '''
        Returns two maps from the gold CSV file:
            image_id -> [chain-code*]
            image_id -> [size*]
        It also does some checks on the CSV to understand the format.
        '''
        image_chain_codes = defaultdict(list)
        image_sizes = defaultdict(list)

        for mask_object in mask_objects.itertuples():
            chain_code = mask_object.expected
            
            # The gold database could contain empty rows for some reason like: 0cfea2cdde8e.
            if isinstance(chain_code, float) and math.isnan(chain_code):
                continue

            image_chain_codes[mask_object.id].append(chain_code)
            image_size = (int(mask_object.height), int(mask_object.width))
            image_sizes[mask_object.id].append(image_size)

        return image_chain_codes, image_sizes

    def load_submission_csv(self, submission_path):
        return pd.read_csv(submission_path, dtype={'id': str, 'predicted': str}, keep_default_na=False)

    def parse_submission_csv(self, sub_csv, interested_list=None):
        submission_meta = defaultdict(list)

        for entry in sub_csv.itertuples():
            if not hasattr(entry, 'predicted'):
                print('No attribute: predicted.')
                continue
            
            if not hasattr(entry, 'id'):
                print('No attribute: id.')
                continue

            if len(entry.predicted) < 3:
                print('Empty row: "%s"' % entry.predicted)
                continue

            if interested_list is not None:
                if entry.id in interested_list:
                    submission_meta[entry.id].append(entry.predicted)
            else:
                submission_meta[entry.id].append(entry.predicted)

        return submission_meta

    def chain_code_to_mask(self, chain_code, mask_shape, color=255):
        '''
        https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/discussion/278663

        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return 
        Returns numpy array, 1 - mask, 0 - background
        '''
        
        # Split the string by space, then convert it into an integer array
        s = np.array(chain_code.split(), dtype=int)

        # Every even value is the start, every odd value is the "run" length
        starts = s[0::2] - 1
        lengths = s[1::2]
        ends = starts + lengths

        # The image image is actually flattened since RLE is a 1D "run"
        h, w = mask_shape
        img = np.zeros((h * w), dtype=np.float32)

        # The color here is actually just any integer you want!
        for lo, hi in zip(starts, ends):
            img[lo : hi] = color
        # Don't forget to change the image back to the original shape
        return img.reshape(mask_shape)

def main():
    evaluators = {
        'dsb2018': Dsb2018Evaluator,
        'sartorius': SartoriusEvaluator,
    }

    parser = argparse.ArgumentParser(description='Process DSB like submissions.')
    parser.add_argument('--submissions-path', type=str, required=True)
    parser.add_argument('--gold-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--evaluator', type=str, required=True, choices=evaluators.keys())
    parser.add_argument('--by-image', action='store_true', default=False)
    parser.add_argument('--eval_threshold_step', type=float, default=.05)

    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    ev = evaluators[args.evaluator](
        Path(args.submissions_path), 
        Path(args.gold_path),
        output_path,
        args.by_image,
        args.eval_threshold_step)
    
    ev.evaluate()

if __name__ == '__main__':
    main()