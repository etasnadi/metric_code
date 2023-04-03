import sys
from pathlib import Path

import numpy as np
import pandas as pd

if len(sys.argv) < 3:
    print('USAGE:')
    print('------')
    print('\t%s <input> <output>' % sys.argv[0])
    print()
    print('ARGUMENTS:')
    print()
    print('\tinput: the directory that contains the csv result for each submission.')
    print()
    print('\toutput: the directory where the final results will be saved in csv format.')
    print()
    exit(1)

sub_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

def filter_float_field(df, col, val, tol=.001):
    assert type(val) in (list, float)
    if type(val) == list:
        assert all([type(f) == float for f in val])
    else:
        val = [val]

    pd_query = ""
    for idx, t in enumerate(val):
        pd_query += "(%s > %f and %s < %f)" % (col, t-tol, col, t+tol)
        if idx < len(val)-1:
            pd_query += ' | '
    df = df.query(pd_query)
    return df


def main():
    filter_threshod_range = (.5, .96, .05)

    print('Processing...')
    tags = ''

    sub_ids = []
    accuracies = []
    precisions = []
    f1s = []
    digits_scores = []

    if filter_threshod_range is not None:
        assert len(filter_threshod_range) == 3
        t1, t2, ts = filter_threshod_range
        thresholds = np.arange(t1, t2, ts).tolist()
        print(filter_threshod_range, thresholds)
        tags += '-filter%.2f_%.2f_%.2f' % (t1, ts, t2)

    for idx, elem in enumerate(sub_path.iterdir()):
        #print('.', end='', flush=True)
        print('Processing: %s' % elem.name)
        sub_df = pd.read_csv(elem)
        if filter_threshod_range is not None:
            sub_df = filter_float_field(sub_df, 'thresh', thresholds)

        sub_ids.append(elem.stem)
        accuracies.append(sub_df['accuracy'].mean())
        precisions.append(sub_df['precision'].mean())
        f1s.append(sub_df['f1'].mean())
        digits_scores.append(sub_df['digits_score'].mean())

        if idx == 0:
            print('Example:')
            print(sub_df)
            sub_df.to_csv(out_path / ('summary-%s%s-example.csv' % (sub_path.name, tags)), index=False)

    print()
    print('Done.')

    data = {
        'submission': sub_ids,
        'mean_accuracy': accuracies,
        'mean_precision': precisions,
        'mean_f1': f1s,
        'mean_digits_score': digits_scores,
    }

    df = pd.DataFrame.from_dict(data)
    df = df.sort_values(by=['mean_accuracy'], ascending=False)
    df.to_csv(out_path / ('summary-%s%s.csv' % (sub_path.name, tags)), index=False)


if __name__ == '__main__':
    main()
