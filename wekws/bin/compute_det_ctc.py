# Copyright (c) 2021 Binbin Zhang(binbzha@qq.com)
#               2022 Shaoqing Yu(954793264@qq.com)
#               2023 Jing Du(thuduj12@163.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse, logging, glob
import json, re, os, numpy as np
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

font = fm.FontProperties(size=15)

def split_mixed_label(input_str):
    tokens = []
    s = input_str.lower()
    while len(s) > 0:
        match = re.match(r'[A-Za-z!?,<>()\']+', s)
        if match is not None:
            word = match.group(0)
        else:
            word = s[0:1]
        tokens.append(word)
        s = s.replace(word, '', 1).strip(' ')
    return tokens


def space_mixed_label(input_str):
    splits = split_mixed_label(input_str)
    space_str = ''.join(f'{sub} ' for sub in splits)
    return space_str.strip()

def load_label_and_score(keywords_list, label_file, score_file):
    # score_table: {uttid: [keywordlist]}
    score_table = {}
    with open(score_file, 'r', encoding='utf8') as fin:
        # read score file and store in table
        for line in fin:
            arr = line.strip().split()
            key = arr[0]
            is_detected = arr[1]
            if is_detected == 'detected':
                if key not in score_table:
                    score_table.update({
                        key: {
                            'kw': space_mixed_label(arr[2]),
                            'confi': float(arr[3])
                        }
                    })
            else:
                if key not in score_table:
                    score_table.update({key: {'kw': 'unknown', 'confi': -1.0}})

    label_lists = []
    with open(label_file, 'r', encoding='utf8') as fin:
        for line in fin:
            obj = json.loads(line.strip())
            label_lists.append(obj)

    # build empty structure for keyword-filler infos
    keyword_filler_table = {}
    for keyword in keywords_list:
        keyword = space_mixed_label(keyword)
        keyword_filler_table[keyword] = {}
        keyword_filler_table[keyword]['keyword_table'] = {}
        keyword_filler_table[keyword]['keyword_duration'] = 0.0
        keyword_filler_table[keyword]['filler_table'] = {}
        keyword_filler_table[keyword]['filler_duration'] = 0.0

    for obj in label_lists:
        assert 'key' in obj
        assert 'wav' in obj
        assert 'tok' in obj   # here we use the tokens
        assert 'duration' in obj

        key = obj['key']
        # wav_file = obj['wav']
        txt = "".join(obj['tok'])
        txt = space_mixed_label(txt)
        txt_regstr_lrblk = ' ' + txt + ' '
        duration = obj['duration']
        assert key in score_table

        for keyword in keywords_list:
            keyword = space_mixed_label(keyword)
            keyword_regstr_lrblk = ' ' + keyword + ' '
            if txt_regstr_lrblk.find(keyword_regstr_lrblk) != -1:
                if keyword == score_table[key]['kw']:
                    keyword_filler_table[keyword]['keyword_table'].update(
                        {key: score_table[key]['confi']})
                else:
                    # uttrance detected but not match this keyword
                    keyword_filler_table[keyword]['keyword_table'].update(
                        {key: -1.0})
                keyword_filler_table[keyword]['keyword_duration'] += duration
            else:
                if keyword == score_table[key]['kw']:
                    keyword_filler_table[keyword]['filler_table'].update(
                        {key: score_table[key]['confi']})
                else:
                    # uttrance if detected, which is not FA for this keyword
                    keyword_filler_table[keyword]['filler_table'].update(
                        {key: -1.0})
                keyword_filler_table[keyword]['filler_duration'] += duration

    return keyword_filler_table

def load_stats_file(stats_file):
    values = []
    with open(stats_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            threshold, fa_per_hour, frr = arr
            values.append([float(fa_per_hour), float(frr) * 100])
    values.reverse()
    return np.array(values)

def plot_det(dets_dir, figure_file, det_title="DetCurve"):
    xlim = '[0,2]'
    # xstep = kwargs.get('xstep', '1')
    ylim = '[15,30]'
    # ystep = kwargs.get('ystep', '5')

    plt.figure(dpi=200)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.size'] = 12

    for file in glob.glob(f'{dets_dir}/*stats*.txt'):
        logging.info(f'reading det data from {file}')
        label = os.path.basename(file).split('.')[0]
        values = load_stats_file(file)
        plt.plot(values[:, 0], values[:, 1], label=label)

    xlim_splits = xlim.strip().replace('[', '').replace(']', '').split(',')
    assert len(xlim_splits) == 2
    ylim_splits = ylim.strip().replace('[', '').replace(']', '').split(',')
    assert len(ylim_splits) == 2

    plt.xlim(float(xlim_splits[0]), float(xlim_splits[1]))
    plt.ylim(float(ylim_splits[0]), float(ylim_splits[1]))

    # plt.xticks(range(0, xlim + x_step, x_step))
    # plt.yticks(range(0, ylim + y_step, y_step))
    plt.xlabel('False Alarm Per Hour')
    plt.ylabel('False Rejection Rate (\\%)')
    plt.title(det_title, fontproperties=font)
    plt.grid(linestyle='--')
    # plt.legend(loc='best', fontsize=6)
    plt.legend(loc='upper right', fontsize=5)
    # plt.show()
    plt.savefig(figure_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='compute det curve')
    parser.add_argument('--test_data', required=True, help='label file')
    parser.add_argument('--keyword', type=str, default=None, help='keyword label')
    parser.add_argument('--score_file', required=True, help='score file')
    parser.add_argument('--step', type=float, default=0.01,
                        help='threshold step')
    parser.add_argument('--window_shift', type=int, default=50,
                        help='window_shift is used to skip the frames after triggered')
    parser.add_argument('--stats_dir',
                        required=False,
                        default=None,
                        help='false reject/alarm stats dir, default in score_file')
    parser.add_argument('--det_curve_path',
                        required=False,
                        default=None,
                        help='det curve path, default is stats_dir/det.png')

    args = parser.parse_args()
    window_shift = args.window_shift
    keywords_list = args.keyword.strip().split(',')
    keyword_filler_table = load_label_and_score(keywords_list, args.test_data, args.score_file)

    for keyword in keywords_list:
        keyword = space_mixed_label(keyword)
        keyword_dur = keyword_filler_table[keyword]['keyword_duration']
        keyword_num = len(keyword_filler_table[keyword]['keyword_table'])
        filler_dur = keyword_filler_table[keyword]['filler_duration']
        filler_num = len(keyword_filler_table[keyword]['filler_table'])
        assert keyword_num > 0, 'Can\'t compute det for {} without positive sample'
        assert filler_num > 0, 'Can\'t compute det for {} without negative sample'

        logging.info('Computing det for {}'.format(keyword))
        logging.info('  Keyword duration: {} Hours, wave number: {}'.format(
            keyword_dur / 3600.0, keyword_num))
        logging.info('  Filler duration: {} Hours'.format(filler_dur / 3600.0))

        if args.stats_dir :
            stats_dir = args.stats_dir
        else:
            stats_dir = os.path.dirname(args.score_file)
        stats_file = os.path.join(stats_dir, 'stats.' + keyword.replace(' ', '_') + '.txt')
        with open(stats_file, 'w', encoding='utf8') as fout:
            threshold = 0.0
            while threshold <= 1.0:
                num_false_reject = 0
                num_true_detect = 0
                # transverse the all keyword_table
                for key, confi in keyword_filler_table[keyword][
                    'keyword_table'].items():
                    if confi < threshold:
                        num_false_reject += 1
                    else:
                        num_true_detect += 1

                num_false_alarm = 0
                # transverse the all filler_table
                for key, confi in keyword_filler_table[keyword][
                    'filler_table'].items():
                    if confi >= threshold:
                        num_false_alarm += 1
                        # print(f'false alarm: {keyword}, {key}, {confi}')

                false_reject_rate = num_false_reject / keyword_num
                true_detect_rate = num_true_detect / keyword_num

                num_false_alarm = max(num_false_alarm, 1e-6)
                false_alarm_per_hour = num_false_alarm / (filler_dur / 3600.0)
                false_alarm_rate = num_false_alarm / filler_num

                fout.write('{:.3f} {:.6f} {:.6f}\n'.format(
                    threshold, false_alarm_per_hour, threshold))
                threshold += args.step
    if args.det_curve_path :
        det_curve_path = args.det_curve_path
    else:
        det_curve_path = os.path.join(stats_dir, 'det.png')
    plot_det(stats_dir, det_curve_path)