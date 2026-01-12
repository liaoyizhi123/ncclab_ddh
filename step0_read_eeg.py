# -*- coding: utf-8 -*-

import matplotlib

matplotlib.use('Agg')

import pickle
from pathlib import Path
import mne, os, re
import numpy as np
import matplotlib
from neuracle_lib.readbdfdata import readbdfdata

EEG_DATA_DIR = '/mnt/dataset4/ddh/movie_data'
film_li = sorted(['agzz', 'lldq2', 'xltfn', 'ymlw', 'rzdf', 'sdd', 'zccsh', 'zgjqxs'])
OUTPUT_DIR = f'/home/liaoyizhi/codes/ncclab_ddh/eeg_data'
notch_freqs = 50.0
bandpass_freqs = [0.1, 75.0]
target_sfreq = 200.0


def read_annotations_bdf(annotations):
    pat = '([+-]\\d+\\.?\\d*)(\x15(\\d+\\.?\\d*))?(\x14.*?)\x14\x00'
    if isinstance(annotations, str):
        with open(annotations, encoding='latin-1') as annot_file:
            triggers = re.findall(pat, annot_file.read())
    else:
        tals = bytearray()
        for chan in annotations:
            this_chan = chan.ravel()
            if this_chan.dtype == np.int32:  # BDF
                this_chan.dtype = np.uint8
                this_chan = this_chan.reshape(-1, 4)
                # Why only keep the first 3 bytes as BDF values
                # are stored with 24 bits (not 32)
                this_chan = this_chan[:, :3].ravel()
                for s in this_chan:
                    tals.extend(s)
            else:
                for s in this_chan:
                    i = int(s)
                    tals.extend(np.uint8([i % 256, i // 256]))

        # use of latin-1 because characters are only encoded for the first 256
        # code points and utf-8 can triggers an "invalid continuation byte"
        # error
        triggers = re.findall(pat, tals.decode('latin-1'))

    events = []
    for ev in triggers:
        onset = float(ev[0])
        duration = float(ev[2]) if ev[2] else 0
        for description in ev[3].split('\x14')[1:]:
            if description:
                events.append([onset, duration, description])
    return zip(*events) if events else (list(), list(), list())


def read_bdf(bdf_path):
    raw = mne.io.read_raw_bdf(bdf_path / 'data.bdf', verbose=False, preload=True)
    # channel positions as numpy (n_channels, 3), NaN if missing

    # 3. filter
    raw.notch_filter(notch_freqs, picks='eeg', verbose=False)
    raw.filter(l_freq=bandpass_freqs[0], h_freq=bandpass_freqs[1], picks='eeg', verbose=False)

    # 5. downsample
    if target_sfreq is not None:
        raw.resample(target_sfreq, verbose=False)

    ch_names = raw.info['ch_names']
    data, _ = raw[: len(ch_names)]
    fs = raw.info['sfreq']
    nchan = raw.info['nchan']

    result = dict()
    try:
        annotationData = mne.io.read_raw_bdf(bdf_path / 'evt.bdf', verbose=False)
        try:
            tal_data = annotationData._read_segment_file([], [], 0, 0, int(annotationData.n_times), None, None)
            print('mne version <= 0.20')
        except:
            idx = np.empty(0, int)
            tal_data = annotationData._read_segment_file(
                np.empty((0, annotationData.n_times)),
                idx,
                0,
                0,
                int(annotationData.n_times),
                np.ones((len(idx), 1)),
                None,
            )
            # print('mne version > 0.20')
        onset, duration, description = read_annotations_bdf(tal_data[0])
        onset = np.array([i * fs for i in onset], dtype=np.int64)
        duration = np.array([int(i) for i in duration], dtype=np.int64)
        desc = np.array([int(i) for i in description], dtype=np.int64)
        events = np.vstack((onset, duration, desc)).T
    except:
        print('not found any event')
        events = []

    result['data'] = data
    result['events'] = events
    result['srate'] = fs
    result['ch_names'] = ch_names
    result['nchan'] = nchan

    return result


def mian():
    # iterate over subjects
    for sub in sorted(list(Path(EEG_DATA_DIR).glob('*'))):
        if sub.name.startswith('.'):
            continue

        if 'sub10_yifan' in str(sub):
            # continue  # FIXME.
            # iterate over sessions
            result_agzz = None
            result_rzdf = None
            result_sdd = None
            result_zccsh = None

            agzz_data = {}
            rzdf_data = {}
            sdd_data = {}
            zccsh_data = {}
            for sess in sorted(list((sub / f'eeg_raw').glob('*'))):
                if sess.name.startswith('.'):
                    continue
                print(f'--------------sssssssssssssssss{sess}')
                if 'agzz' in str(sess):
                    # 找到data.bdf文件
                    data_bdf_file = list(Path(sess).rglob('data.bdf'))[0]
                    evt_bdf_file = list(Path(sess).rglob('evt.bdf'))[0]

                    result_agzz = read_bdf(data_bdf_file.parent)
                    # print(result['events'])
                    events = result_agzz['events']
                    for i, event in enumerate(events):
                        try:
                            if events[i][2] == 8 and events[i + 1][2] == 16 and events[i + 2][2] == 32:
                                print(
                                    f'--------------fixation time {(events[i+1][0]-events[i][0])/result_agzz['srate']}s, {((events[i+1][0]-events[i][0])/result_agzz['srate'])//60}minuts, {((events[i+1][0]-events[i][0])/result_agzz['srate'])%60: .4f}seconds'
                                )
                                print(
                                    f'--------------movie time {(events[i+2][0]-events[i+1][0])/result_agzz['srate']}s, {((events[i+2][0]-events[i+1][0])/result_agzz['srate'])//60}minuts, {((events[i+2][0]-events[i+1][0])/result_agzz['srate'])%60: .4f}seconds'
                                )
                                fixation_start = events[i][0]
                                start_play = events[i + 1][0]
                                rating_page_1 = events[i + 2][0]
                                # EEG 数据
                                eeg_segment = result_agzz['data'][:, start_play:rating_page_1]
                                fixation_segment = result_agzz['data'][
                                    :, int(fixation_start) : int(fixation_start) + int(2 * 60 * result_agzz['srate'])
                                ]
                                # 往eeg_segment['data']这个listappend数据，如果不存在则创建一个新的list
                                if 'data' not in agzz_data:
                                    agzz_data['data'] = []
                                agzz_data['data'].append(eeg_segment)
                                if 'fixation' not in agzz_data:
                                    agzz_data['fixation'] = []
                                agzz_data['fixation'].append(fixation_segment)
                                pass
                        except IndexError:
                            pass

                if 'rzdf' in str(sess):
                    # 找到data.bdf文件
                    data_bdf_file = list(Path(sess).rglob('data.bdf'))[0]
                    evt_bdf_file = list(Path(sess).rglob('evt.bdf'))[0]
                    result_rzdf = read_bdf(data_bdf_file.parent)
                    events = result_rzdf['events']
                    for i, event in enumerate(events):
                        try:
                            if events[i][2] == 8 and events[i + 1][2] == 16 and events[i + 2][2] == 32:
                                print(
                                    f'--------------fixation time {(events[i+1][0]-events[i][0])/result_rzdf['srate']}s, {((events[i+1][0]-events[i][0])/result_rzdf['srate'])//60}minuts, {((events[i+1][0]-events[i][0])/result_rzdf['srate'])%60: .4f}seconds'
                                )
                                print(
                                    f'--------------movie time {(events[i+2][0]-events[i+1][0])/result_rzdf['srate']}s, {((events[i+2][0]-events[i+1][0])/result_rzdf['srate'])//60}minuts, {((events[i+2][0]-events[i+1][0])/result_rzdf['srate'])%60: .4f}seconds'
                                )
                                fixation_start = events[i][0]
                                start_play = events[i + 1][0]
                                rating_page_1 = events[i + 2][0]
                                # EEG 数据
                                eeg_segment = result_rzdf['data'][:, start_play:rating_page_1]
                                fixation_segment = result_rzdf['data'][
                                    :, int(fixation_start) : int(fixation_start) + int(2 * 60 * result_rzdf['srate'])
                                ]
                                if 'data' not in rzdf_data:
                                    rzdf_data['data'] = []
                                rzdf_data['data'].append(eeg_segment)
                                if 'fixation' not in rzdf_data:
                                    rzdf_data['fixation'] = []
                                rzdf_data['fixation'].append(fixation_segment)
                        except IndexError:
                            pass

                if 'sdd' in str(sess):
                    data_bdf_file = list(Path(sess).rglob('data.bdf'))[0]
                    evt_bdf_file = list(Path(sess).rglob('evt.bdf'))[0]
                    result_sdd = read_bdf(data_bdf_file.parent)
                    events = result_sdd['events']
                    # 删除events的第8，9，10行
                    events = np.delete(events, [8, 9, 10], axis=0)

                    for i, event in enumerate(events):
                        try:
                            if events[i][2] == 8 and events[i + 1][2] == 16 and events[i + 2][2] == 32:
                                print(
                                    f'--------------fixation time {(events[i+1][0]-events[i][0])/result_sdd['srate']}s, {((events[i+1][0]-events[i][0])/result_sdd['srate'])//60}minuts, {((events[i+1][0]-events[i][0])/result_sdd['srate'])%60: .4f}seconds'
                                )
                                print(
                                    f'--------------movie time {(events[i+2][0]-events[i+1][0])/result_sdd['srate']}s, {((events[i+2][0]-events[i+1][0])/result_sdd['srate'])//60}minuts, {((events[i+2][0]-events[i+1][0])/result_sdd['srate'])%60: .4f}seconds'
                                )
                                fixation_start = events[i][0]
                                start_play = events[i + 1][0]
                                rating_page_1 = events[i + 2][0]
                                # EEG 数据
                                eeg_segment = result_sdd['data'][:, start_play:rating_page_1]
                                fixation_segment = result_sdd['data'][
                                    :, int(fixation_start) : int(fixation_start) + int(2 * 60 * result_sdd['srate'])
                                ]
                                if 'data' not in sdd_data:
                                    sdd_data['data'] = []
                                sdd_data['data'].append(eeg_segment)
                                if 'fixation' not in sdd_data:
                                    sdd_data['fixation'] = []
                                sdd_data['fixation'].append(fixation_segment)
                        except IndexError:
                            pass

                if 'zccsh' in str(sess):
                    data_bdf_file = list(Path(sess).rglob('data.bdf'))[0]
                    evt_bdf_file = list(Path(sess).rglob('evt.bdf'))[0]
                    result_zccsh = read_bdf(data_bdf_file.parent)
                    events = result_zccsh['events']
                    for i, event in enumerate(events):
                        try:
                            if events[i][2] == 8 and events[i + 1][2] == 16 and events[i + 2][2] == 32:
                                print(
                                    f'--------------fixation time {(events[i+1][0]-events[i][0])/result_zccsh['srate']}s, {((events[i+1][0]-events[i][0])/result_zccsh['srate'])//60}minuts, {((events[i+1][0]-events[i][0])/result_zccsh['srate'])%60: .4f}seconds'
                                )
                                print(
                                    f'--------------movie time {(events[i+2][0]-events[i+1][0])/result_zccsh['srate']}s, {((events[i+2][0]-events[i+1][0])/result_zccsh['srate'])//60}minuts, {((events[i+2][0]-events[i+1][0])/result_zccsh['srate'])%60: .4f}seconds'
                                )
                                fixation_start = events[i][0]
                                start_play = events[i + 1][0]
                                rating_page_1 = events[i + 2][0]
                                # EEG 数据
                                eeg_segment = result_zccsh['data'][:, start_play:rating_page_1]
                                fixation_segment = result_zccsh['data'][
                                    :, int(fixation_start) : int(fixation_start) + int(2 * 60 * result_zccsh['srate'])
                                ]
                                if 'data' not in zccsh_data:
                                    zccsh_data['data'] = []
                                zccsh_data['data'].append(eeg_segment)
                                if 'fixation' not in zccsh_data:
                                    zccsh_data['fixation'] = []
                                zccsh_data['fixation'].append(fixation_segment)
                        except IndexError:
                            pass

            agzz_data['srate'] = result_agzz['srate']
            agzz_data['ch_names'] = result_agzz['ch_names']
            agzz_data['nchan'] = result_agzz['nchan']

            rzdf_data['srate'] = result_rzdf['srate']
            rzdf_data['ch_names'] = result_rzdf['ch_names']
            rzdf_data['nchan'] = result_rzdf['nchan']

            sdd_data['srate'] = result_sdd['srate']
            sdd_data['ch_names'] = result_sdd['ch_names']
            sdd_data['nchan'] = result_sdd['nchan']

            zccsh_data['srate'] = result_zccsh['srate']
            zccsh_data['ch_names'] = result_zccsh['ch_names']
            zccsh_data['nchan'] = result_zccsh['nchan']

            # save data
            sub_name = sess.parent.parent.name
            save_path = Path(OUTPUT_DIR) / sub_name
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / 'agzz.pkl', 'wb') as f:
                pickle.dump(agzz_data, f)
            with open(save_path / 'rzdf.pkl', 'wb') as f:
                pickle.dump(rzdf_data, f)
            with open(save_path / 'sdd.pkl', 'wb') as f:
                pickle.dump(sdd_data, f)
            with open(save_path / 'zccsh.pkl', 'wb') as f:
                pickle.dump(zccsh_data, f)

        elif 'sub11_yingyue' in str(sub):
            # continue # FIXME.
            # iterate over sessions
            result_agzz = None
            result_rzdf = None
            result_zccsh = None
            result_zgjqxs = None

            agzz_data = {}
            rzdf_data = {}
            zccsh_data = {}
            zgjqxs_data = {}
            for sess in sorted(list((sub / f'eeg_raw').glob('*'))):
                if sess.name.startswith('.'):
                    continue
                print(f'--------------sssssssssssssssss{sess}')
                if 'agzz' in str(sess):
                    # 找到data.bdf文件
                    data_bdf_file = list(Path(sess).rglob('data.bdf'))[0]
                    evt_bdf_file = list(Path(sess).rglob('evt.bdf'))[0]

                    result_agzz = read_bdf(data_bdf_file.parent)
                    # print(result['events'])
                    events = result_agzz['events']
                    for i, event in enumerate(events):
                        try:
                            if events[i][2] == 8 and events[i + 1][2] == 16 and events[i + 2][2] == 32:
                                print(
                                    f'--------------fixation time {(events[i+1][0]-events[i][0])/result_agzz['srate']}s, {((events[i+1][0]-events[i][0])/result_agzz['srate'])//60}minuts, {((events[i+1][0]-events[i][0])/result_agzz['srate'])%60: .4f}seconds'
                                )
                                print(
                                    f'--------------movie time {(events[i+2][0]-events[i+1][0])/result_agzz['srate']}s, {((events[i+2][0]-events[i+1][0])/result_agzz['srate'])//60}minuts, {((events[i+2][0]-events[i+1][0])/result_agzz['srate'])%60: .4f}seconds'
                                )
                                fixation_start = events[i][0]
                                start_play = events[i + 1][0]
                                rating_page_1 = events[i + 2][0]
                                # EEG 数据
                                eeg_segment = result_agzz['data'][:, start_play:rating_page_1]
                                fixation_segment = result_agzz['data'][
                                    :, int(fixation_start) : int(fixation_start) + int(2 * 60 * result_agzz['srate'])
                                ]
                                # 往eeg_segment['data']这个listappend数据，如果不存在则创建一个新的list
                                if 'data' not in agzz_data:
                                    agzz_data['data'] = []
                                agzz_data['data'].append(eeg_segment)
                                if 'fixation' not in agzz_data:
                                    agzz_data['fixation'] = []
                                agzz_data['fixation'].append(fixation_segment)
                                pass
                        except IndexError:
                            pass

                if 'rzdf' in str(sess):
                    # 找到data.bdf文件
                    data_bdf_file = list(Path(sess).rglob('data.bdf'))[0]
                    evt_bdf_file = list(Path(sess).rglob('evt.bdf'))[0]
                    result_rzdf = read_bdf(data_bdf_file.parent)
                    events = result_rzdf['events']
                    for i, event in enumerate(events):
                        try:
                            if events[i][2] == 8 and events[i + 1][2] == 16 and events[i + 2][2] == 32:
                                print(
                                    f'--------------fixation time {(events[i+1][0]-events[i][0])/result_rzdf['srate']}s, {((events[i+1][0]-events[i][0])/result_rzdf['srate'])//60}minuts, {((events[i+1][0]-events[i][0])/result_rzdf['srate'])%60: .4f}seconds'
                                )
                                print(
                                    f'--------------movie time {(events[i+2][0]-events[i+1][0])/result_rzdf['srate']}s, {((events[i+2][0]-events[i+1][0])/result_rzdf['srate'])//60}minuts, {((events[i+2][0]-events[i+1][0])/result_rzdf['srate'])%60: .4f}seconds'
                                )
                                fixation_start = events[i][0]
                                start_play = events[i + 1][0]
                                rating_page_1 = events[i + 2][0]
                                # EEG 数据
                                eeg_segment = result_rzdf['data'][:, start_play:rating_page_1]
                                fixation_segment = result_rzdf['data'][
                                    :, int(fixation_start) : int(fixation_start) + int(2 * 60 * result_rzdf['srate'])
                                ]
                                if 'data' not in rzdf_data:
                                    rzdf_data['data'] = []
                                rzdf_data['data'].append(eeg_segment)
                                if 'fixation' not in rzdf_data:
                                    rzdf_data['fixation'] = []
                                rzdf_data['fixation'].append(fixation_segment)
                        except IndexError:
                            pass

                if 'zccsh' in str(sess):
                    data_bdf_file = list(Path(sess).rglob('data.bdf'))[0]
                    evt_bdf_file = list(Path(sess).rglob('evt.bdf'))[0]
                    result_zccsh = read_bdf(data_bdf_file.parent)
                    events = result_zccsh['events']
                    for i, event in enumerate(events):
                        try:
                            if events[i][2] == 8 and events[i + 1][2] == 16 and events[i + 2][2] == 32:
                                print(
                                    f'--------------fixation time {(events[i+1][0]-events[i][0])/result_zccsh['srate']}s, {((events[i+1][0]-events[i][0])/result_zccsh['srate'])//60}minuts, {((events[i+1][0]-events[i][0])/result_zccsh['srate'])%60: .4f}seconds'
                                )
                                print(
                                    f'--------------movie time {(events[i+2][0]-events[i+1][0])/result_zccsh['srate']}s, {((events[i+2][0]-events[i+1][0])/result_zccsh['srate'])//60}minuts, {((events[i+2][0]-events[i+1][0])/result_zccsh['srate'])%60: .4f}seconds'
                                )
                                fixation_start = events[i][0]
                                start_play = events[i + 1][0]
                                rating_page_1 = events[i + 2][0]
                                # EEG 数据
                                eeg_segment = result_zccsh['data'][:, start_play:rating_page_1]
                                fixation_segment = result_zccsh['data'][
                                    :, int(fixation_start) : int(fixation_start) + int(2 * 60 * result_zccsh['srate'])
                                ]
                                if 'data' not in zccsh_data:
                                    zccsh_data['data'] = []
                                zccsh_data['data'].append(eeg_segment)
                                if 'fixation' not in zccsh_data:
                                    zccsh_data['fixation'] = []
                                zccsh_data['fixation'].append(fixation_segment)
                        except IndexError:
                            pass

                if 'zgjqxs' in str(sess):
                    data_bdf_file = list(Path(sess).rglob('data.bdf'))[0]
                    evt_bdf_file = list(Path(sess).rglob('evt.bdf'))[0]
                    result_zgjqxs = read_bdf(data_bdf_file.parent)
                    events = result_zgjqxs['events']
                    for i, event in enumerate(events):
                        try:
                            if events[i][2] == 8 and events[i + 1][2] == 16 and events[i + 2][2] == 32:
                                print(
                                    f'--------------fixation time {(events[i+1][0]-events[i][0])/result_zgjqxs['srate']}s, {((events[i+1][0]-events[i][0])/result_zgjqxs['srate'])//60}minuts, {((events[i+1][0]-events[i][0])/result_zgjqxs['srate'])%60: .4f}seconds'
                                )
                                print(
                                    f'--------------movie time {(events[i+2][0]-events[i+1][0])/result_zgjqxs['srate']}s, {((events[i+2][0]-events[i+1][0])/result_zgjqxs['srate'])//60}minuts, {((events[i+2][0]-events[i+1][0])/result_zgjqxs['srate'])%60: .4f}seconds'
                                )
                                fixation_start = events[i][0]
                                start_play = events[i + 1][0]
                                rating_page_1 = events[i + 2][0]
                                # EEG 数据
                                eeg_segment = result_zgjqxs['data'][:, start_play:rating_page_1]
                                fixation_segment = result_zgjqxs['data'][
                                    :, int(fixation_start) : int(fixation_start) + int(2 * 60 * result_zgjqxs['srate'])
                                ]
                                if 'data' not in zgjqxs_data:
                                    zgjqxs_data['data'] = []
                                zgjqxs_data['data'].append(eeg_segment)
                                if 'fixation' not in zgjqxs_data:
                                    zgjqxs_data['fixation'] = []
                                zgjqxs_data['fixation'].append(fixation_segment)
                        except IndexError:
                            pass

            agzz_data['srate'] = result_agzz['srate']
            agzz_data['ch_names'] = result_agzz['ch_names']
            agzz_data['nchan'] = result_agzz['nchan']

            rzdf_data['srate'] = result_rzdf['srate']
            rzdf_data['ch_names'] = result_rzdf['ch_names']
            rzdf_data['nchan'] = result_rzdf['nchan']

            zccsh_data['srate'] = result_zccsh['srate']
            zccsh_data['ch_names'] = result_zccsh['ch_names']
            zccsh_data['nchan'] = result_zccsh['nchan']

            zgjqxs_data['srate'] = result_zgjqxs['srate']
            zgjqxs_data['ch_names'] = result_zgjqxs['ch_names']
            zgjqxs_data['nchan'] = result_zgjqxs['nchan']  # 采样率为1000

            # save data
            sub_name = sess.parent.parent.name
            save_path = Path(OUTPUT_DIR) / sub_name
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / 'agzz.pkl', 'wb') as f:
                pickle.dump(agzz_data, f)
            with open(save_path / 'rzdf.pkl', 'wb') as f:
                pickle.dump(rzdf_data, f)
            with open(save_path / 'zccsh.pkl', 'wb') as f:
                pickle.dump(zccsh_data, f)
            with open(save_path / 'zgjqxs.pkl', 'wb') as f:
                pickle.dump(zgjqxs_data, f)

        elif 'sub13_penghe' in str(sub):
            # continue  # FIXME.
            pass
            result_agzz = None
            result_zccsh = None

            agzz_data = {}
            zccsh_data = {}
            # 先删掉多余标签
            for sess in sorted(list((sub / f'eeg_raw').glob('*'))):
                if sess.name.startswith('.'):
                    continue
                print(f'--------------sssssssssssssssss{sess}')
                if 'agzz' in str(sess):
                    # 找到data.bdf文件
                    data_bdf_file = list(Path(sess).rglob('data.bdf'))[0]
                    evt_bdf_file = list(Path(sess).rglob('evt.bdf'))[0]

                    result_agzz = read_bdf(data_bdf_file.parent)
                    # print(result['events'])
                    events = result_agzz['events']
                    events = np.delete(events, [8, 9, 10], axis=0)
                    for i, event in enumerate(events):
                        try:
                            if events[i][2] == 8 and events[i + 1][2] == 16 and events[i + 2][2] == 32:
                                print(
                                    f'--------------fixation time {(events[i+1][0]-events[i][0])/result_agzz['srate']}s, {((events[i+1][0]-events[i][0])/result_agzz['srate'])//60}minuts, {((events[i+1][0]-events[i][0])/result_agzz['srate'])%60: .4f}seconds'
                                )
                                print(
                                    f'--------------movie time {(events[i+2][0]-events[i+1][0])/result_agzz['srate']}s, {((events[i+2][0]-events[i+1][0])/result_agzz['srate'])//60}minuts, {((events[i+2][0]-events[i+1][0])/result_agzz['srate'])%60: .4f}seconds'
                                )
                                fixation_start = events[i][0]
                                start_play = events[i + 1][0]
                                rating_page_1 = events[i + 2][0]
                                # EEG 数据
                                eeg_segment = result_agzz['data'][:, start_play:rating_page_1]
                                fixation_segment = result_agzz['data'][
                                    :, int(fixation_start) : int(fixation_start) + int(2 * 60 * result_agzz['srate'])
                                ]
                                # 往eeg_segment['data']这个listappend数据，如果不存在则创建一个新的list
                                if 'data' not in agzz_data:
                                    agzz_data['data'] = []
                                agzz_data['data'].append(eeg_segment)
                                if 'fixation' not in agzz_data:
                                    agzz_data['fixation'] = []
                                agzz_data['fixation'].append(fixation_segment)
                                pass
                        except IndexError:
                            pass

                if 'zccsh' in str(sess):
                    data_bdf_file = list(Path(sess).rglob('data.bdf'))[0]
                    evt_bdf_file = list(Path(sess).rglob('evt.bdf'))[0]
                    result_zccsh = read_bdf(data_bdf_file.parent)
                    events = result_zccsh['events']
                    events = np.delete(events, [18, 19, 20], axis=0)
                    for i, event in enumerate(events):
                        try:
                            if events[i][2] == 8 and events[i + 1][2] == 16 and events[i + 2][2] == 32:
                                print(
                                    f'--------------fixation time {(events[i+1][0]-events[i][0])/result_zccsh['srate']}s, {((events[i+1][0]-events[i][0])/result_zccsh['srate'])//60}minuts, {((events[i+1][0]-events[i][0])/result_zccsh['srate'])%60: .4f}seconds'
                                )
                                print(
                                    f'--------------movie time {(events[i+2][0]-events[i+1][0])/result_zccsh['srate']}s, {((events[i+2][0]-events[i+1][0])/result_zccsh['srate'])//60}minuts, {((events[i+2][0]-events[i+1][0])/result_zccsh['srate'])%60: .4f}seconds'
                                )
                                fixation_start = events[i][0]
                                start_play = events[i + 1][0]
                                rating_page_1 = events[i + 2][0]
                                # EEG 数据
                                eeg_segment = result_zccsh['data'][:, start_play:rating_page_1]
                                fixation_segment = result_zccsh['data'][
                                    :, int(fixation_start) : int(fixation_start) + int(2 * 60 * result_zccsh['srate'])
                                ]
                                if 'data' not in zccsh_data:
                                    zccsh_data['data'] = []
                                zccsh_data['data'].append(eeg_segment)
                                if 'fixation' not in zccsh_data:
                                    zccsh_data['fixation'] = []
                                zccsh_data['fixation'].append(fixation_segment)
                        except IndexError:
                            pass

            agzz_data['srate'] = result_agzz['srate']
            agzz_data['ch_names'] = result_agzz['ch_names']
            agzz_data['nchan'] = result_agzz['nchan']

            zccsh_data['srate'] = result_zccsh['srate']
            zccsh_data['ch_names'] = result_zccsh['ch_names']
            zccsh_data['nchan'] = result_zccsh['nchan']

            # save data
            sub_name = sess.parent.parent.name
            save_path = Path(OUTPUT_DIR) / sub_name
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / 'agzz.pkl', 'wb') as f:
                pickle.dump(agzz_data, f)
            with open(save_path / 'zccsh.pkl', 'wb') as f:
                pickle.dump(zccsh_data, f)

        elif 'sub4_qianyi' in str(sub):
            # continue # FIXME.
            agzz_data = {}
            lldq2_data = {}
            xltfn_data = {}
            ymlw_data = {}

            result_agzz = None
            result_lldq2 = None
            result_xltfn = None
            result_ymlw = None

            for sess in sorted(list((sub / f'eeg_raw').glob('*'))):
                if sess.name.startswith('.'):
                    continue
                print(f'--------------sssssssssssssssss{sess}')
                if 'agzz' in str(sess):
                    # 找到data.bdf文件
                    data_bdf_file = list(Path(sess).rglob('data.bdf'))[0]
                    evt_bdf_file = list(Path(sess).rglob('evt.bdf'))[0]

                    result_agzz = read_bdf(data_bdf_file.parent)
                    # print(result['events'])
                    events = result_agzz['events']
                    for i, event in enumerate(events):
                        try:
                            if events[i][2] == 8 and events[i + 1][2] == 16 and events[i + 2][2] == 32:
                                print(
                                    f'--------------fixation time {(events[i+1][0]-events[i][0])/result_agzz['srate']}s, {((events[i+1][0]-events[i][0])/result_agzz['srate'])//60}minuts, {((events[i+1][0]-events[i][0])/result_agzz['srate'])%60: .4f}seconds'
                                )
                                print(
                                    f'--------------movie time {(events[i+2][0]-events[i+1][0])/result_agzz['srate']}s, {((events[i+2][0]-events[i+1][0])/result_agzz['srate'])//60}minuts, {((events[i+2][0]-events[i+1][0])/result_agzz['srate'])%60: .4f}seconds'
                                )
                                fixation_start = events[i][0]
                                start_play = events[i + 1][0]
                                rating_page_1 = events[i + 2][0]
                                # EEG 数据
                                eeg_segment = result_agzz['data'][:, start_play:rating_page_1]
                                fixation_segment = result_agzz['data'][
                                    :, int(fixation_start) : int(fixation_start) + int(2 * 60 * result_agzz['srate'])
                                ]
                                # 往eeg_segment['data']这个listappend数据，如果不存在则创建一个新的list
                                if 'data' not in agzz_data:
                                    agzz_data['data'] = []
                                agzz_data['data'].append(eeg_segment)
                                if 'fixation' not in agzz_data:
                                    agzz_data['fixation'] = []
                                agzz_data['fixation'].append(fixation_segment)
                                pass
                        except IndexError:
                            pass

                if 'lldq2' in str(sess):
                    data_bdf_file = list(Path(sess).rglob('data.bdf'))[0]
                    evt_bdf_file = list(Path(sess).rglob('evt.bdf'))[0]
                    result_lldq2 = read_bdf(data_bdf_file.parent)
                    events = result_lldq2['events']
                    events = np.delete(events, [28, 29, 30], axis=0)
                    for i, event in enumerate(events):
                        try:
                            if events[i][2] == 8 and events[i + 1][2] == 16 and events[i + 2][2] == 32:
                                print(
                                    f'--------------fixation time {(events[i+1][0]-events[i][0])/result_lldq2['srate']}s, {((events[i+1][0]-events[i][0])/result_lldq2['srate'])//60}minuts, {((events[i+1][0]-events[i][0])/result_lldq2['srate'])%60: .4f}seconds'
                                )
                                print(
                                    f'--------------movie time {(events[i+2][0]-events[i+1][0])/result_lldq2['srate']}s, {((events[i+2][0]-events[i+1][0])/result_lldq2['srate'])//60}minuts, {((events[i+2][0]-events[i+1][0])/result_lldq2['srate'])%60: .4f}seconds'
                                )
                                fixation_start = events[i][0]
                                start_play = events[i + 1][0]
                                rating_page_1 = events[i + 2][0]
                                # EEG 数据
                                eeg_segment = result_lldq2['data'][:, start_play:rating_page_1]
                                fixation_segment = result_lldq2['data'][
                                    :, int(fixation_start) : int(fixation_start) + int(2 * 60 * result_lldq2['srate'])
                                ]
                                if 'data' not in lldq2_data:
                                    lldq2_data['data'] = []
                                lldq2_data['data'].append(eeg_segment)
                                if 'fixation' not in lldq2_data:
                                    lldq2_data['fixation'] = []
                                lldq2_data['fixation'].append(fixation_segment)
                        except IndexError:
                            pass

                # if 'xltfn' in str(sess):
                #     data_bdf_file = list(Path(sess).rglob('data.bdf'))[0]
                #     evt_bdf_file = list(Path(sess).rglob('evt.bdf'))[0]
                #     result_xltfn = read_bdf(data_bdf_file.parent)
                #     events = result_xltfn['events']
                #     for i, event in enumerate(events):
                #         try:
                #             if events[i][2] == 8 and events[i + 1][2] == 16 and events[i + 2][2] == 32:
                #                 print(f'--------------fixation time {(events[i+1][0]-events[i][0])/result_xltfn['srate']}s, {((events[i+1][0]-events[i][0])/result_xltfn['srate'])//60}minuts, {((events[i+1][0]-events[i][0])/result_xltfn['srate'])%60: .4f}seconds')
                #                 print(f'--------------movie time {(events[i+2][0]-events[i+1][0])/result_xltfn['srate']}s, {((events[i+2][0]-events[i+1][0])/result_xltfn['srate'])//60}minuts, {((events[i+2][0]-events[i+1][0])/result_xltfn['srate'])%60: .4f}seconds')
                #                 fixation_start = events[i][0]
                #                 start_play = events[i + 1][0]
                #                 rating_page_1 = events[i + 2][0]
                #                 # EEG 数据
                #                 eeg_segment = result_xltfn['data'][:, start_play:rating_page_1]
                #                 fixation_segment = result_xltfn['data'][:, int(fixation_start) : int(fixation_start)+int(2 * 60 * result_xltfn['srate'])]
                #                 if 'data' not in xltfn_data:
                #                     xltfn_data['data'] = []
                #                 xltfn_data['data'].append(eeg_segment)
                #                 if 'fixation' not in xltfn_data:
                #                     xltfn_data['fixation'] = []
                #                 xltfn_data['fixation'].append(fixation_segment)
                #         except IndexError:
                #             pass
                if 'ymlw' in str(sess):
                    data_bdf_file = list(Path(sess).rglob('data.bdf'))[0]
                    evt_bdf_file = list(Path(sess).rglob('evt.bdf'))[0]
                    result_ymlw = read_bdf(data_bdf_file.parent)
                    events = result_ymlw['events']
                    for i, event in enumerate(events):
                        try:
                            if events[i][2] == 8 and events[i + 1][2] == 16 and events[i + 2][2] == 32:
                                print(
                                    f'--------------fixation time {(events[i+1][0]-events[i][0])/result_ymlw['srate']}s, {((events[i+1][0]-events[i][0])/result_ymlw['srate'])//60}minuts, {((events[i+1][0]-events[i][0])/result_ymlw['srate'])%60: .4f}seconds'
                                )
                                print(
                                    f'--------------movie time {(events[i+2][0]-events[i+1][0])/result_ymlw['srate']}s, {((events[i+2][0]-events[i+1][0])/result_ymlw['srate'])//60}minuts, {((events[i+2][0]-events[i+1][0])/result_ymlw['srate'])%60: .4f}seconds'
                                )
                                fixation_start = events[i][0]
                                start_play = events[i + 1][0]
                                rating_page_1 = events[i + 2][0]
                                # EEG 数据
                                eeg_segment = result_ymlw['data'][:, start_play:rating_page_1]
                                fixation_segment = result_ymlw['data'][
                                    :, int(fixation_start) : int(fixation_start) + int(2 * 60 * result_ymlw['srate'])
                                ]
                                if 'data' not in ymlw_data:
                                    ymlw_data['data'] = []
                                ymlw_data['data'].append(eeg_segment)
                                if 'fixation' not in ymlw_data:
                                    ymlw_data['fixation'] = []
                                ymlw_data['fixation'].append(fixation_segment)
                        except IndexError:
                            pass

            agzz_data['srate'] = result_agzz['srate']
            agzz_data['ch_names'] = result_agzz['ch_names']
            agzz_data['nchan'] = result_agzz['nchan']

            lldq2_data['srate'] = result_lldq2['srate']
            lldq2_data['ch_names'] = result_lldq2['ch_names']
            lldq2_data['nchan'] = result_lldq2['nchan']

            # xltfn_data['srate'] = result_xltfn['srate']
            # xltfn_data['ch_names'] = result_xltfn['ch_names']
            # xltfn_data['nchan'] = result_xltfn['nchan']

            ymlw_data['srate'] = result_ymlw['srate']
            ymlw_data['ch_names'] = result_ymlw['ch_names']
            ymlw_data['nchan'] = result_ymlw['nchan']

            # save data
            sub_name = sess.parent.parent.name
            save_path = Path(OUTPUT_DIR) / sub_name
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / 'agzz.pkl', 'wb') as f:
                pickle.dump(agzz_data, f)
            with open(save_path / 'lldq2.pkl', 'wb') as f:
                pickle.dump(lldq2_data, f)
            # with open(save_path/'xltfn.pkl', 'wb') as f:
            #     pickle.dump(xltfn_data, f)
            with open(save_path / 'ymlw.pkl', 'wb') as f:
                pickle.dump(ymlw_data, f)

        elif 'sub5_xulong' in str(sub):
            # continue # FIXME.
            agzz_data = {}
            ymlw_data = {}

            result_agzz = None
            result_ymlw = None

            for sess in sorted(list((sub / f'eeg_raw').glob('*'))):
                if sess.name.startswith('.'):
                    continue
                print(f'--------------sssssssssssssssss{sess}')
                if 'agzz' in str(sess):
                    # 找到data.bdf文件
                    data_bdf_file = list(Path(sess).rglob('data.bdf'))[0]
                    evt_bdf_file = list(Path(sess).rglob('evt.bdf'))[0]
                    result_agzz = read_bdf(data_bdf_file.parent)
                    # print(result['events'])
                    events = result_agzz['events']
                    for i, event in enumerate(events):
                        try:
                            if events[i][2] == 8 and events[i + 1][2] == 16 and events[i + 2][2] == 32:
                                print(
                                    f'--------------fixation time {(events[i+1][0]-events[i][0])/result_agzz['srate']}s, {((events[i+1][0]-events[i][0])/result_agzz['srate'])//60}minuts, {((events[i+1][0]-events[i][0])/result_agzz['srate'])%60: .4f}seconds'
                                )
                                print(
                                    f'--------------movie time {(events[i+2][0]-events[i+1][0])/result_agzz['srate']}s, {((events[i+2][0]-events[i+1][0])/result_agzz['srate'])//60}minuts, {((events[i+2][0]-events[i+1][0])/result_agzz['srate'])%60: .4f}seconds'
                                )
                                fixation_start = events[i][0]
                                start_play = events[i + 1][0]
                                rating_page_1 = events[i + 2][0]
                                # EEG 数据
                                eeg_segment = result_agzz['data'][:, start_play:rating_page_1]
                                fixation_segment = result_agzz['data'][
                                    :, int(fixation_start) : int(fixation_start) + int(2 * 60 * result_agzz['srate'])
                                ]
                                # 往eeg_segment['data']这个listappend数据，如果不存在则创建一个新的list
                                if 'data' not in agzz_data:
                                    agzz_data['data'] = []
                                agzz_data['data'].append(eeg_segment)
                                if 'fixation' not in agzz_data:
                                    agzz_data['fixation'] = []
                                agzz_data['fixation'].append(fixation_segment)
                                pass
                        except IndexError:
                            pass
                if 'ymlw' in str(sess):
                    data_bdf_file = list(Path(sess).rglob('data.bdf'))[0]
                    evt_bdf_file = list(Path(sess).rglob('evt.bdf'))[0]
                    result_ymlw = read_bdf(data_bdf_file.parent)
                    events = result_ymlw['events']
                    events = np.delete(events, [27, 28, 29], axis=0)
                    for i, event in enumerate(events):
                        try:
                            if events[i][2] == 8 and events[i + 1][2] == 16 and events[i + 2][2] == 32:
                                print(
                                    f'--------------fixation time {(events[i+1][0]-events[i][0])/result_ymlw['srate']}s, {((events[i+1][0]-events[i][0])/result_ymlw['srate'])//60}minuts, {((events[i+1][0]-events[i][0])/result_ymlw['srate'])%60: .4f}seconds'
                                )
                                print(
                                    f'--------------movie time {(events[i+2][0]-events[i+1][0])/result_ymlw['srate']}s, {((events[i+2][0]-events[i+1][0])/result_ymlw['srate'])//60}minuts, {((events[i+2][0]-events[i+1][0])/result_ymlw['srate'])%60: .4f}seconds'
                                )
                                fixation_start = events[i][0]
                                start_play = events[i + 1][0]
                                rating_page_1 = events[i + 2][0]
                                # EEG 数据
                                eeg_segment = result_ymlw['data'][:, start_play:rating_page_1]
                                fixation_segment = result_ymlw['data'][
                                    :, int(fixation_start) : int(fixation_start) + int(2 * 60 * result_ymlw['srate'])
                                ]
                                if 'data' not in ymlw_data:
                                    ymlw_data['data'] = []
                                ymlw_data['data'].append(eeg_segment)
                                if 'fixation' not in ymlw_data:
                                    ymlw_data['fixation'] = []
                                ymlw_data['fixation'].append(fixation_segment)
                        except IndexError:
                            pass
            agzz_data['srate'] = result_agzz['srate']
            agzz_data['ch_names'] = result_agzz['ch_names']
            agzz_data['nchan'] = result_agzz['nchan']

            ymlw_data['srate'] = result_ymlw['srate']
            ymlw_data['ch_names'] = result_ymlw['ch_names']
            ymlw_data['nchan'] = result_ymlw['nchan']

            # save data
            sub_name = sess.parent.parent.name
            save_path = Path(OUTPUT_DIR) / sub_name
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / 'agzz.pkl', 'wb') as f:
                pickle.dump(agzz_data, f)
            with open(save_path / 'ymlw.pkl', 'wb') as f:
                pickle.dump(ymlw_data, f)

        elif 'sub9_tongyu' in str(sub):
            agzz_data = {}
            rzdf_data = {}

            result_agzz = None
            result_rzdf = None

            for sess in sorted(list((sub / f'eeg_raw').glob('*'))):
                if sess.name.startswith('.'):
                    continue
                print(f'--------------sssssssssssssssss{sess}')
                if 'agzz' in str(sess):
                    # 找到data.bdf文件
                    data_bdf_file = list(Path(sess).rglob('data.bdf'))[0]
                    evt_bdf_file = list(Path(sess).rglob('evt.bdf'))[0]
                    result_agzz = read_bdf(data_bdf_file.parent)
                    # print(result['events'])
                    events = result_agzz['events']
                    for i, event in enumerate(events):
                        try:
                            if events[i][2] == 8 and events[i + 1][2] == 16 and events[i + 2][2] == 32:
                                print(
                                    f'--------------fixation time {(events[i+1][0]-events[i][0])/result_agzz['srate']}s, {((events[i+1][0]-events[i][0])/result_agzz['srate'])//60}minuts, {((events[i+1][0]-events[i][0])/result_agzz['srate'])%60: .4f}seconds'
                                )
                                print(
                                    f'--------------movie time {(events[i+2][0]-events[i+1][0])/result_agzz['srate']}s, {((events[i+2][0]-events[i+1][0])/result_agzz['srate'])//60}minuts, {((events[i+2][0]-events[i+1][0])/result_agzz['srate'])%60: .4f}seconds'
                                )
                                fixation_start = events[i][0]
                                start_play = events[i + 1][0]
                                rating_page_1 = events[i + 2][0]
                                # EEG 数据
                                eeg_segment = result_agzz['data'][:, start_play:rating_page_1]
                                fixation_segment = result_agzz['data'][
                                    :, int(fixation_start) : int(fixation_start) + int(2 * 60 * result_agzz['srate'])
                                ]
                                # 往eeg_segment['data']这个listappend数据，如果不存在则创建一个新的list
                                if 'data' not in agzz_data:
                                    agzz_data['data'] = []
                                agzz_data['data'].append(eeg_segment)
                                if 'fixation' not in agzz_data:
                                    agzz_data['fixation'] = []
                                agzz_data['fixation'].append(fixation_segment)
                                pass
                        except IndexError:
                            pass
                if 'rzdf' in str(sess):
                    data_bdf_file = list(Path(sess).rglob('data.bdf'))[0]
                    evt_bdf_file = list(Path(sess).rglob('evt.bdf'))[0]
                    result_rzdf = read_bdf(data_bdf_file.parent)
                    events = result_rzdf['events']
                    events = np.delete(events, [8, 9, 10, 37, 38, 39, 50, 51, 52, 63, 64, 65], axis=0)
                    for i, event in enumerate(events):
                        try:
                            if events[i][2] == 8 and events[i + 1][2] == 16 and events[i + 2][2] == 32:
                                print(
                                    f'--------------fixation time {(events[i+1][0]-events[i][0])/result_rzdf['srate']}s, {((events[i+1][0]-events[i][0])/result_rzdf['srate'])//60}minuts, {((events[i+1][0]-events[i][0])/result_rzdf['srate'])%60: .4f}seconds'
                                )
                                print(
                                    f'--------------movie time {(events[i+2][0]-events[i+1][0])/result_rzdf['srate']}s, {((events[i+2][0]-events[i+1][0])/result_rzdf['srate'])//60}minuts, {((events[i+2][0]-events[i+1][0])/result_rzdf['srate'])%60: .4f}seconds'
                                )
                                fixation_start = events[i][0]
                                start_play = events[i + 1][0]
                                rating_page_1 = events[i + 2][0]
                                # EEG 数据
                                eeg_segment = result_rzdf['data'][:, start_play:rating_page_1]
                                fixation_segment = result_rzdf['data'][
                                    :, int(fixation_start) : int(fixation_start) + int(2 * 60 * result_rzdf['srate'])
                                ]
                                if 'data' not in rzdf_data:
                                    rzdf_data['data'] = []
                                rzdf_data['data'].append(eeg_segment)
                                if 'fixation' not in rzdf_data:
                                    rzdf_data['fixation'] = []
                                rzdf_data['fixation'].append(fixation_segment)
                        except IndexError:
                            pass

            agzz_data['srate'] = result_agzz['srate']
            agzz_data['ch_names'] = result_agzz['ch_names']
            agzz_data['nchan'] = result_agzz['nchan']

            rzdf_data['srate'] = result_rzdf['srate']
            rzdf_data['ch_names'] = result_rzdf['ch_names']
            rzdf_data['nchan'] = result_rzdf['nchan']

            # save data
            sub_name = sess.parent.parent.name
            save_path = Path(OUTPUT_DIR) / sub_name
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / 'agzz.pkl', 'wb') as f:
                pickle.dump(agzz_data, f)
            with open(save_path / 'rzdf.pkl', 'wb') as f:
                pickle.dump(rzdf_data, f)


if __name__ == '__main__':
    mian()
