# DailyMovie
from pathlib import Path
import mne
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm


root_path = Path(__file__).resolve().parent
eeg_base_dir = Path('/home/liaoyizhi/codes/ncclab_ddh/eeg_data')

window_size = 2.0
stride_size = 2.0  # FIXME. stride_size没用上
label_path = Path('/home/liaoyizhi/codes/ncclab_ddh/results/smooth_scores')

h5_dir_root = Path(f'/mnt/dataset2/Processed_datasets/EEG_Bench')
h5_dir_path = h5_dir_root / f"Daily_Movie_hdf5_T={window_size}s_stride={stride_size}"
h5_dir_path.mkdir(parents=True, exist_ok=True)


def show(name, obj):
    if isinstance(obj, h5py.Group):
        print(f"[G] {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"[D] {name}  shape={obj.shape} dtype={obj.dtype}")
        if len(obj.attrs) > 0:
            print("    attrs:", list(obj.attrs.keys()))


sub_list = ['sub4_qianyi', 'sub5_xulong', 'sub9_tongyu', 'sub10_yifan', 'sub11_yingyue', 'sub13_penghe']
subj_anonymous_li = {
    'sub4_qianyi': 'sub01',
    'sub5_xulong': 'sub02',
    'sub9_tongyu': 'sub03',
    'sub10_yifan': 'sub04',
    'sub11_yingyue': 'sub05',
    'sub13_penghe': 'sub06',
}

episode_dict = {
    'agzz': [
        'final_segment_0',
        'final_segment_1',
        'final_segment_2',
        'final_segment_3',
        'final_segment_4',
        'final_segment_5',
        'final_segment_6',
        'final_segment_7',
        'final_segment_8',
    ],
    'lldq2': [
        'final_segment_0',
        'final_segment_1',
        'final_segment_2',
        'final_segment_3',
        'final_segment_4',
        'final_segment_5',
        'final_segment_6',
        'final_segment_7',
        'final_segment_8',
        'final_segment_9',
        'final_segment_10',
    ],
    'rzdf': [
        'final_segment_0',
        'final_segment_1',
        'final_segment_2',
        'final_segment_3',
        'final_segment_4',
        'final_segment_5',
        'final_segment_6',
        'final_segment_7',
        'final_segment_8',
    ],
    'sdd': [
        'final_segment_0',
        'final_segment_1',
        'final_segment_2',
        'final_segment_3',
        'final_segment_4',
        'final_segment_5',
        'final_segment_6',
        'final_segment_7',
        'final_segment_8',
    ],
    'xltfn': [
        'final_segment_0',
        'final_segment_1',
        'final_segment_2',
        'final_segment_3',
        'final_segment_4',
        'final_segment_5',
    ],
    'ymlw': [
        'final_segment_0',
        'final_segment_1',
        'final_segment_2',
        'final_segment_3',
        'final_segment_4',
        'final_segment_5',
        'final_segment_6',
        'final_segment_7',
    ],
    'zgjqxs': ['final_segment_0', 'final_segment_1', 'final_segment_2', 'final_segment_3', 'final_segment_4'],
    'zccsh': [
        'final_segment_0',
        'final_segment_1',
        'final_segment_2',
        'final_segment_3',
        'final_segment_4',
        'final_segment_5',
        'final_segment_6',
        'final_segment_7',
        'final_segment_8',
    ],
}

for sub_idx, sub in enumerate(sub_list):
    with h5py.File(str(h5_dir_path / f"sub_{sub_idx}.h5"), 'w') as h5f_root:
        h5f_root.attrs['subject_id'] = sub_idx
        h5f_root.attrs['dataset_name'] = 'Daily_Movie'
        h5f_root.attrs['task_type'] = 'None'
        h5f_root.attrs['downstream_task_type'] = 'None'
        root_attrs_set = False

        movie_name_li = [_.stem for _ in sorted(list((eeg_base_dir / sub).glob("*.pkl")))]
        video_li = [f'{movie_name}_{episode}' for movie_name in movie_name_li for episode in episode_dict[movie_name]]

        # iterate episodes/trials
        for trial_idx, trial_name in enumerate(tqdm(video_li, desc=f"{sub} trials")):

            # create trial group
            trial_group = h5f_root.create_group(f'trial{trial_idx}')
            trial_group.attrs['trial_id'] = trial_idx
            trial_group.attrs['session_id'] = 'None'

            movie_name = trial_name.split('_final_segment_')[0]
            episode_idx = int(trial_name.split('_final_segment_')[-1])
            eeg_path = eeg_base_dir / sub / f"{movie_name}.pkl"

            # load pkl
            with open(eeg_path, 'rb') as f:
                import pickle

                eeg_data = pickle.load(f)
                sfreq = eeg_data['srate']
                channel_li = eeg_data['ch_names'][:59]
                max_time = int((eeg_data['data'][episode_idx].shape[-1] / float(sfreq)) // 10) * 10
                data_ = eeg_data['data'][episode_idx][:59, : int(max_time * sfreq)]

            assert data_ is not None
            if not root_attrs_set:
                h5f_root.attrs['rsFreq'] = float(sfreq)
                h5f_root.attrs['chn_name'] = channel_li
                h5f_root.attrs['chn_pos'] = 'None'
                h5f_root.attrs['chn_ori'] = 'None'
                h5f_root.attrs['chn_type'] = 'EEG'
                h5f_root.attrs['montage'] = '10_20'
                root_attrs_set = True

            segment_count = data_.shape[-1] // sfreq // window_size
            # load markers
            assert len(list(label_path.glob(f'{trial_name}_smooth.csv'))) == 1
            markers_path = list(label_path.glob(f'{trial_name}_smooth.csv'))[0]
            labels_df = pd.read_csv(markers_path, header=0)
            markers_np = labels_df.to_numpy()[:, 1:]  # (n_times, n_label_dims)

            if markers_np.shape[0] != int(segment_count):
                pass
                if markers_np.shape[0] > int(segment_count):
                    markers_np = markers_np[: int(segment_count), :]
                else:
                    segment_count = markers_np.shape[0]
            assert markers_np.shape[0] == int(segment_count)  # check
            # iterate segments
            for segment_idx in range(int(segment_count)):
                start_sample = int(segment_idx * window_size * sfreq)
                stop_sample = int(start_sample + window_size * sfreq)
                data_segment = data_[:, start_sample:stop_sample]
                label = markers_np[segment_idx]

                # create segment group
                segment_group = trial_group.create_group(f'segment{segment_idx}')
                dataset = segment_group.create_dataset('eeg', data=data_segment, compression="gzip")
                dataset.attrs['segment_id'] = segment_idx
                dataset.attrs['start_time'] = float(start_sample) / float(sfreq)
                dataset.attrs['end_time'] = float(stop_sample) / float(sfreq)
                dataset.attrs['time_length'] = float(window_size)
                dataset.attrs['label'] = label

            # pass  # segments end
            # h5f_root.visititems(show)
            # data = h5f_root['trial0/segment0/eeg'][:]
            pass  # segments end

        pass  # subject end
    pass  # subject end
pass  # all subj end
