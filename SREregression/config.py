config ={
    'model_name': 'rMatchBoxNet-3x2x112',
    'model_dir': '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/SREregression/models_4sec',
    'noise_dir': '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/data/ESC-50_16khz/audio',
    'train_dir': '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/data/LibriSpeechChuncked_sil_removed/split_4sec/train-clean-100-fast-augmented',
    'val_dir': '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/data/LibriSpeechChuncked_sil_removed/split_4sec/dev-clean-fast-augmented',
    'test_dir': '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/data/LibriSpeechChuncked_sil_removed/split_4sec/test-clean',
    'model_error_dir': '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/SREregression/abs_error_histogram_m4sec',
    'sample_rate': 16000,
    'input_len': 66032, # 35098, # 66032, # change model architectires upper bound  to the train max label
    'frame_length': 35,
    'window_shift': 10,
    'train_params': {
        'batch_size': 200,
        'max_epochs':300,
        'steps_per_epoch': None,
        # 'latest_checkpoint_step': 1,
        # 'summary_step': 50,
        'max_checkpoints_to_keep': 8,
    },
    'model_params':{
        'f_n': 8,
        'f_l': 30 # ms
    },
    'feature': {
        'window_size_ms': 25,
        'window_stride': 15,
        'fft_length': 512,
        'mfcc_lower_edge_hertz': 0.0,
        'mfcc_upper_edge_hertz': 8000.0,
        'mfcc_num_mel_bins': 64
    }
}