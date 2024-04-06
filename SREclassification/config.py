config ={
    'model_name': 'cMatchBoxNet-3x2x112',
    'model_dir': '/data/saten/diana/SpeakingRateEstimation/SREclassification/Models',
    'noise_dir': '/data/saten/diana/SpeakingRateEstimation/data/ESC-50_16khz/audio',
    'train_dir': '/data/saten/diana/SpeakingRateEstimation/data/LibriSpeechChuncked_sil_removed/train-clean-100-fast-augmented',
    'val_dir': '/data/saten/diana/SpeakingRateEstimation/data/LibriSpeechChuncked_sil_removed/dev-clean',
    'test_dir': '/data/saten/diana/SpeakingRateEstimation/data/LibriSpeechChuncked_sil_removed/test-clean',
    'sample_rate': 16000,
    'input_len': 35098, # max input length
    'frame_length': 35,
    'window_shift': 10,
    'train_params': {
        'batch_size': 200,
        'max_epochs':200,
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