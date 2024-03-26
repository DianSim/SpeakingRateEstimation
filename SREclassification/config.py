config ={
    'model_name': 'cMatchBoxNet6x2x64',
    'model_dir': '/data/saten/diana/SpeakingRateEstimation/SREclassification/models',
    'noise_dir': '/data/saten/diana/SpeakingRateEstimation/data/ESC-50-master/audio',
    'train_dir': '/data/saten/diana/SpeakingRateEstimation/data/LibriSpeechChuncked_v2/train-clean-100',
    'val_dir': '/data/saten/diana/SpeakingRateEstimation/data/LibriSpeechChuncked_v2/dev-clean',
    'test_dir': '/data/saten/diana/SpeakingRateEstimation/data/LibriSpeechChuncked_v2/test-clean',
    'sample_rate': 16000,
    'input_len': 33601, # max input length
    'frame_length': 35,
    'window_shift': 10,
    'train_params': {
        'batch_size': 200,
        'max_epochs':200,
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
