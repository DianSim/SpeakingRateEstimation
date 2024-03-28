config ={
    'model_name': 'rLSTM128_augm',
    'model_dir': '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/SREregression/models',
    'noise_dir': '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/data/ESC-50_16khz/audio',
    'train_dir': '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/data/LibriSpeechChuncked_sil_removed/train-clean-100-fast-augmented',
    'val_dir': '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/data/LibriSpeechChuncked_sil_removed/dev-clean',
    'test_dir': '/home/dianasimonyan/Desktop/Thesis/SpeakingRateEstimation/data/LibriSpeechChuncked_sil_removed/test-clean',
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
