config ={
    'model_name': 'LSTM128_dense25_noise_augm',
    'sample_rate': 16000,
    'input_len': 32325, # max input length
    'frame_length': 35,
    'window_shift': 10,
    'train_params': {
        'batch_size': 8,
        'epochs':200,
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