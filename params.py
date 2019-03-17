seed = 0
n_jobs = 16

file_path_model = 'model.bin'
file_path_train = 'data/geo880_train600.tsv'
file_path_dev = 'data/geo880_test280.tsv'

# test end-to-end
experiment1_test = {
    'augment': {'nesting+entity+concat2'},
    'pre_train': {True},
    'lr': {0.001},
    'decoder_type': {'rnn'},
    'domain_name': {'geoquery'},
    'aug_frac': {1.0},
    'embed_size': {128},
    'hidden_size': {128},
    'seed': {seed},
    'dropout': {0.3},
    'cuda': {False},
    'batch_size_train': {256},
    'batch_size_dev': {128},
    'valid_niter': {100},
    'max_epoch': {1000},
    'beam_size': {5},
    'max_sentence_length': {1000},
    'encoder_type': {'brnn'},
    'file_path_train': {file_path_train},
    'file_path_dev': {file_path_dev},
    'file_path_model': {file_path_model}
}

# actual grid
experiment1 = {
    'augment':  {None, 'nesting+entity+concat2', 'co', 'nesting+entity+concat2+co'},
    'pre_train': {True, False},
    'lr': {0.01, 0.001},
    'decoder_type': {'rnn', 'transformer'},
    'domain_name': {'geoquery'},
    'aug_frac': {1.0},
    'embed_size': {64, 128},
    'hidden_size': {64, 128, 256},
    'seed': {seed},
    'dropout': {0.3},
    'cuda': {False},
    'batch_size_train': {256},
    'batch_size_dev': {128},
    'valid_niter': {100},
    'max_epoch': {1000},
    'beam_size': {5},
    'max_sentence_length': {1000},
    'encoder_type': {'brnn'},
    'file_path_train': {file_path_train},
    'file_path_dev': {file_path_dev},
    'file_path_model': {file_path_model}
}
