seed = 0
n_jobs = 8

file_path_train = {'geoquery': 'data/geo880_train600.tsv'}
file_path_dev = {'geoquery': 'data/geo880_test280.tsv'}
file_path_model = 'model.bin'

experiment1 = {
        'augment': {'none', 'nesting+entity+concat2', 'co', 'nesting+entity+concat2+co'},
        'pre_train': {True, False},
        'lr': {0.1, 0.01, 0.001},
        'decoder_type': {'rnn', 'transformer'},
        'domain_name': {'geoquery'},  # {'overnight-socialnetwork'},
        'aug_frac': {0.5, 1.0, 2.0},
        'embed_size': {128},  # {64, 128, 256},
        'hidden_size': {128},  # {64, 128, 256}
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
}
