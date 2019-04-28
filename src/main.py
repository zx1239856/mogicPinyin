if(__name__=="__main__"):
    import argparse
    import ime as m_ime
    import os.path as path
    import config
    parser = argparse.ArgumentParser(description="Intellipinyin IME powered by zx1239856")
    parser.add_argument('--shell', help = 'Enable shell interaction', dest='shell', action='store_true')
    parser.set_defaults(shell = True)
    parser.add_argument('-n', '--n-grams', help='Select to use 2-gram or 3-gram model', default = 3)
    parser.add_argument('--file', help='Enable file io', dest = 'fileio', action='store_true')
    parser.set_defaults(fileio = False)
    parser.add_argument('-i', '--input-file', help='In-file name')
    parser.add_argument('-o', '--output-file', help='Out-file name')
    parser.add_argument('--lm', help='Param for 2-gram', default = 1e-3)
    parser.add_argument('--mu', help='Param for 3-gram', default = 1.03)
    args = parser.parse_args()
    ime = None
    best_2_gram = float(args.lm)
    best_3_gram = float(args.mu)
    if(int(args.n_grams) == 2):
        print("Using 2-gram model")
        ime = m_ime.Ime(path.join(config.TRAINED_DIR, config.PINYIN), path.join(config.TRAINED_DIR, config.CHARACTER_LIST),
        path.join(config.TRAINED_DIR, config.FREQ_GRAM1), path.join(config.TRAINED_DIR, config.FREQ_GRAM2), None, best_2_gram, best_3_gram)
    elif(int(args.n_grams) == 3):
        print("Using 3-gram model, this will be slower")
        ime = m_ime.Ime(path.join(config.TRAINED_DIR, config.PINYIN), path.join(config.TRAINED_DIR, config.CHARACTER_LIST),
        path.join(config.TRAINED_DIR, config.FREQ_GRAM1), path.join(config.TRAINED_DIR, config.FREQ_GRAM2),
        path.join(config.TRAINED_DIR, config.FREQ_GRAM3), best_2_gram, best_3_gram)
    else:
        print("Invalid n_grams. Expected 2 or 3")
        exit(-1)
    if(args.fileio):
        ime.fileio(args.input_file, args.output_file, False)
    elif(args.shell):
        ime.shell()
