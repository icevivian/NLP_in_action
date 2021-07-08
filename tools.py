# data preprocessor
def pre_process_data(filepath):
    """
    load pos and neg examples from separate dirs then shuffle them together
    """
    positive_path = os.path.join(filepath, 'pos')
    negative_path = os.path.join(filepath, 'pos')
    pos_label = 1
    neg_label = 0
    dataset = []
    for filename in glob.glob(os.path.join(positive_path, '*.txt')): # 遍历pos文件夹下的所有txt文件
        with open(filename, 'r') as f:
            dataset.append((pos_label, f.read()))
    for filename in glob.glob(os.path.join(negative_path, '*.txt')):
        with open(filename, 'r') as f:
            dataset.append((neg_label, f.read()))
    shuffle(dataset)
    return dataset

# data tokenizer + vectorizer
def tokenize_and_vectorizer(dataset):
    tokenizer = TreebankWordTokenizer()
    vectorized_data = []
    for sample in dataset:
        tokens = tokenizer.tokenize(sample[1])
        sample_vecs = []
        for token in tokens:
            try:
                sample_vecs.append(word_vectors[token])
            except KeyError:
                pass
        vectorized_data.append(sample_vecs)
    return vectorized_data

# target unzipper
def collected_expected(dataset):
    """
    peel off the target values from the dataset
    """
    expected = []
    for sample in dataset:
        expected.append(sample[0])
    return expected

def pad_trunc(data, maxlen):
    """
    for a given dataset pad with zero vectors or truncate to maxlen
    """
    new_data = []
    zero_vector = []
    for _ in range(len(data[0][0])):
        zero_vector.append(0.0)
    for sample in data:
        if len(sample) > maxlen:
            temp = sample[:maxlen]
        elif len(sample) < maxlen:
            temp = sample
            additional_elems = maxlen - len(sample)
            for _ in range(additional_elems):
                temp.append(zero_vector)
        else:
            temp = sample
        new_data.append(temp)
    return new_data