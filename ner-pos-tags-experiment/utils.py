def get_pos_tag_id_map():
    pos_tags = [
        'DET', 'ADJ', 'NOUN', 'ADP', 'NUM', 'VERB', 'PUNCT', 'ADV',
        'PART', 'CCONJ', 'PRON', 'X', 'INTJ', 'PROPN', 'SYM', 'AUX',
        '<UNK>', 'SCONJ', 'SPACE'
    ]
    pos_tags.sort()
    pos_tag_to_id_map = {x: i for i, x in enumerate(pos_tags)}
    pos_id_to_tag_map = {i: x for i, x in enumerate(pos_tags)}
    return (pos_tag_to_id_map, pos_id_to_tag_map)


def get_label_map():
    label_list = ['O', 'B-SUBJ']
    num_labels = len(label_list)
    label_to_id = {i: i for i in range(len(label_list))}
    return (label_list, num_labels, label_to_id)
