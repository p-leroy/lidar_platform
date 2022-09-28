import json

def load_id_name_dictionary(file):
    with open(file) as fp:
        dict_ = json.load(fp)
    return {int(value): key for value, key in dict_.items()}


def get_name_id_dict(lines):
    # produce two dictionaries stored in two files
    # id_name.json with point source id as key and line name as value
    # name__id_list.json with line name as key and point source id as value
    head, tail = os.path.split(lines[0])
    name__id_list = {}
    for line in lines:
        head, tail = os.path.split(line)
        print(f'Read line: {line}')
        f = laspy.read(line)
        list_np_fmt = list(np.unique(f.point_source_id))
        name__id_list[tail] = [int(val) for val in list_np_fmt]
    # check that all values are unique (we should have only one point_source_id per line)
    multi_id_lines = [name for name, id_list in name__id_list.items() if len(id_list) != 1]
    if multi_id_lines:
        for line in multi_id_lines:
            print(f'name__id_list {name__id_list[line]}')
            print(f'multi point_source_id in {line}')
        raise ValueError("some lines contain multi point_source_id")
    # reverse dictionary: key = point_source_id, value = name
    id_name = {int(id_list[0]): name for name, id_list in name__id_list.items()}

    with open(os.path.join(head, 'id_name.json'), 'w') as f:
        json.dump(id_name, f)
    with open(os.path.join(head, 'name__id_list.json'), 'w') as f:
        json.dump(name__id_list, f)

    return name__id_list, id_name
