CROSS_VALIDATION_SPLITS = {


    "Win5LID": {
        1: ['Sphynx', 'Bikes'],
        2: ['rosemary', 'Flowers'],
        3: ['Vespa', 'Palais_du_Luxembourg'],
        4: ['museum','Swans' ],
        5: ['dishes', 'greek'],
    },

    "NBU": {
        1: ['Danger de Mort', 'Rosemary', 'Vespa'],
        2: ['Dishes', 'Mirabelle Prune Tree', 'Rolex Learning Center'],
        3: ['Bikes', 'Museum', 'Sphynx'],
        4: ['Boardgames', 'Kitchen', 'Swans 1'],
        5: ['Greek', 'Railway Lines 1'],
    },

    "SHU": {
        1: ['ISO_Chart_12', 'Stone_Pillars_Outside'],
        2: ['Color_Chart_1', 'Friends_1'],
        3: ['Fountain_&_Vincent_2'],
        4: ['Flowers'],
        5: ['Danger_de_Mort', 'bike'],
    }


}


def get_split_for_fold(dataset_key: str, all_scenes: list, fold_num: int) -> tuple[list, list]:

    if dataset_key not in CROSS_VALIDATION_SPLITS:
        raise ValueError(f"No cross-validation splits defined for dataset: {dataset_key}")
    if fold_num not in CROSS_VALIDATION_SPLITS[dataset_key]:
        raise ValueError(f"Fold {fold_num} not defined for dataset: {dataset_key}")

    test_scenes = CROSS_VALIDATION_SPLITS[dataset_key][fold_num]

    test_scenes_set = set(test_scenes)
    all_scenes_set = set(all_scenes)
    if not test_scenes_set.issubset(all_scenes_set):
        missing = test_scenes_set - all_scenes_set
        raise ValueError(f"Scenes {missing} from test split are not in the full scene list of {dataset_key}.")

    train_scenes = list(all_scenes_set - test_scenes_set)

    return sorted(train_scenes), sorted(test_scenes)