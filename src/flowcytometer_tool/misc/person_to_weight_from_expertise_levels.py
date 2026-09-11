from flowcytometer_tool.misc.normalise_training_person_name import _normalise_training_person_name


def _person_to_weight_from_expertise_levels(expertise_levels):
    expertise_weights = {"expert": 3, "advanced": 2, "non_expert": 1}
    person_to_weight = {}
    for level, people in (expertise_levels or {}).items():
        for person in people:
            person_to_weight[_normalise_training_person_name(person)] = expertise_weights.get(level, 1)
    return person_to_weight

