from validation.organ.organ_tree import load_taxonomy, shortest_path


def hierarchical_score(input_term, ground_truth_term, lookup, graph):
    """
    Computes the hierarchical score between two taxonomy terms using the lookup and graph.

    Scoring:
      - 1.0 if the input and ground truth resolve to the same node.
      - 0.75 if the nodes are one relation apart (direct neighbor: parent-child or sibling).
      - 0.5 if two relations apart.
      - 0.0 otherwise.
    """
    input_term = input_term.lower()  # Convert input term to all lowercase
    ground_truth_term = ground_truth_term.lower()

    if input_term not in lookup:
        print(f"Input term '{input_term}' not found in taxonomy.")
        return 0.0
    if ground_truth_term not in lookup:
        print(f"Ground truth term '{ground_truth_term}' not found in taxonomy.")
        return 0.0

    input_node = lookup[input_term]
    ground_truth_node = lookup[ground_truth_term]

    if input_node == ground_truth_node:
        return 1.0

    dist = shortest_path(graph, ground_truth_node, input_node)

    if dist == 1:
        return 0.75
    elif dist == 2:
        return 0.5
    else:
        return 0.0


def compute_organ_score(input_term, ground_truth_term, taxonomy_file):
    """
    Computes the hierarchical score for the input term against one or more ground truth terms.

    Args:
        input_term (str): Term provided by the pathologist/model.
        ground_truth_term (str): Comma-separated ground truth terms.
        taxonomy_file (str): Path to the YAML taxonomy file.

    Returns:
        float: The highest hierarchical score among all ground truth options.
    """
    _, lookup, graph = load_taxonomy(taxonomy_file)

    # Split and normalize multiple ground truth terms
    gt_terms = [gt.strip().lower() for gt in ground_truth_term.split(",")]
    input_term = input_term.strip().lower()

    # Compute hierarchical scores for each ground truth term
    scores = [hierarchical_score(input_term, gt, lookup, graph) for gt in gt_terms]

    return max(scores) if scores else 0.0
