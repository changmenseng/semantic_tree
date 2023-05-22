from nltk.tree import Tree

def get_constituents(seq_ptb, labeled=False):
    tree = Tree.fromstring(seq_ptb)
    tree.collapse_unary(True, True)
    for i, pos in enumerate(tree.treepositions('leaves')):
        tree[pos] += f'_{i}'
    constituents = set()
    for subtree in tree.subtrees():
        leaves = subtree.leaves()
        start_id = int(leaves[0].split('_')[-1])
        end_id = int(leaves[-1].split('_')[-1])
        if start_id != end_id:
            constituent = [start_id, end_id]
            if labeled:
                constituent.append(subtree.label())
            constituents.add(tuple(constituent))
    return constituents

def get_f1(pred_constituents, gold_constituents):
    ncorrects = len(pred_constituents & gold_constituents)
    prec = ncorrects / (len(pred_constituents) + 1e-8)
    recall = ncorrects / (len(gold_constituents) + 1e-8)
    f1 = 2 * prec * recall / (prec + recall + 1e-8)
    return prec, recall, f1


def eval_single_tree(pred_ptb, gold_ptb, labeled=True):
    pred_constituents = get_constituents(pred_ptb, labeled)
    gold_constituents = get_constituents(gold_ptb, labeled)
    return get_f1(pred_constituents, gold_constituents)