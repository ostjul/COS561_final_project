import tree
import jax.numpy as jnp
import os
import shutil

def tree_stack(list_):
    assert len(list_) > 0
    leaves_list = []
    for struct in list_:
        leaves = tree.flatten(struct)
        leaves_list.append(leaves)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return tree.unflatten_as(list_[0], result_leaves)

def tree_index(struct, idx):
  return tree.map_structure(lambda x: x[idx], struct)

# This is the inverse operation of tree_stack.
def tree_unstack(struct):
    n, = set([len(x) for x in tree.flatten(struct)])
    return [tree_index(struct, i) for i in range(n)]

def remove_and_create_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)