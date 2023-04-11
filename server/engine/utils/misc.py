import torch
import matplotlib.pyplot as plt

def visualize_matrix(weight, limit_dim = 100):
    weight = torch.stack([w[:limit_dim] for w in weight[:limit_dim]]).cpu()
    limit = max(abs(weight.min().item()), abs(weight.max().item()))
    img = plt.imshow(
        weight,
        cmap='RdBu', interpolation='nearest', 
        vmin = -limit, vmax = limit
    )
    plt.colorbar(img, orientation='vertical')
    plt.show()


def low_rank_approximation(weight, rank = 10):
    typecache = weight.dtype
    weight = weight.to(torch.float32)
    svd = weight.svd()
    wgt_est = torch.zeros(weight.shape).to(weight.device)
    for i in range(rank):
        wgt_est += svd.S[i] * (svd.U[:, i][None].T @ svd.V[:, i][None])
    # print(f"approximation error ==> {torch.dist(weight, wgt_est)}")
    approx_err = torch.dist(weight, wgt_est)
    print(f"rank {rank} >> ", approx_err)
    weight = weight.to(typecache)
    return wgt_est.to(typecache)


import copy

child_last   = "└───"
child_middle = "├───"
space_pre    = "    "
middle_pre   = "│   "
def check_structure_tree(obj, key='#', level=0, level_info = {}, max_depth = 2):
    if(level == max_depth+1):
        return

    if(level > 0):
        for i in range(level-1):
            if(level_info[i] == 'last'):
                print(space_pre, end="")
            else:
                print(middle_pre, end="")
        if(level_info[level-1] == 'last'):
            child_pre = child_last
        else:
            child_pre = child_middle
        print(child_pre, end="")
    
    if(key != '#'):
        print(key, end=": ")
    
    num_elem = ""
    if(isinstance(obj, tuple) or isinstance(obj, list)):
        num_elem = f'[{len(obj)}]'
    print(type(obj), num_elem, end=" ")
    if(type(obj) is torch.Tensor):
        print("[{}] {}".format(obj.shape, obj.device))
    else:
        print()
    if(isinstance(obj, tuple) or isinstance(obj, list) or isinstance(obj, dict)):
        if(isinstance(obj, dict)):
            keys = list(obj.keys())
        else:
            keys = list(range(len(obj)))
        
        for idx in range(len(keys)):
            li = copy.deepcopy(level_info)
            if(idx == len(obj)-1):
                li[level] = 'last'
            else:
                li[level] = 'middle'
            check_structure_tree(obj[keys[idx]], key=keys[idx], level = level + 1, level_info = li, max_depth = max_depth)