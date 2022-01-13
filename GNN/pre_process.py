import math
import os
import random
import time

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn.decomposition import PCA

from dataset import load_dataset
from utils import (calculate_homophily, clear_memory,
                   entropy, inner_distance, outer_distance)


def neighbor_average_features_by_chunks(g, feat, args, style="all", stats=False, memory_efficient=False, target_nid=None):
    """
    Compute multi-hop neighbor-averaged node features by chunks
    """
    if args.chunks == 1:
        return neighbor_average_features(g, feat, args, style=style, stats=stats, memory_efficient=memory_efficient, target_nid=target_nid)
    aggr_device = torch.device("cpu" if args.aggr_gpu < 0 else "cuda:{}".format(args.aggr_gpu))
    feat_size = feat.shape[1]
    chunk_size = int(math.ceil(feat_size / args.chunks))
    
    print("Saving temporary initial feature chunks……")
    tmp_dir = os.path.join(args.data_dir, "_".join(args.dataset.split("-")), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    for part, i in enumerate(tqdm.tqdm(range(0, feat_size, chunk_size))):
        chunk_path = os.path.join(tmp_dir, f"feat_{part}.npy")
        if os.path.exists(chunk_path):
            continue
        chunk = feat[:, i: min(i+chunk_size, feat_size)]
        np.save(chunk_path, chunk.cpu().numpy())

    del feat
    clear_memory(aggr_device)
    print("Perform feature propagation by chunks……")
    for part, i in enumerate(tqdm.tqdm(range(0, feat_size, chunk_size))):
        chunk = torch.from_numpy(np.load(os.path.join(tmp_dir, f"feat_{i}.npy"))).to(aggr_device)
        out_chunk = neighbor_average_features(g, chunk, args, style=style, stats=stats, memory_efficient=memory_efficient)
        if style=="all":
            chunk = [c.cpu().numpy() for c in chunk]
        else:
            chunk = chunk.cpu().numpy()
        np.save(os.path.join(tmp_dir, f"smoothed_feat_{part}.npy"), out_chunk)
    del chunk
    clear_memory(aggr_device)
    print("Loading aggregated chunks……")
    if style=="all":
        out_feat = [torch.empty_like(feat, device=aggr_device) for k in range(args.K+1)]  
    else:
        out_feat = torch.empty_like(feat, device=aggr_device)
    for part, i in enumerate(tqdm.tqdm(range(0, feat_size, chunk_size))):
        out_chunk = torch.from_numpy(np.load(os.path.join(tmp_dir, f"smoothed_feat_{part}.npy"))).to(aggr_device)
        if style == "all":
            for k in range(args.K+1):
                out_feat[k][:, i: min(i+chunk_size, feat_size)] = out_chunk[k]
        else:
            out_feat[:, i: min(i+chunk_size, feat_size)] = out_chunk
    print("Removing temporary files……")
    for part, i in enumerate(tqdm.tqdm(range(0, feat_size, chunk_size))):
        # os.remove(os.path.join(tmp_dir, f"feat_{i}.pt"))
        os.remove(os.path.join(tmp_dir, f"smoothed_feat_{part}.pt"))
    del out_chunk
    clear_memory(aggr_device)
    return out_feat


def neighbor_average_features(g, feat, args, style="all", stats=True, memory_efficient=False, target_nid=None):
    """
    Compute multi-hop neighbor-averaged node features
    """
    print("Compute neighbor-averaged feats", style)
    
    aggr_device = torch.device("cpu" if args.aggr_gpu < 0 else "cuda:{}".format(args.aggr_gpu))
    g = g.to(aggr_device)
    feat = feat.to(aggr_device)
    tmp_dir = os.path.join(args.data_dir, "_".join(args.dataset.split("-")), "tmp")
    idx = target_nid if target_nid is not None else torch.arange(len(feat)).to(aggr_device)
    os.makedirs(tmp_dir, exist_ok=True)
    print(tmp_dir)
    if style == "all":
        if memory_efficient:
            torch.save(feat[idx].clone(), os.path.join(tmp_dir, '0.pt'))
            res = []
        else:
            res = [feat[idx].clone()]
        
            
        # print(g.ndata["feat"].shape)
        # print(norm.shape)
        if args.use_norm:
            degs = g.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp)
        for hop in range(1, args.K + 1):
            g.ndata['f'] = feat
            if args.use_norm:
                g.ndata['f'] = g.ndata['f'] * norm
                g.update_all(fn.copy_src(src=f'f', out='msg'),
                            fn.sum(msg='msg', out=f'f'))
                g.ndata['f'] = g.ndata['f'] * norm
            else:
                g.update_all(fn.copy_src(src='f', out='msg'),
                            fn.mean(msg='msg', out='f'))
            feat = g.ndata.pop("f")
            if memory_efficient:
                torch.save(feat[idx].clone(), os.path.join(tmp_dir, f'{hop}.pt'))
            else:
                res.append(feat[idx].clone())
        
        del feat
        clear_memory(aggr_device)
        if memory_efficient:
            for hop in range(args.K+1):
                res.append(torch.load(os.path.join(tmp_dir, f'{hop}.pt')))
                os.remove(os.path.join(tmp_dir, f'{hop}.pt'))

        clear_memory(aggr_device)

    # del g.ndata['pre_label_emb']
    elif style in ["last", "ppnp"]:
        if stats:
            feat_0 = feat.clone()
            train_mask = g.ndata["train_mask"]
            print(f"hop 0: outer distance {outer_distance(feat_0, feat_0, train_mask):.4f}, inner distance {inner_distance(feat_0, train_mask):.4f}")
        if style == "ppnp": init_feat = feat
        if args.use_norm:
            degs = g.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp)
        for hop in range(1, args.label_K+1):         
            if args.use_norm:
                feat = feat * norm
                g.ndata['f'] = feat
                g.update_all(fn.copy_src(src='f', out='msg'),
                            fn.sum(msg='msg', out='f'))
                feat = g.ndata.pop('f')
                feat = feat * norm
            else:
                g.ndata['f'] = feat
                g.update_all(fn.copy_src(src='f', out='msg'),
                            fn.mean(msg='msg', out='f'))
                feat = g.ndata.pop('f')
            if style == "ppnp":
                feat = 0.5 * feat + 0.5 * init_feat
            if stats:
                print(f"hop {hop}: outer distance {outer_distance(feat_0, feat, train_mask):.4f}, inner distance {inner_distance(feat, train_mask):.4f}")
            
        res = feat[idx].clone()
        del feat
        clear_memory(aggr_device)

        if args.dataset == "ogbn-mag":
            # For MAG dataset, only return features for target node types (i.e.
            # paper nodes)
            target_mask = g.ndata['target_mask']
            target_ids = g.ndata[dgl.NID][target_mask]
            num_target = target_mask.sum().item()
            new_res = torch.zeros((num_target,) + feat.shape[1:],
                                    dtype=feat.dtype, device=feat.device)
            new_res[target_ids] = res[target_mask]
            res = new_res

    return res

def gen_rel_subset_feature(g, rel_subset, args, device):
    """
    Build relation subgraph given relation subset and generate multi-hop
    neighbor-averaged feature on this subgraph
    """

    new_edges = {}
    ntypes = set()
    for etype in rel_subset:
        stype, _, dtype = g.to_canonical_etype(etype)
        src, dst = g.all_edges(etype=etype)
        src = src.numpy()
        dst = dst.numpy()
        new_edges[(stype, etype, dtype)] = (src, dst)
        new_edges[(dtype, etype + "_r", stype)] = (dst, src)
        ntypes.add(stype)
        ntypes.add(dtype)
    new_g = dgl.heterograph(new_edges)

    # set node feature and calc deg
    for ntype in ntypes:
        num_nodes = new_g.number_of_nodes(ntype)
        if num_nodes < g.nodes[ntype].data["feat"].shape[0]:
            new_g.nodes[ntype].data["hop_0"] = g.nodes[ntype].data["feat"][:num_nodes, :]
        else:
            new_g.nodes[ntype].data["hop_0"] = g.nodes[ntype].data["feat"]
        deg = 0
        for etype in new_g.etypes:
            _, _, dtype = new_g.to_canonical_etype(etype)
            if ntype == dtype:
                deg = deg + new_g.in_degrees(etype=etype)
        norm = 1.0 / deg.float()
        norm[torch.isinf(norm)] = 0
        new_g.nodes[ntype].data["norm"] = norm.view(-1, 1).to(device)

    res = []

    # compute k-hop feature
    for hop in range(1, args.K + 1):
        ntype2feat = {}
        for etype in new_g.etypes:
            stype, _, dtype = new_g.to_canonical_etype(etype)
            new_g[etype].update_all(fn.copy_u(f'hop_{hop-1}', 'm'), fn.sum('m', 'new_feat'))
            new_feat = new_g.nodes[dtype].data.pop("new_feat")
            assert("new_feat" not in new_g.nodes[stype].data)
            if dtype in ntype2feat:
                ntype2feat[dtype] += new_feat
            else:
                ntype2feat[dtype] = new_feat
        for ntype in new_g.ntypes:
            assert ntype in ntype2feat  # because subgraph is not directional
            feat_dict = new_g.nodes[ntype].data
            old_feat = feat_dict.pop(f"hop_{hop-1}")
            if ntype == "paper":
                res.append(old_feat.cpu())
            feat_dict[f"hop_{hop}"] = ntype2feat.pop(ntype).mul_(feat_dict["norm"])

    res.append(new_g.nodes["paper"].data.pop(f"hop_{args.K}").cpu())
    del new_g, feat_dict, new_edges
    clear_memory(device)
    return res

def prepare_data(device, args, fold, mask, probs_path, stage=0, load_embs=False, load_label_emb=False, subset_list=None,test_mode=False):
    """
    Load dataset and compute neighbor-averaged node features used by scalable GNN model
    Note that we select only one integrated representation as node feature input for mlp 
    """
    aggr_device = torch.device("cpu" if args.aggr_gpu < 0 else "cuda:{}".format(args.aggr_gpu))
    emb_path = os.path.join(args.data_dir, '_'.join(args.dataset.split('-')), "embedding",
                f"smoothed_embs_K_{args.K}.pt")
    label_emb_path = os.path.join(args.data_dir, '_'.join(args.dataset.split('-')), "embedding", 
                f"smoothed_label_emb_K_{args.label_K}.pt")
    if not os.path.exists(os.path.dirname(emb_path)):
        os.makedirs(os.path.dirname(emb_path))

    data = load_dataset(aggr_device, args, fold=fold ,mask=mask, test_mode=test_mode)
    t1 = time.time()
    
    g, labels, n_classes, train_nid, val_nid, test_nid = data

    tr_va_te_nid = torch.cat([train_nid, val_nid, test_nid], dim=0)
    ################################ 保留全部节点
    tr_va_te_nid = torch.arange(len(labels))
    ################################
    
    feat_averaging_style = "all" if args.model != "mlp" else "ppnp"
    label_averaging_style = "last"

    train_mask = torch.BoolTensor(np.isin(np.arange(len(labels)), train_nid))
    g.ndata["train_mask"] = train_mask.to(aggr_device)

    in_feats = g.ndata['feat'].shape[1]
    # n_classes = (labels.max() + 1).item() if labels.dim() == 1 else labels.size(1)
    print("in_feats:", in_feats)
    feat = g.ndata.pop('feat')
    print("nodes",feat.shape)

    
    if stage > 0:
        threshold = args.threshold[stage-1] if stage <= len(args.threshold) else args.threshold[-1]
        teacher_probs = torch.load(probs_path).to(aggr_device)

        # assert len(teacher_probs) == len(feat)

        confident_nid_inner = torch.arange(len(teacher_probs))[teacher_probs.max(1)[0] > threshold]
        extra_confident_nid_inner = confident_nid_inner[confident_nid_inner >= len(train_nid)]
        confident_nid = tr_va_te_nid[confident_nid_inner]
        extra_confident_nid = tr_va_te_nid[extra_confident_nid_inner]
        print(f"pseudo label number: {len(confident_nid)}")

        pseudo_labels = torch.argmax(teacher_probs, dim=1).to(labels.device)
        labels_with_pseudos = torch.zeros_like(labels)
        train_nid_with_pseudos = np.union1d(train_nid, confident_nid)
        print(f"enhanced train set number: {len(train_nid_with_pseudos)}")
        labels_with_pseudos[train_nid] = labels[train_nid]
        labels_with_pseudos[extra_confident_nid] = pseudo_labels[extra_confident_nid_inner]
    else:
        teacher_probs = None
        pseudo_labels = None
        labels_with_pseudos = labels.clone()
        confident_nid = train_nid
        train_nid_with_pseudos = train_nid
    
    if args.use_labels & ((not args.inductive) or stage > 0):
        print("using label information")
        label_emb = torch.zeros([feat.shape[0], n_classes]).to(labels.device)
        label_emb[train_nid_with_pseudos] = F.one_hot(labels_with_pseudos[train_nid_with_pseudos], num_classes=n_classes).float().to(labels.device)
    else:
        label_emb = None
    

    # for transductive setting
    if (stage == 0) and load_label_emb and os.path.exists(label_emb_path):
        pass
    else:
        if label_emb is not None:
            label_emb = neighbor_average_features_by_chunks(g, label_emb, args,
                                                            style=label_averaging_style,
                                                            stats=args.dataset not in ["ogbn-papers100M"],
                                                            memory_efficient=args.memory_efficient,
                                                            target_nid=tr_va_te_nid if args.dataset=="ogbn-papers100M" else None)

        if load_label_emb and stage == 0:
            if (not os.path.exists(label_emb_path)):
                print("saving initial label embeddings to " + label_emb_path)
                torch.save(label_emb, label_emb_path)
            del label_emb
            clear_memory(device)

    if load_embs and os.path.exists(emb_path):
        pass
    else:
        feats = neighbor_average_features_by_chunks(g, feat, args,
                                                    style=feat_averaging_style,
                                                    stats=args.dataset not in [ "ogbn-papers100M"],
                                                    memory_efficient=args.memory_efficient,
                                                    target_nid=tr_va_te_nid if args.dataset=="ogbn-papers100M" else None)
        if load_embs:
            if not os.path.exists(emb_path):
                print("saving smoothed node features to " + emb_path)
                torch.save(feats, emb_path)
            del feats, feat
            clear_memory(device)
    del g
    clear_memory(device)

    # save smoothed node features and initial smoothed node label embeddings, 
    # if "load" is set true and they have not been saved
    if load_embs:
        print("load saved embeddings")
        feats = torch.load(emb_path)
    if load_label_emb and (stage == 0):
        print("load saved label embedding")
        label_emb = torch.load(label_emb_path)


    labels = labels.to(device)
    labels_with_pseudos = labels_with_pseudos.to(device)

    train_nid = train_nid.to(device)
    train_nid_with_pseudos = torch.LongTensor(train_nid_with_pseudos).to(device)
    val_nid = val_nid.to(device)
    test_nid = test_nid.to(device)
    t2 = time.time()

    return feats, label_emb, teacher_probs, labels, labels_with_pseudos, in_feats, n_classes, \
        train_nid, train_nid_with_pseudos, val_nid, test_nid, t2 - t1
