import argparse
import math
import os
import random
import time
from copy import deepcopy

import dgl
import dgl.function as fn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from dataset import load_dataset
from gen_models import get_model
from pre_process import prepare_data
from train_process import test, train
from utils import read_subset_list, generate_subset_list, get_n_params, seed

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def adjust_lr(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group["lr"] = 0.0001 + (epoch % 5)*0.0001
        # param_group["lr"] = 0.005

epsilon = 1 - math.log(2)
def cross_entropy(x, labels):
    y = F.cross_entropy(x, labels, reduction="none")
    y = torch.log(epsilon + y) - math.log(epsilon)
    return torch.mean(y)

device = torch.device("cpu")
def add_labels(feat, labels, idx):
    onehot = torch.zeros([feat.shape[0], 23]).to(device)
    # onehot[idx, labels[idx, 0]] = 1
    onehot[idx, labels[idx]] = 1
    return torch.cat([feat, onehot], dim=-1)

def run(args, data, device, stage=0, subset_list=None):
    feats, label_emb, teacher_probs, labels, labels_with_pseudos, in_feats, n_classes, \
        train_nid, train_nid_with_pseudos, val_nid, test_nid, _ = data

    # Raw training set loader
    train_loader = torch.utils.data.DataLoader(
        train_nid, batch_size=args.batch_size, shuffle=True, drop_last=False)
    # Enhanced training set loader (but equal to raw one if stage == 0)
    train_loader_with_pseudos = torch.utils.data.DataLoader(
        train_nid_with_pseudos, batch_size=args.batch_size, shuffle=True, drop_last=False)
    # Validation set loader
    val_loader = torch.utils.data.DataLoader(
        val_nid, batch_size=args.eval_batch_size,
        shuffle=False, drop_last=False)
    # Test set loader
    test_loader = torch.utils.data.DataLoader(
        torch.cat([train_nid, val_nid, test_nid], dim=0), batch_size=args.eval_batch_size,
        shuffle=False, drop_last=False)
    # All nodes loader (including nodes without labels)
    print('total nodes', feats[0].shape[0])
    all_loader = torch.utils.data.DataLoader(
        torch.arange(feats[0].shape[0]), batch_size=args.eval_batch_size,
        shuffle=False, drop_last=False)

    # Initialize model and optimizer for each run
    label_in_feats = label_emb.shape[1] if label_emb is not None else n_classes
    model = get_model(in_feats, label_in_feats, n_classes, stage, args, subset_list=subset_list)
    model = model.to(device)
    print("# Params:", get_n_params(model))
    

    # For multiclass classification
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = cross_entropy
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr,
                    weight_decay=args.weight_decay)

    # Start training
    best_epoch = 0
    best_val = 0
    best_val_loss = 1e9
    best_test = 0
    num_epochs = args.epoch_setting[stage]
    train_time = []
    inference_time = []
    val_accs = []
    val_loss = []

    # 节点特征
    base_path = ""
    base_path = ""
    node_info = np.load(os.path.join(base_path, 'node_info.npy'))
    # features_n2v = np.load(os.path.join(base_path, 'features_n2v.npy'))

    features_n2v1 = np.load(os.path.join(base_path, 'features_n2v_1_2.npy'))
    features_n2v2 = np.load(os.path.join(base_path, 'features_n2v_1_1.npy'))
    features_n2v3 = np.load(os.path.join(base_path, 'features_n2v_1_0_5.npy'))
    # features_n2v = (features_n2v1 + features_n2v2 + features_n2v3)/3
    # del features_n2v1,features_n2v2,features_n2v3
    # gc.collect()
    # node_info = np.hstack([node_info,features_n2v])
    # del features_n2v
    walk_label_features = np.load(os.path.join(base_path, 'walk_label_features.npy'))
    # pretrain_feature = np.load(os.path.join(base_path, 'pretrain_feature.npy'))
    # kmeans_features = np.load(os.path.join(base_path, 'clustering_features.npy'))
    node_info = np.hstack([node_info, features_n2v1,features_n2v2,features_n2v3, walk_label_features])
    del walk_label_features, features_n2v1,features_n2v2,features_n2v3
    gc.collect()
    node_info = torch.tensor(node_info)

    stop_count = 0
    for epoch in range(1, num_epochs + 1):
        start = time.time()
        ############ 调整学习率
        adjust_lr(optimizer, epoch)
        ##############
        # 数据切换
        if epoch % 10 == 0:
            print('#################data mask chang!!!####################')
            del feats, label_emb, teacher_probs, labels, labels_with_pseudos, in_feats, n_classes, \
        train_nid, train_nid_with_pseudos, val_nid, test_nid, _
            gc.collect()
            probs_path = ""
            feats, label_emb, teacher_probs, labels, labels_with_pseudos, in_feats, n_classes, \
            train_nid, train_nid_with_pseudos, val_nid, test_nid, _ = prepare_data(device, args, fold, mask, probs_path,
                                                                                   stage, load_embs=args.load_embs,
                                                                                   load_label_emb=args.load_label_emb,
                                                                                   subset_list=subset_list)
            # Enhanced training set loader (but equal to raw one if stage == 0)
            train_loader_with_pseudos = torch.utils.data.DataLoader(
                train_nid_with_pseudos, batch_size=args.batch_size, shuffle=True, drop_last=False)
            # Validation set loader
            val_loader = torch.utils.data.DataLoader(
                val_nid, batch_size=args.eval_batch_size,
                shuffle=False, drop_last=False)
            # Test set loader
            test_loader = torch.utils.data.DataLoader(
                torch.cat([train_nid, val_nid, test_nid], dim=0), batch_size=args.eval_batch_size,
                shuffle=False, drop_last=False)
            # All nodes loader (including nodes without labels)
            print('total nodes', feats[0].shape[0])
            all_loader = torch.utils.data.DataLoader(
                torch.arange(feats[0].shape[0]), batch_size=args.eval_batch_size,
                shuffle=False, drop_last=False)



        train(model, feats, node_info, label_emb, teacher_probs, labels_with_pseudos, loss_fcn, optimizer, train_loader_with_pseudos, args)
        med = time.time()

        if epoch % args.eval_every == 0:
            with torch.no_grad():
                acc = test(model, feats, node_info, label_emb, teacher_probs, labels, loss_fcn, val_loader, test_loader,
                           train_nid, val_nid, test_nid, args)
            end = time.time()

            # We can choose val_acc or val_loss to select best model (usually it does not matter)
            # if (acc[1] > best_val and args.acc_loss == "acc") or (acc[3] < best_val_loss and args.acc_loss == "loss"):
            if (acc[1] > best_val and args.acc_loss == "acc"):
                best_epoch = epoch
                best_val = acc[1]
                best_test = acc[2]
                best_val_loss = acc[3]
                best_model = deepcopy(model)
                # best_feats = deepcopy(feats)
                stop_count = 0
            else:
                stop_count += 1
            if stop_count == 15:
                break

            train_time.append(med - start)
            inference_time.append(acc[-1])
            val_accs.append(acc[1])
            val_loss.append(acc[-2])
            log = "Epoch {}, Time(s): {:.4f} {:.4f}, ".format(epoch, med - start, acc[-1])
            log += "Best Val loss: {:.4f}, Accs: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}, Best Val: {:.4f}, Best Test: {:.4f}".format(best_val_loss, acc[0], acc[1], acc[2], best_val, best_test)
            print(log)
            
    print("Stage: {}, Best Epoch {}, Val {:.4f}, Test {:.4f}".format(
        stage, best_epoch, best_val, best_test))

    # 测试阶段重新生成数据，考虑全部use label
    # del feats, label_emb, teacher_probs, labels, labels_with_pseudos, model
    # gc.collect()
    # with torch.cuda.device(device):
    #     torch.cuda.empty_cache()
    # probs_path = ''
    # with torch.no_grad():
    #     feats, label_emb, teacher_probs, labels, labels_with_pseudos, in_feats, n_classes, \
    #     train_nid, train_nid_with_pseudos, val_nid, test_nid, _ = prepare_data(device, args, fold, mask, probs_path,
    #                                                                            stage, load_embs=args.load_embs,
    #                                                                            load_label_emb=args.load_label_emb,
    #                                                                            subset_list=subset_list,
    #                                                                            test_mode=True)



    # inference
    with torch.no_grad():
        best_model.eval()
        ############
        # feats = deepcopy(best_feats) # 使用最佳mode那个epoch的数据
        ############
        probs = []
        if (args.model in ["sagn", "plain_sagn"] and args.weight_style=="attention") and (not args.avoid_features):
            attn_weights = []
        else:
            attn_weights = None
        for batch in all_loader:
        # for batch in test_loader:
            batch = batch.long()
            batch_feats = [x[batch].to(device) for x in feats] if isinstance(feats, list) else feats[batch].to(device)
            if label_emb is not None:
                batch_label_emb = label_emb[batch].to(device)
            else:
                batch_label_emb = None
            # 节点特征
            batch_node_info = node_info[batch].to(device)
            if (args.model in ["sagn", "plain_sagn"]) and (not args.avoid_features):
                out, a = best_model(batch_feats, batch_node_info, batch_label_emb)
            else:
                out = best_model(batch_feats, batch_label_emb)

            out = out.softmax(dim=1)
            # remember to transfer output probabilities to cpu
            probs.append(out.cpu())
            if (args.model in ["sagn", "plain_sagn"] and args.weight_style=="attention") and (not args.avoid_features):
                attn_weights.append(a.cpu().squeeze(1).squeeze(1))
        probs = torch.cat(probs, dim=0)
        if (args.model in ["sagn", "plain_sagn"] and args.weight_style=="attention") and (not args.avoid_features):
            attn_weights = torch.cat(attn_weights)
        
    del best_model
    del model
    del feats, label_emb, teacher_probs, labels, labels_with_pseudos
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
    return best_val, best_test, probs, train_time, inference_time, val_accs, val_loss, attn_weights


def main(args,fold,mask):
    device = torch.device("cpu" if args.gpu < 0 else "cuda:{}".format(args.gpu))
    aggr_device = torch.device("cpu" if args.aggr_gpu < 0 else f"cuda:{args.aggr_gpu}")


    total_best_val_accs = []
    total_best_test_accs = []
    total_val_accs = []
    total_val_losses = []
    total_preprocessing_times = []
    total_train_times = []
    total_inference_times = []

    for i in range(args.num_runs):
        print("-" * 100)
        print(f"Run {i} start training")
        seed(seed=args.seed + i)
        

        subset_list = None

        best_val_accs = []
        best_test_accs = []
        val_accs = []
        val_losses = []

        preprocessing_times = []
        train_times = []
        inference_times = []

        for stage in range(len(args.epoch_setting)):

            # if stage > 0:
            #     args.load_embs = True

            
            if args.warmup_stage > -1:
                if stage <= args.warmup_stage:
                    probs_path = os.path.join(args.probs_dir, 
                                              args.dataset, 
                                              args.model if (args.weight_style == "attention") else (args.model + "_" + args.weight_style),
                                              f'use_labels_{args.use_labels}_use_feats_{not args.avoid_features}_K_{args.K}_label_K_{args.label_K}_probs_seed_{args.seed + i}_stage_{stage}.pt')
                    print(probs_path)
                    if os.path.exists(probs_path):
                        print(f"bypass stage {stage} since warmup_stage is set and associated file exists.")
                        continue
            print("-" * 100)
            print(f"Stage {stage} start training")
            if stage > 0:
                probs_path = os.path.join(args.probs_dir, args.dataset, 
                                args.model if (args.weight_style == "attention") else (args.model + "_" + args.weight_style),
                                f'use_labels_{args.use_labels}_use_feats_{not args.avoid_features}_K_{args.K}_label_K_{args.label_K}_probs_seed_{args.seed + i}_stage_{stage - 1}_fold_{fold}_{best_val}.pt')
            else:
                probs_path = ''

            with torch.no_grad():
                data = prepare_data(device, args, fold, mask, probs_path, stage, load_embs=args.load_embs, load_label_emb=args.load_label_emb, subset_list=subset_list)
            preprocessing_times.append(data[-1])
            print(f"Preprocessing costs {(data[-1]):.4f} s")
            best_val, best_test, probs, train_time, inference_time, val_acc, val_loss, attn_weights = run(args, data, device, stage, subset_list=subset_list)
            train_times.append(train_time)
            inference_times.append(inference_time)
            val_accs.append(val_acc[0].cpu().numpy())
            val_losses.append(val_loss)
            new_probs_path = os.path.join(args.probs_dir, args.dataset, 
                                args.model if (args.weight_style == "attention") else (args.model + "_" + args.weight_style),
                                f'use_labels_{args.use_labels}_use_feats_{not args.avoid_features}_K_{args.K}_label_K_{args.label_K}_probs_seed_{args.seed + i}_stage_{stage}_fold_{fold}_{best_val}.pt')
            if not os.path.exists(os.path.dirname(new_probs_path)):
                os.makedirs(os.path.dirname(new_probs_path))
            # print(probs.shape)
            # print(probs)
            torch.save(probs, new_probs_path)
            best_val_accs.append(best_val.cpu().numpy())
            best_test_accs.append(best_test.cpu().numpy())


            del data, probs, attn_weights
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        print(best_val_accs)
        print(best_test_accs)
        total_best_val_accs.append(best_val_accs)
        total_best_test_accs.append(best_test_accs)
        total_val_accs.append(val_accs)
        total_val_accs.append(val_losses)
        total_preprocessing_times.append(preprocessing_times)
        total_train_times.append(train_times)
        total_inference_times.append(inference_times)

    total_best_val_accs = np.array(total_best_val_accs)
    total_best_test_accs = np.array(total_best_test_accs)
    # total_val_accs = np.array(total_val_accs)
    total_preprocessing_times = np.array(total_preprocessing_times)
    total_train_times = np.array(total_train_times, dtype=object)
    total_inference_times = np.array(total_inference_times, dtype=object)


    for stage in range(len(args.epoch_setting)):
        print(f"Stage: {stage}, Val accuracy: {np.mean(total_best_val_accs[:, stage]):.4f}±"
            f"{np.std(total_best_val_accs[:, stage]):.4f}")
        print(f"Stage: {stage}, Test accuracy: {np.mean(total_best_test_accs[:, stage]):.4f}±"
            f"{np.std(total_best_test_accs[:, stage]):.4f}")
        print(f"Stage: {stage}, Preprocessing time: {np.mean(total_preprocessing_times[:, stage]):.4f}±"
            f"{np.std(total_preprocessing_times[:, stage]):.4f}")
        print(f"Stage: {stage}, Training time: {np.hstack(total_train_times[:, stage]).mean():.4f}±"
            f"{np.hstack(total_train_times[:, stage]).std():.4f}")
        print(f"Stage: {stage}, Inference time: {np.hstack(total_inference_times[:, stage]).mean():.4f}±"
            f"{np.hstack(total_inference_times[:, stage]).std():.4f}")

    gc.collect()


def define_parser():
    parser = argparse.ArgumentParser(description="Scalable Adaptive Graph neural Networks with Self-Label-Enhance")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epoch-setting", nargs='+',type=int, default=[2000])
    parser.add_argument("--warmup-stage", type=int, default=-1,
                        help="(Only for testing) select the stage from which the script starts to train \
                              based on trained files, -1 for cold starting")
    parser.add_argument("--load-embs", action="store_true",
                        help="This option is used to save memory cost when performing aggregations.", default=False)
    parser.add_argument("--load-label-emb", action="store_true",
                        help="This option is used to save memory cost when performing first label propagation.", default=False)
    parser.add_argument("--acc-loss", type=str, default="acc")
    parser.add_argument("--avoid-features", action="store_true")
    parser.add_argument("--use-labels", action="store_true", default=False)
    parser.add_argument("--inductive", action="store_true")
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--use-norm", action='store_true')
    parser.add_argument("--memory-efficient", action='store_true', default=True)
    parser.add_argument("--num-hidden", type=int, default=512)
    parser.add_argument("--K", type=int, default=4,
                        help="number of hops")
    parser.add_argument("--label-K", type=int, default=1,
                        help="number of label propagation hops")
    parser.add_argument("--zero-inits", action="store_true", 
                        help="Whether to initialize hop attention vector as zeros", default=True)
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--dataset", type=str, default="ogbn-papers100M")
    parser.add_argument("--data_dir", type=str, default="/mnt/ssd/ssd/dataset")
    parser.add_argument("--model", type=str, default="sagn")
    parser.add_argument("--pretrain-model", type=str, default="ComplEx")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--weight-style", type=str, default="attention")
    parser.add_argument("--focal", type=str, default="first")
    parser.add_argument("--mag-emb", action="store_true")
    parser.add_argument("--position-emb", action="store_true")
    parser.add_argument("--label-residual", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.36966048605288365,
                        help="dropout on activation")
    parser.add_argument("--input-drop", type=float, default=0.035775444928082135, # 考虑n2v时需要drop
                        help="dropout on input features")
    parser.add_argument("--attn-drop", type=float, default=0.17219661510062576,
                        help="dropout on hop-wise attention scores")
    parser.add_argument("--label-drop", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--aggr-gpu", type=int, default=-1)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4096)
    # parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=100000,
                        help="evaluation batch size")
    parser.add_argument("--mlp-layer", type=int, default=3,
                        help="number of MLP layers")
    parser.add_argument("--label-mlp-layer", type=int, default=4,
                        help="number of label MLP layers")
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--threshold", type=float, nargs="+", default=[0.95],
                        help="threshold used to generate pseudo hard labels")
    parser.add_argument("--num-runs", type=int, default=1,
                        help="number of times to repeat the experiment")
    parser.add_argument("--example-subsets-path", type=str, default="/home/scx/NARS/sample_relation_subsets/examples")
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--fixed-subsets", action="store_true")
    parser.add_argument("--emb-path", type=str, default="/home/scx/NARS/")
    parser.add_argument("--probs_dir", type=str, default="../intermediate_outputs")
    return parser

if __name__ == "__main__":
    parser = define_parser()
    args = parser.parse_args()
    print(args)
    for fold in range(7):
        for mask in range(2):
            # if fold==7 and mask==0:
            #     continue
            # else:
            main(args, fold, mask)


