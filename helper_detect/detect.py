import torch
import pdb
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse
import pandas as pd
from helper_detect.load_data import load_data, load_data_adv


from cfg import *

from misc import (
    args_handling,
    print_args,
    create_dir,
    save_to_pt,
    create_pth,
    create_log_file,
    save_log
)


if __name__ == "__main__":
    # processing the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "mnist", "butterfly32", "celebaHQ256"], help="dataset LR or RF is trained.")
    parser.add_argument("--dataset_transfer", default="cifar10", help="cifar10, butterfly32")
    parser.add_argument("--extract", default="features", choices=["features",  "fft", "multiLID"], help="normal, features, fft, multiLID")
    parser.add_argument("--model", default="rn18", help="ResNet18")
    parser.add_argument("--mode",  default="nor", choices=["nor", "adv"], help="normal, adv")
    parser.add_argument("--clf", default="lr",  choices=["lr", "rf"], help="lr, rf")
    parser.add_argument("--nrsamples", default=1000, help="number samples")
    parser.add_argument("--load_clf", default=False, help="number samples")

    parser.add_argument("--load_extr_nor", default="nor_train.pt", help="save_extr_nor")
    parser.add_argument("--load_extr_adv", default="adv_ddpm.pt",  help="save_extr_adv")

    parser.add_argument("--save_json", default="", help="Save settings to file in json format. Ignored in json file")
    parser.add_argument("--load_json", default="", help="Load settings from file in json format. Command line options override values in file.")
    args = parser.parse_args()

    # args.save_json = os.path.join(cfg_detect_path, args.model + "_" + args.clf + "_" + args.dataset + "-" + args.dataset_transfer)
    # args.load_json = os.path.join(cfg_detect_path, load_cfg)
    # args.load_json = os.path.join(cfg_detect_path, args.mode + "_" + args.dataset)

    load_cfg = args.load_json

    args = args_handling(args, parser, cfg_detect_path)
    print_args(args)

    log = create_log_file(args)

    random_state = 21
    factor = 0.6
    train_samples = int(args.nrsamples*factor)

    X_nor, X_adv = load_data(args, train_samples, args.dataset)

    print("X_nor shape> ", X_nor.shape)
    print("X_adv shape> ", X_adv.shape)
    assert(X_nor.shape[0] == X_adv.shape[0])

    y_nor = np.zeros(args.nrsamples)
    y_adv = np.ones(args.nrsamples)


    if "10000_multiLID_target_30classes" in args.load_json:
        y_nor = pd.read_csv("/home/lorenzp/DeepfakeDetection/analysis/artifact/all_real10000_target.csv")[:10000]['target'].to_numpy()
        y_adv = pd.read_csv("/home/lorenzp/DeepfakeDetection/analysis/artifact/all_fake10000_target.csv")[:10000]['target'].to_numpy()
    elif "10000_multiLID_target_3classes" in args.load_json:
        y_nor = pd.read_csv("/home/lorenzp/DeepfakeDetection/analysis/artifact/all_real10000_target_3classes.csv")[:10000]['target'].to_numpy()
        y_adv = pd.read_csv("/home/lorenzp/DeepfakeDetection/analysis/artifact/all_fake10000_target_3classes.csv")[:10000]['target'].to_numpy()
    elif "10500_multiLID_target_balanced" in args.load_json:
        y_nor = pd.read_csv("/home/lorenzp/DeepfakeDetection/analysis/artifact/all_real10500_target_3classes_balanced.csv")[:10500]['target'].to_numpy()
        y_adv = pd.read_csv("/home/lorenzp/DeepfakeDetection/analysis/artifact/all_fake10500_target_3classes_balanced.csv")[:10500]['target'].to_numpy()
    elif "20000_multiLID_target_balanced" in args.load_json:
        Xnor_2 = torch.load("/home/lorenzp/workspace/DeepfakeDetection/results/extract/af-all/wb-multiLID/nor_af-all_10000_diff_balanced_csv.pt")[:10000]
        Xadv_2 = torch.load("/home/lorenzp/workspace/DeepfakeDetection/results/extract/af-all/wb-multiLID/adv_af-all_10000_diff_balanced_csv.pt")[:10000]
        Xnor_2 = Xnor_2.reshape((Xnor_2.shape[0], -1))
        Xadv_2 = Xadv_2.reshape((Xadv_2.shape[0], -1))

        X_nor = np.vstack([X_nor, Xnor_2])
        X_adv = np.vstack([X_adv, Xadv_2])

        y_nor = np.zeros(10000*2)
        y_adv = np.np.concatenate([np.ones(10000), np.ones(10000)*2])



    if hasattr(args, 'load_extr_adv1') or hasattr(args, 'load_extr_adv2'):
        adv1 = load_data_adv(args, args.nrsamples, args.dataset, args.load_extr_adv1)
        adv2 = load_data_adv(args, args.nrsamples, args.dataset, args.load_extr_adv2)

        X_adv = np.vstack([X_adv, adv1, adv2])

    if hasattr(args, 'load_extr_adv1'):
        y_adv0 = np.ones(args.nrsamples, dtype=np.uint)
        y_adv1 = np.ones(args.nrsamples, dtype=np.uint)*2
        y_adv2 = np.ones(args.nrsamples, dtype=np.uint)*3

        y_adv = np.concatenate([y_adv0, y_adv1, y_adv2])


    x_train_n, x_test_n, y_train_n, y_test_n = train_test_split(X_nor, y_nor, test_size=1-factor, train_size=factor, random_state=random_state)
    x_train_a, x_test_a, y_train_a, y_test_a = train_test_split(X_adv, y_adv, test_size=1-factor, train_size=factor, random_state=random_state)

    x_train = np.concatenate((x_train_n, x_train_a))
    y_train = np.concatenate((y_train_n, y_train_a))
    
    if args.extract == "features":
        X_train = x_train.reshape(*x_train.shape[:-3], -1)
    else:
        X_train = x_train

    if args.clf == "lr":
        if "10000_multiLID_target_3classes" in args.load_json:
            clf = LogisticRegression(class_weight={0:1, 1:2, 2:4})
        else:
            clf = LogisticRegression()
    
    elif args.clf == "rf":
        if "10000_multiLID_target_3classes" in args.load_json:
            clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight={0:1, 1:2, 2:4})
        elif  "10500_multiLID_target_balanced" in args.load_json or '20000_multiLID_target_balanced' in args.load_json:
            clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight={0:1, 1:2, 2:2})
        else:
            clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        clf.set_params(random_state=21, verbose=0)

    if args.load_clf:
        print("load classifier ...")
        raise NotImplementedError("load clf")
    else:
        print("train classifier ...")
        
        clf.fit(X_train,y_train)


    x_test = np.concatenate((x_test_n, x_test_a))
    y_test = np.concatenate((y_test_n, y_test_a))


    if args.extract == "features":
        X_test = x_test.reshape(*x_test.shape[:-3], -1)
    else:
        X_test = x_test

    print ("predict ...")
    

    predictions = clf.predict(X_test)


    if "10000_multiLID_target_30classes" in args.load_json:
        cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        fig = disp.plot()
        fig.figure_.savefig('/home/lorenzp/DeepfakeDetection/analysis/artifact/{}_conf_mat_30classes.png'.format(args.clf),  bbox_inches='tight', dpi=300)
        
    elif "10000_multiLID_target_3classes" in args.load_json:
        cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        fig = disp.plot()

        fig.figure_.savefig('/home/lorenzp/DeepfakeDetection/analysis/artifact/{}_conf_mat_3classes.png'.format(args.clf),  bbox_inches='tight', dpi=300)

    elif "10500_multiLID_target_balanced" in args.load_json:
        cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        fig = disp.plot()

        fig.ax_.set_title("ArtiFact")

        fig.ax_.xaxis.set_ticklabels(['real', 'gan', 'diff'])
        fig.ax_.yaxis.set_ticklabels(['real', 'gan', 'diff'])
        
        fig.figure_.savefig('/home/lorenzp/DeepfakeDetection/analysis/artifact/{}_conf_mat_3target_balanced.png'.format(args.clf),  bbox_inches='tight', dpi=300)

    elif '20000_multiLID_target_balanced' in args.load_json:
        cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
        fig = disp.plot()

        fig.figure_.savefig('/home/lorenzp/DeepfakeDetection/analysis/artifact/{}_conf_mat_3target_20000_balanced.png'.format(args.clf),  bbox_inches='tight', dpi=300)


    score = clf.score(X_test, y_test)
    print("dataset train/test> ", args.dataset)

    auroc = -1
    if not hasattr(args, 'load_extr_adv1'):
        fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1) # works only for binary problems
        auroc = metrics.auc(fpr, tpr)
    print("score> ", score, ", auroc> ", auroc)

    if not args.load_clf:
        path = os.path.join(ws_detect_path, args.load_json ) 
        basename = os.path.basename(path).replace("json", "clf")
        dirname = os.path.dirname(path)
        create_dir(dirname)

        save_clf = os.path.join(dirname, basename)
        print("save clf> ", save_clf)
        torch.save( clf, save_clf )

    log["score"] = score
    log["auroc"] = auroc

    if args.dataset != args.dataset_transfer or (len(args.load_extr_nor_trans)>0 and not (args.load_extr_nor == args.load_extr_nor_trans)):
        transfer_adv, transfer_nor = load_data(args, train_samples, args.dataset_transfer, transfer=True)
        test_samples = int(transfer_nor.shape[0]*(1-factor))
        transfer_test_nor = transfer_nor[test_samples:]
        transfer_test_adv = transfer_adv[test_samples:]


        y_test_nor = np.zeros(transfer_test_adv.shape[0])
        y_test_adv = np.ones(transfer_test_adv.shape[0])


        x_test = np.concatenate((transfer_test_nor, transfer_test_adv))
        y_test = np.concatenate((y_test_nor, y_test_adv))

        if args.extract == "features":
            X_test = x_test.reshape(*x_test.shape[:-3], -1)
        else:
            X_test = x_test

        print ("predict ...")
        predictions = clf.predict(X_test)

        score_transfer = clf.score(X_test, y_test)
        print("transfer test>  ", args.dataset_transfer)
        print("score_transfer> ", score_transfer)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)
        auroc_transfer = metrics.auc(fpr, tpr)
        print("auroc_transfer> ", auroc_transfer)

        log["score_transfer"] = score_transfer
        log["auroc_transfer"] = auroc_transfer

    save_log(args, log, load_cfg)