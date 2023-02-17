import argparse
import json

import torch
from torch import optim

from baseline_crf import BaselineCrf, train_model, eval_model, format_dict
from imSitu import ImSituVerbRoleLocalNounEncoder, ImSituSituation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="imsitu Situation CRF. Training, evaluation, prediction and features.")
    parser.add_argument("--command", choices=["train", "eval", "predict", "features"], required=True)  # 命令，必选
    parser.add_argument("--output_dir", default="./outputs",
                        help="location to put output, such as models, features, predictions")  # 输出dir
    parser.add_argument("--image_dir", default="./dataset/resized_256", help="location of images to process")  # 图片文件地址
    parser.add_argument("--dataset_dir", default="./dataset", help="location of train.json, dev.json, ect.")  # 数据文件地址
    parser.add_argument("--weights_file", help="the model to start from")  # 参数文件
    parser.add_argument("--encoding_file", help="a file corresponding to the encoder")  # encoder文件
    parser.add_argument("--cnn_type", choices=["resnet_34", "resnet_50", "resnet_101"], default="resnet_101",
                        help="the cnn to initilize the crf with")  # cnn模型选择
    parser.add_argument("--batch_size", default=64, help="batch size for training", type=int)
    parser.add_argument("--learning_rate", default=1e-5, help="learning rate for ADAM", type=float)
    parser.add_argument("--weight_decay", default=5e-4, help="learning rate decay for ADAM", type=float)
    parser.add_argument("--eval_frequency", default=500, help="evaluate on dev set every N training steps", type=int)
    parser.add_argument("--training_epochs", default=20, help="total number of training epochs", type=int)
    parser.add_argument("--eval_file", default="dev.json",
                        help="the dataset file to evaluate on, ex. dev.json test.json")
    parser.add_argument("--top_k", default="10", type=int, help="topk to use for writing predictions to file")

    args = parser.parse_args()
    if args.command == "train":
        print("command = training")
        train_set = json.load(open(args.dataset_dir + "/train.json"))
        dev_set = json.load(open(args.dataset_dir + "/dev.json"))

        if args.encoding_file is None:
            encoder = ImSituVerbRoleLocalNounEncoder(train_set)
            torch.save(encoder, args.output_dir + "encoder")
        else:
            encoder = torch.load(args.encoding_file)

        ngpus = 1
        model = BaselineCrf(encoder, cnn_type=args.cnn_type, ngpus=ngpus)

        if args.weights_file is not None:
            model.load_state_dict(torch.load(args.weights_file))
        dataset_train = ImSituSituation(args.image_dir, train_set, encoder, model.train_preprocess())
        dataset_dev = ImSituSituation(args.image_dir, dev_set, encoder, model.dev_preprocess())

        device_array = [i for i in range(0, ngpus)]
        batch_size = args.batch_size * ngpus

        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=3)
        dev_loader = torch.utils.data.DataLoader(dataset_dev, batch_size=batch_size, shuffle=True, num_workers=3)

        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        train_model(args.training_epochs, args.eval_frequency, train_loader, dev_loader, model, encoder, optimizer,
                    args.output_dir, device_array)
    elif args.command == "eval":
        print("command = evaluating")
        eval_file = json.load(open(args.dataset_dir + "/" + args.eval_file))

        if args.encoding_file is None:
            print("expecting encoder file to run evaluation")
            exit()
        else:
            encoder = torch.load(args.encoding_file)
        print("creating model...")
        model = BaselineCrf(encoder, cnn_type=args.cnn_type)

        if args.weights_file is None:
            print("expecting weight file to run features")
            exit()

        print("loading model weights...")
        model.load_state_dict(torch.load(args.weights_file))
        model.cuda()

        dataset = ImSituSituation(args.image_dir, eval_file, encoder, model.dev_preprocess())
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=3)

        (top1, top5) = eval_model(loader, encoder, model)
        top1_a = top1.get_average_results()
        top5_a = top5.get_average_results()

        avg_score = top1_a["verb"] + top1_a["value"] + top1_a["value-all"] + top5_a["verb"] + top5_a["value"] + top5_a[
            "value-all"] + top5_a["value*"] + top5_a["value-all*"]
        avg_score /= 8

        print("Average :{:.2f} {} {}".format(avg_score * 100, format_dict(top1_a, "{:.2f}", "1-"),
                                             format_dict(top5_a, "{:.2f}", "5-")))
