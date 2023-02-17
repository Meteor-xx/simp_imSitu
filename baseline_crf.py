import time

import torch
from torch import nn
import torchvision as tv
from torchvision.models import ResNet101_Weights, ResNet50_Weights, ResNet34_Weights

from imSitu import ImSituTensorEvaluation
from utils import init_linear
from torch.autograd import Variable


class ResnetModifiedLarge(nn.Module):
    def __init__(self):
        super(ResnetModifiedLarge, self).__init__()
        self.resnet = tv.models.resnet101(weights=ResNet101_Weights.DEFAULT)
        # probably want linear, relu, dropout
        self.linear = nn.Linear(7 * 7 * 2048, 1024)
        self.dropout2d = nn.Dropout2d(.5)
        self.dropout = nn.Dropout(.5)
        self.relu = nn.LeakyReLU()
        init_linear(self.linear)

    def base_size(self):
        return 2048

    def rep_size(self):
        return 1024

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.dropout2d(x)

        # print x.size()
        return self.dropout(self.relu(self.linear(x.view(-1, 7 * 7 * self.base_size()))))


class ResnetModifiedMedium(nn.Module):
    def __init__(self):
        super(ResnetModifiedMedium, self).__init__()
        self.resnet = tv.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # probably want linear, relu, dropout
        self.linear = nn.Linear(7 * 7 * 2048, 1024)
        self.dropout2d = nn.Dropout2d(.5)
        self.dropout = nn.Dropout(.5)
        self.relu = nn.LeakyReLU()
        init_linear(self.linear)

    def base_size(self):
        return 2048

    def rep_size(self):
        return 1024

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.dropout2d(x)

        # print x.size()
        return self.dropout(self.relu(self.linear(x.view(-1, 7 * 7 * self.base_size()))))


class ResnetModifiedSmall(nn.Module):
    def __init__(self):
        super(ResnetModifiedSmall, self).__init__()
        self.resnet = tv.models.resnet34(weights=ResNet34_Weights.DEFAULT)
        # probably want linear, relu, dropout
        self.linear = nn.Linear(7 * 7 * 512, 1024)
        self.dropout2d = nn.Dropout2d(.5)
        self.dropout = nn.Dropout(.5)
        self.relu = nn.LeakyReLU()
        init_linear(self.linear)

    def base_size(self):
        return 512

    def rep_size(self):
        return 1024

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.dropout2d(x)

        return self.dropout(self.relu(self.linear(x.view(-1, 7 * 7 * self.base_size()))))


class BaselineCrf(nn.Module):
    def train_preprocess(self):
        return self.train_transform

    def dev_preprocess(self):
        return self.dev_transform

    # these seem like decent splits of imsitu, freq = 0,50,100,282 , prediction type can be "max_max" or "max_marginal"
    def __init__(self, encoding, splits=[50, 100, 283], prediction_type="max_max", ngpus=1, cnn_type="resnet_101"):
        super(BaselineCrf, self).__init__()

        self.normalize = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.train_transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.RandomCrop(224),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            self.normalize,
        ])

        self.dev_transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            self.normalize,
        ])

        self.broadcast = []
        self.nsplits = len(splits)
        self.splits = splits
        self.encoding = encoding
        self.prediction_type = prediction_type
        self.n_verbs = encoding.n_verbs()
        self.split_vr = {}
        self.v_roles = {}
        # cnn
        print(cnn_type)
        if cnn_type == "resnet_101":
            self.cnn = ResnetModifiedLarge()
        elif cnn_type == "resnet_50":
            self.cnn = ResnetModifiedMedium()
        elif cnn_type == "resnet_34":
            self.cnn = ResnetModifiedSmall()
        else:
            print("unknown base network")
            exit()
        self.rep_size = self.cnn.rep_size()
        for s in range(0, len(splits)):
            self.split_vr[s] = []

        # sort by length
        remapping = []
        for (vr, ns) in encoding.vr_id_n.items():
            remapping.append((vr, len(ns)))

        # find the right split
        for (vr, l) in remapping:
            i = 0
            for s in splits:
                if l <= s: break
                i += 1
            _id = (i, vr)
            self.split_vr[i].append(_id)
        total = 0
        for (k, v) in self.split_vr.items():
            # print "{} {} {}".format(k, len(v), splits[k]*len(v))
            total += splits[k] * len(v)
            # print "total compute : {}".format(total)

        # keep the splits sorted by vr id, to keep the model const w.r.t the encoding
        for i in range(0, len(splits)):
            s = sorted(self.split_vr[i], key=lambda x: x[1])
            self.split_vr[i] = []
            # enumerate?
            for (x, vr) in s:
                _id = (x, len(self.split_vr[i]), vr)
                self.split_vr[i].append(_id)
                (v, r) = encoding.id_vr[vr]
                if v not in self.v_roles: self.v_roles[v] = []
                self.v_roles[v].append(_id)

        # create the mapping for grouping the roles back to the verbs later
        max_roles = encoding.max_roles()

        # need a list that is nverbs by 6
        self.v_vr = [0 for i in range(0, self.encoding.n_verbs() * max_roles)]
        splits_offset = []
        for i in range(0, len(splits)):
            if i == 0:
                splits_offset.append(0)
            else:
                splits_offset.append(splits_offset[-1] + len(self.split_vr[i - 1]))

        # and we need to compute the position of the corresponding roles, and pad with the 0 symbol
        for i in range(0, self.encoding.n_verbs()):
            offset = max_roles * i
            roles = sorted(self.v_roles[i], key=lambda x: x[2])  # stored in role order
            self.v_roles[i] = roles
            k = 0
            for (s, pos, r) in roles:
                # add one to account of the 0th element being the padding
                self.v_vr[offset + k] = splits_offset[s] + pos + 1
                k += 1
            # pad
            while k < max_roles:
                self.v_vr[offset + k] = 0
                k += 1

        gv_vr = Variable(torch.LongTensor(self.v_vr).cuda())  # .view(self.encoding.n_verbs(), -1)
        for g in range(0, ngpus):
            self.broadcast.append(Variable(torch.LongTensor(self.v_vr).cuda(g)))
        self.v_vr = gv_vr
        # print self.v_vr

        # verb potential
        self.linear_v = nn.Linear(self.rep_size, self.encoding.n_verbs())
        # verb-role-noun potentials
        self.linear_vrn = nn.ModuleList(
            [nn.Linear(self.rep_size, splits[i] * len(self.split_vr[i])) for i in range(0, len(splits))])
        self.total_vrn = 0
        for i in range(0, len(splits)): self.total_vrn += splits[i] * len(self.split_vr[i])
        print("total encoding vrn : {0}, with padding in {1} groups : {2}".format(encoding.n_verbrolenoun(),
                                                                                  self.total_vrn, len(splits)))

        # initilize everything
        init_linear(self.linear_v)
        for _l in self.linear_vrn:
            init_linear(_l)
        self.mask_args()

    def mask_args(self):
        # go through the and set the weights to negative infinity for out of domain items
        neg_inf = float("-infinity")
        for v in range(0, self.encoding.n_verbs()):
            for (s, pos, r) in self.v_roles[v]:
                linear = self.linear_vrn[s]
                # get the offset
                #         print self.splits
                start = self.splits[s] * pos + len(self.encoding.vr_n_id[r])
                end = self.splits[s] * (pos + 1)
                for k in range(start, end):
                    linear.bias.data[k] = -100  # neg_inf

    # expects a list of vectors, BxD
    # returns the max index of every vector, max value of each vector and the log_sum_exp of the vector
    def log_sum_exp(self, vec):
        max_score, max_i = torch.max(vec, 1)
        max_score_broadcast = max_score.view(-1, 1).expand(vec.size())
        return (max_i, max_score, max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), 1)))

    def forward_max(self, images):
        (_, _, _, _, scores, values) = self.forward(images)
        return (scores, values)

    def forward_features(self, images):
        return self.cnn(images)

    def forward(self, image):
        batch_size = image.size()[0]

        rep = self.cnn(image)
        # print self.rep_size
        # print batch_size
        v_potential = self.linear_v(rep)

        vrn_potential = []
        vrn_marginal = []
        vr_max = []
        vr_maxi = []
        # first compute the norm
        # step 1 compute the verb-role marginals
        # this loop allows a memory/parrelism tradeoff.
        # To use less memory but achieve less parrelism, increase the number of groups
        for i, vrn_group in enumerate(self.linear_vrn):
            # linear for the group
            _vrn = vrn_group(rep).view(-1, self.splits[i])

            _vr_maxi, _vr_max, _vrn_marginal = self.log_sum_exp(_vrn)
            _vr_maxi = _vr_maxi.view(-1, len(self.split_vr[i]))
            _vr_max = _vr_max.view(-1, len(self.split_vr[i]))
            _vrn_marginal = _vrn_marginal.view(-1, len(self.split_vr[i]))

            vr_maxi.append(_vr_maxi)
            vr_max.append(_vr_max)
            vrn_potential.append(_vrn.view(batch_size, -1, self.splits[i]))
            vrn_marginal.append(_vrn_marginal)

        # concat role groups with the padding symbol
        zeros = Variable(torch.zeros(batch_size, 1).cuda())  # this is the padding
        zerosi = Variable(torch.LongTensor(batch_size, 1).zero_().cuda())
        vrn_marginal.insert(0, zeros)
        vr_max.insert(0, zeros)
        vr_maxi.insert(0, zerosi)

        # print vrn_marginal
        vrn_marginal = torch.cat(vrn_marginal, 1)
        vr_max = torch.cat(vr_max, 1)
        vr_maxi = torch.cat(vr_maxi, 1)

        # print vrn_marginal
        # step 2 compute verb marginals
        # we need to reorganize the role potentials so it is BxVxR
        # gather the marginals in the right way
        v_vr = self.broadcast[torch.cuda.current_device()]
        vrn_marginal_grouped = vrn_marginal.index_select(1, v_vr).view(batch_size, self.n_verbs,
                                                                       self.encoding.max_roles())
        vr_max_grouped = vr_max.index_select(1, v_vr).view(batch_size, self.n_verbs, self.encoding.max_roles())
        vr_maxi_grouped = vr_maxi.index_select(1, v_vr).view(batch_size, self.n_verbs, self.encoding.max_roles())

        # product ( sum since we are in log space )
        v_marginal = torch.sum(vrn_marginal_grouped, 2).view(batch_size, self.n_verbs) + v_potential

        # step 3 compute the final sum over verbs
        _, _, norm = self.log_sum_exp(v_marginal)
        # compute the maxes

        # max_max probs
        v_max = torch.sum(vr_max_grouped, 2).view(batch_size, self.n_verbs) + v_potential  # these are the scores
        # we don't actually care, we want a max prediction per verb
        # max_max_vi , max_max_v_score = max(v_max,1)
        # max_max_prob = exp(max_max_v_score - norm)
        # max_max_vrn_i = vr_maxi_grouped.gather(1,max_max_vi.view(batch_size,1,1).expand(batch_size,1,self.max_roles))

        # offset so we can use index select... is there a better way to do this?
        # max_marginal probs
        # max_marg_vi , max_marginal_verb_score = max(v_marginal, 1)
        # max_marginal_prob = exp(max_marginal_verb_score - norm)
        # max_marg_vrn_i = vr_maxi_grouped.gather(1,max_marg_vi.view(batch_size,1,1).expand(batch_size,1,self.max_roles))

        # this potentially does not work with parrelism, in which case we should figure something out
        if self.prediction_type == "max_max":
            rv = (rep, v_potential, vrn_potential, norm, v_max, vr_maxi_grouped)
        elif self.prediction_type == "max_marginal":
            rv = (rep, v_potential, vrn_potential, norm, v_marginal, vr_maxi_grouped)
        else:
            print("unkown inference type")
            rv = ()
        return rv

    # computes log( (1 - exp(x)) * (1 - exp(y)) ) =  1 - exp(y) - exp(x) + exp(y)*exp(x) = 1 - exp(V), so V=  log(exp(y) + exp(x) - exp(x)*exp(y))
    # returns the the log of V
    def logsumexp_nx_ny_xy(self, x, y):
        # _,_, v = self.log_sum_exp(torch.cat([x, y, torch.log(torch.exp(x+y))]).view(1,3))
        if x > y:
            return torch.log(torch.exp(y - x) + 1 - torch.exp(y) + 1e-8) + x
        else:
            return torch.log(torch.exp(x - y) + 1 - torch.exp(x) + 1e-8) + y

    def sum_loss(self, v_potential, vrn_potential, norm, situations, n_refs):
        # compute the mil losses... perhaps this should be a different method to facilitate parrelism?
        batch_size = v_potential.size()[0]
        mr = self.encoding.max_roles()
        for i in range(0, batch_size):
            _norm = norm[i]
            _v = v_potential[i]
            _vrn = []
            _ref = situations[i]
            for pot in vrn_potential: _vrn.append(pot[i])
            for r in range(0, n_refs):
                v = _ref[0]
                pots = _v[v]
                for (pos, (s, idx, rid)) in enumerate(self.v_roles[v]):
                    pots = pots + _vrn[s][idx][_ref[1 + 2 * mr * r + 2 * pos + 1]]
                if pots.data[0] > _norm.data[0]:
                    print("inference error")
                    print(pots)
                    print(_norm)
                if i == 0 and r == 0:
                    loss = pots - _norm
                else:
                    loss = loss + pots - _norm
        return -loss / (batch_size * n_refs)

    def mil_loss(self, v_potential, vrn_potential, norm, situations, n_refs):
        # compute the mil losses... perhaps this should be a different method to facilitate parrelism?
        batch_size = v_potential.size()[0]
        mr = self.encoding.max_roles()
        for i in range(0, batch_size):
            _norm = norm[i]
            _v = v_potential[i]
            _vrn = []
            _ref = situations[i]
            for pot in vrn_potential: _vrn.append(pot[i])
            for r in range(0, n_refs):
                v = _ref[0]
                pots = _v[v]
                for (pos, (s, idx, rid)) in enumerate(self.v_roles[v.item()]):
                    #    print _vrn[s][idx][_ref[1 + 2*mr*r + 2*pos + 1]]
                    # _vrn[s][idx][
                    pots = pots + _vrn[s][idx][_ref[1 + 2 * mr * r + 2 * pos + 1]]
                if pots.item() > _norm.item():
                    print("inference error")
                    print(pots)
                    print(_norm)
                if r == 0:
                    _tot = pots - _norm
                else:
                    _tot = self.logsumexp_nx_ny_xy(_tot, pots - _norm)
            if i == 0:
                loss = _tot
            else:
                loss = loss + _tot
        return -loss / batch_size


def train_model(max_epoch, eval_frequency, train_loader, dev_loader, model,
                encoding, optimizer, save_dir, device_array, timing=False):
    model.train()

    time_all = time.time()

    pmodel = torch.nn.DataParallel(model, device_ids=device_array)
    top1 = ImSituTensorEvaluation(1, 3, encoding)
    top5 = ImSituTensorEvaluation(5, 3, encoding)
    loss_total = 0
    print_freq = 10
    total_steps = 0
    avg_scores = []

    for k in range(0, max_epoch):
        for i, (index, input, target) in enumerate(train_loader):
            total_steps += 1

            t0 = time.time()
            t1 = time.time()

            input_var = torch.autograd.Variable(input.cuda())
            target_var = torch.autograd.Variable(target.cuda())
            (_, v, vrn, norm, scores, predictions) = pmodel(input_var)
            (s_sorted, idx) = torch.sort(scores, 1, True)
            # print norm
            if timing: print("forward time = {}".format(time.time() - t1))
            optimizer.zero_grad()
            t1 = time.time()
            loss = model.mil_loss(v, vrn, norm, target, 3)
            if timing: print("loss time = {}".format(time.time() - t1))
            t1 = time.time()
            loss.backward()
            # print loss
            if timing: print("backward time = {}".format(time.time() - t1))
            optimizer.step()
            loss_total += loss.item()
            # score situation
            t2 = time.time()
            top1.add_point(target, predictions.data, idx.data)
            top5.add_point(target, predictions.data, idx.data)

            if timing: print("eval time = {}".format(time.time() - t2))
            if timing: print("batch time = {}".format(time.time() - t0))
            if total_steps % print_freq == 0:
                top1_a = top1.get_average_results()
                top5_a = top5.get_average_results()
                print("{},{},{}, {} , {}, loss = {:.2f}, avg loss = {:.2f}, batch time = {:.2f}".format(
                    total_steps - 1, k, i, format_dict(top1_a, "{:.2f}", "1-"), format_dict(top5_a, "{:.2f}", "5-"),
                    loss.item(), loss_total / ((total_steps - 1) % eval_frequency),
                    (time.time() - time_all) / ((total_steps - 1) % eval_frequency)))
            if total_steps % eval_frequency == 0:
                print("eval...")
                etime = time.time()
                (top1, top5) = eval_model(dev_loader, encoding, model)
                model.train()
                print("... done after {:.2f} s".format(time.time() - etime))
                top1_a = top1.get_average_results()
                top5_a = top5.get_average_results()

                avg_score = top1_a["verb"] + top1_a["value"] + top1_a["value-all"] + top5_a["verb"] + top5_a[
                    "value"] + top5_a["value-all"] + top5_a["value*"] + top5_a["value-all*"]
                avg_score /= 8

                print("Dev {} average :{:.2f} {} {}".format(total_steps - 1, avg_score * 100,
                                                            format_dict(top1_a, "{:.2f}", "1-"),
                                                            format_dict(top5_a, "{:.2f}", "5-")))

                avg_scores.append(avg_score)
                maxv = max(avg_scores)

                if maxv == avg_scores[-1]:
                    torch.save(model.state_dict(), save_dir + "/{0}.model".format(maxv))
                    print("new best model saved! {0}".format(maxv))

                top1 = ImSituTensorEvaluation(1, 3, encoding)
                top5 = ImSituTensorEvaluation(5, 3, encoding)
                loss_total = 0
                time_all = time.time()


def format_dict(d, s, p):
    rv = ""
    for (k, v) in d.items():
        if len(rv) > 0:
            rv += " , "
        rv += p + str(k) + ":" + s.format(v * 100)
    return rv


def eval_model(dataset_loader, encoding, model):
    model.eval()
    print("evaluating model...")
    top1 = ImSituTensorEvaluation(1, 3, encoding)
    top5 = ImSituTensorEvaluation(5, 3, encoding)

    mx = len(dataset_loader)
    for i, (index, input, target) in enumerate(dataset_loader):
        print("{}/{} batches\r".format(i + 1, mx)),
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target.cuda(), volatile=True)
        (scores, predictions) = model.forward_max(input_var)
        (s_sorted, idx) = torch.sort(scores, 1, True)
        top1.add_point(target, predictions.data, idx.data)
        top5.add_point(target, predictions.data, idx.data)

    print("\ndone.")
    return (top1, top5)
