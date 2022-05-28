## General imports
import os
import json
import time
import torch.nn
import argparse
import numpy as np
import torch.optim as optim

### DL2 Imports
from training.supervised.models import MLP
from training.supervised.oracles import DL2_Oracle

## Local imports
from constraints import *
from dataloader import MIMIC3
from utils import calculate_metrics


use_cuda = torch.cuda.is_available()

def parse_arguments():
    """
    Evaluate the arguments given to the program, and add all necessary default arguments.

    Returns
    -------
    dict
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Train NN with constraints.')
    parser = dl2.add_default_parser_args(parser)
    parser.add_argument('--batch-size', type=int, default=64, help='Number of samples in a batch.')
    parser.add_argument('--num-iters', type=int, default=50, help='Number of oracle iterations.')
    parser.add_argument('--num-epochs', type=int, default=300, help='Number of epochs to train for.')
    parser.add_argument('--l2', type=int, default=0.01, help='L2 regularizxation.')
    parser.add_argument('--pos-weight', type=int, default=3, help='Weight of positive examples.')
    parser.add_argument('--grid-search', default=False, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--dl2-weight', type=float, default=0.0, help='Weight of DL2 loss.')
    parser.add_argument('--delay', type=int, default=0, help='How many epochs to wait before training with constraints.')
    parser.add_argument('--constraint', type=str, required=True, help='the constraint to train with: LipschitzT(L), LipschitzG(eps, L), RobustnessT(eps1, eps2), RobustnessG(eps, delta), CSimiliarityT(), CSimilarityG(), LineSegmentG()')
    parser.add_argument('--print-freq', type=int, default=10, help='Print frequency.')
    parser.add_argument('--report-dir', type=str, required=True, help='Directory where results should be stored')
    parser.add_argument('--network-output', type=str, choices=['logits', 'prob', 'logprob'], default='logits', help='Wether to treat the output of the network as logits, probabilities or log(probabilities) in the constraints.')
    return parser.parse_args()


def oracle_train(args, x_batch, y_batch, oracle):
    """
    Perform general attack on the batches generating counterexamples.
    Then, evaluate the pre-specified constraints for the given oracle on the found counterexamples.
    
    Parameters
    ----------
    ...

    Returns
    -------
    dl2_batch_loss: tensor<float>
        Values of the loss function for current batch.
    constr_acc: tensor<float>
        
    """
    n_batch = int(x_batch.size()[0])
    x_batches, y_batches = [], []
    k = n_batch // oracle.constraint.n_tvars
    assert n_batch % oracle.constraint.n_tvars == 0, 'Batch size must be divisible by number of train variables!'
    
    for i in range(oracle.constraint.n_tvars):
        x_batches.append(x_batch[i:(i + k)])
        y_batches.append(y_batch[i:(i + k)])

    if oracle.constraint.n_gvars > 0:
        domains = oracle.constraint.get_domains(x_batches, y_batches)
        z_batches = oracle.general_attack(x_batches, y_batches, domains, num_restarts=1, num_iters=args.num_iters, args=args)
        _, dl2_batch_loss, constr_acc = oracle.evaluate(x_batches, y_batches, z_batches, args)
    else:
        _, dl2_batch_loss, constr_acc = oracle.evaluate(x_batches, y_batches, None, args)

    return (dl2_batch_loss, constr_acc)


def train(args, net, oracle, device, train_loader, optimizer, epoch):
    """
    Train the specified network.

    Parameters
    ----------
    ...

    Returns
    -------
    ...    
    """
    t1 = time.time()
    predictions, labels = [], []
    avg_ce_loss, avg_dl2_loss, avg_constr_acc, num_steps = 0, 0, 0, 0
    ce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight]))

    if args.verbose:
        print('Epoch ', epoch)

    for batch_idx, (data, target) in enumerate(train_loader):
        num_steps += 1
        x_batch, y_batch = data.to(device), target.to(device)
        
        x_output = net(x_batch)
        ce_batch_loss = ce_loss(x_output, y_batch)
        x_prob = torch.sigmoid(x_output)
        predictions.extend(x_prob.detach().numpy().flatten())
        labels.extend(y_batch.detach().numpy().flatten())
        
        net.eval()
        avg_ce_loss += ce_batch_loss.item()

        if oracle and args.dl2_weight >= 1e-7:
            (dl2_batch_loss, constr_acc) = oracle_train(args, x_batch, y_batch, oracle)
            net.train()
            optimizer.zero_grad()
            tot_batch_loss = args.dl2_weight * dl2_batch_loss + ce_batch_loss
            tot_batch_loss.backward()
            optimizer.step()
            avg_dl2_loss += dl2_batch_loss.item()
            avg_constr_acc += constr_acc.item()
        
            if args.verbose and batch_idx % args.print_freq == 0:
                print('[%d] CE loss: %.3lf, dl2 loss: %.3lf, constr acc: %.3lf' % (batch_idx, ce_batch_loss.item(), dl2_batch_loss.item(), constr_acc.item()))
        
        else:
            net.train()
            optimizer.zero_grad()
            ce_batch_loss.backward()
            optimizer.step()
            
            if args.verbose and batch_idx % args.print_freq == 0:
                print('[%d] CE loss: %.3lf' % (batch_idx, ce_batch_loss.item()))
    
    if args.verbose:
        print()

    t2 = time.time()
    t = t2 - t1
    metrics = calculate_metrics(labels, predictions, args.verbose)    
    metrics['epoch_time']  = t
    metrics['loss']        = avg_ce_loss / float(num_steps)
    metrics['constr_loss'] = avg_dl2_loss / float(num_steps)
    metrics['constr_acc']  = avg_constr_acc / float(num_steps)

    if args.verbose:
        print('[Train Set] Train acc: %.4f, CE loss: %.3lf, aucroc: %.4f, aucprc: %.4f\n' % (
                    metrics['acc'], metrics['loss'], metrics['auroc'], metrics['auprc']))

    return metrics




def test(args, model, oracle, device, test_loader):
    """
    Test the network.

    Parameters
    ----------
    ...

    Returns
    -------
    ...    
    """
    loss = torch.nn.BCEWithLogitsLoss(pos_weight==torch.tensor([args.pos_weight]))
    model.eval()
    predictions, labels = [], []
    avg_ce_loss, avg_dl2_loss, avg_constr_acc, num_steps = 0, 0, 0, 0
    
    for data, target in test_loader:
        num_steps += 1
        x_batch, y_batch = data.to(device), target.to(device)

        if oracle:
            (dl2_batch_loss, constr_acc) = oracle_train(args, x_batch, y_batch, oracle)
            avg_dl2_loss += dl2_batch_loss.item()
            avg_constr_acc += constr_acc.item()

        x_output = model(x_batch)
        avg_ce_loss += loss(x_output, y_batch).item()
        avg_dl2_loss += dl2_batch_loss.item()
        avg_constr_acc += constr_acc.item()

        x_prob = torch.sigmoid(x_output)
        predictions.extend(x_prob.detach().numpy().flatten())
        labels.extend(y_batch.detach().numpy().flatten())

    metrics = calculate_metrics(labels, predictions, args.verbose)
    metrics['loss']  = avg_ce_loss / float(num_steps)
    metrics['constr_loss'] = avg_dl2_loss / float(num_steps)
    metrics['constr_acc']  = avg_constr_acc / float(num_steps)

    if args.verbose:
        print('[Test Set] acc: %.4f, CE loss: %.3lf, aucroc: %.4f, aucprc: %.4f\n' % (
                    metrics['acc'], metrics['loss'], metrics['auroc'], metrics['auprc']))

    return metrics


def Mimic3(eps1, eps2):
    """
    Setup the Mimic3 constraint. 
    """
    return lambda model, use_cuda, network_output: Mimic3DatasetConstraint(model, eps1, eps2, use_cuda=use_cuda, network_output=network_output)


torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
args = parse_arguments()
model = MLP(714, 1, 1000, 3).to(device)
constraint = eval(args.constraint)(model, use_cuda, network_output=args.network_output)
oracle = DL2_Oracle(learning_rate=0.01, net=model, constraint=constraint, use_cuda=use_cuda)
mimic_train = MIMIC3(mode='train')

if args.grid_search:
    mimic_test   = MIMIC3(mode='val')
    l2_weights = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    pos_weights = [7, 6, 5, 4, 3, 2, 1]
else:
    mimic_test   = MIMIC3(mode='test')
    l2_weights = [args.l2]
    pos_weights = [args.pos_weight]

train_loader = torch.utils.data.DataLoader(mimic_train, shuffle=True, batch_size=args.batch_size, **kwargs)
test_loader   = torch.utils.data.DataLoader(mimic_test, shuffle=True, batch_size=args.batch_size, **kwargs)

if not os.path.exists(args.report_dir):
    os.makedirs(args.report_dir)

for (l2, pos_weight) in [(l2, pos) for l2 in l2_weights for pos in pos_weights]:
    tstamp = int(time.time())
    args.pos_weight = pos_weight
    exptype = 'baseline' if args.dl2_weight < 1e-7 else 'dl2'
    report_file = os.path.join(args.report_dir, 'report_%s_%s_%d.json' % (constraint.name, exptype, tstamp))

    data_dict = {
        'num_epochs': args.num_epochs,
        'l2_weight': l2,
        'pos_weight': pos_weight,
        'dl2_weight': args.dl2_weight,
        'name': constraint.name,
        'constraint_txt': args.constraint,
        'constraint_params': constraint.params(),
        'num_iters': args.num_iters,
        'train_loss': [],
        'train_constr_loss': [],
        'train_acc': [],
        'train_constr_acc': [],
        'train_aucroc': [],
        'train_aucprc': [],
        'train_conf': [],
        'epoch_time': [],
        'loss': [],
        'constr_loss': [],
        'acc': [],
        'constr_acc': [],
        'aucroc': [],
        'aucprc': [],
        'conf': []
    }

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=l2)

    for epoch in range(1, args.num_epochs + 1):
        train_metrics = train(args, model, oracle, device, train_loader, optimizer, epoch)
        data_dict['train_acc'].append(train_metrics['acc'].item())
        data_dict['train_constr_acc'].append(train_metrics['constr_acc'])
        data_dict['train_loss'].append(train_metrics['loss'])
        data_dict['train_constr_loss'].append(train_metrics['constr_loss'])
        data_dict['train_aucroc'].append(train_metrics['auroc'].item())
        data_dict['train_aucprc'].append(train_metrics['auprc'].item())
        data_dict['train_conf'].append(train_metrics['cf'])
        data_dict['epoch_time'].append(train_metrics['epoch_time'])

        test_metrics = test(args, model, oracle, device, test_loader)
        data_dict['acc'].append(test_metrics['acc'].item())
        data_dict['constr_acc'].append(test_metrics['constr_acc'])
        data_dict['loss'].append(test_metrics['loss'])
        data_dict['constr_loss'].append(test_metrics['constr_loss'])
        data_dict['aucroc'].append(test_metrics['auroc'].item())
        data_dict['aucprc'].append(test_metrics['auprc'].item())
        data_dict['conf'].append(test_metrics['cf'])
        print('Epoch Time [s]: %.4f\n' % (train_metrics['epoch_time']))

    with open(report_file, 'w') as fou:
        json.dump(data_dict, fou, indent=4)