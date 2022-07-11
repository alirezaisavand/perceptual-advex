from typing import Dict, List
import torch
import csv
import argparse
import torch.nn as nn


def normalize(X):
    cifar10_mean = (0.4914, 0.4822, 0.4465)  # equals np.mean(train_set.train_data, axis=(0,1,2))/255
    cifar10_std = (0.2471, 0.2435, 0.2616)  # equals np.std(train_set.train_data, axis=(0,1,2))/255

    mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
    std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()

    return (X - mu) / std


class Wrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        return self.model(normalize(x))


from perceptual_advex.utilities import add_dataset_model_arguments, \
    get_dataset_model
import adversarial_loss_final
from perceptual_advex.attacks import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Adversarial training evaluation')

    add_dataset_model_arguments(parser, include_checkpoint=True)
    parser.add_argument('attacks', metavar='attack', type=str, nargs='+',
                        help='attack names')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='number of examples/minibatch')
    parser.add_argument('--parallel', type=int, default=1,
                        help='number of GPUs to train on')
    parser.add_argument('--num_batches', type=int, required=False,
                        help='number of batches (default entire dataset)')
    parser.add_argument('--per_example', action='store_true', default=False,
                        help='output per-example accuracy')
    parser.add_argument('--output', type=str, help='output CSV')

    args = parser.parse_args()

    dataset = get_dataset_model(args)

    pure_model = adversarial_loss_final.PreActResNet18()
    model = Wrapper(pure_model)

    pure_model = nn.DataParallel(pure_model).cuda()

    check_point = torch.load('model_best.pth')
    pure_model.load_state_dict(check_point['state_dict'])

    _, val_loader = dataset.make_loaders(1, args.batch_size, only_val=True)

    pure_model.eval()

    if torch.cuda.is_available():
        pure_model.cuda()

    # changed here
    # could operate wrong on those which take dataset as input because we normalize input
    attack_names: List[str] = args.attacks
    attacks = [eval(attack_name) for attack_name in attack_names]
    # attacks = []
    # attack_names = []
    #
    # attack = LagrangePerceptualAttack(
    # model,
    # num_iterations=10,
    # The LPIPS distance bound on the adversarial examples.
    # bound=0.5,
    # The model to use for calculate LPIPS; here we use AlexNet.
    # You can also use 'self' to perform a self-bounded attack.
    # lpips_model='alexnet_cifar',
    # )
    # attacks.append(attack)
    # attack_names.append('Lagrange Perceptual')
    ###

    # Parallelize
    if torch.cuda.is_available():
        device_ids = list(range(args.parallel))
        pure_model = nn.DataParallel(pure_model, device_ids)
        attacks = [nn.DataParallel(attack, device_ids) for attack in attacks]

    batches_correct: Dict[str, List[torch.Tensor]] = \
        {attack_name: [] for attack_name in attack_names}
    print('number of batches:', len(val_loader), sep=' ')

    for batch_index, (inputs, labels) in enumerate(val_loader):
        print(f'BATCH {batch_index:05d}')

        if (
                args.num_batches is not None and
                batch_index >= args.num_batches
        ):
            break

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()

        for attack_name, attack in zip(attack_names, attacks):
            # normalized_inputs = normalize(inputs)
            adv_inputs = attack(inputs, labels)
            with torch.no_grad():
                # todo check when we should normalize input that doesn't affect the attacks
                adv_logits = wrapper(adv_inputs)
            batch_correct = (adv_logits.argmax(1) == labels).detach()

            batch_accuracy = batch_correct.float().mean().item()
            print(f'ATTACK {attack_name}',
                  f'accuracy = {batch_accuracy * 100:.1f}',
                  sep='\t')
            batches_correct[attack_name].append(batch_correct)

    print('OVERALL')
    accuracies = []
    attacks_correct: Dict[str, torch.Tensor] = {}
    for attack_name in attack_names:
        attacks_correct[attack_name] = torch.cat(batches_correct[attack_name])
        accuracy = attacks_correct[attack_name].float().mean().item()
        print(f'ATTACK {attack_name}',
              f'accuracy = {accuracy * 100:.1f}',
              sep='\t')
        accuracies.append(accuracy)

    with open(args.output, 'w') as out_file:
        out_csv = csv.writer(out_file)
        out_csv.writerow(attack_names)
        if args.per_example:
            for example_correct in zip(*[
                attacks_correct[attack_name] for attack_name in attack_names
            ]):
                out_csv.writerow(
                    [int(attack_correct.item()) for attack_correct
                     in example_correct])
        out_csv.writerow(accuracies)
