"""
Dynamic Routing Between Capsules
https://arxiv.org/abs/1710.09829

PyTorch implementation inspired and based on Kenta Iwasaki @ Gram.AI.
"""
import sys
import os
import shutil
import time
import argparse
sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import subprocess

import logging
logging.captureWarnings(True)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Dynamic-Routing-between-capsules implementation')
parser.add_argument('-act', type=str, default='squash', metavar='A',
                    help='activation-function (default: squash, others: sig, relu, lrelu))')
parser.add_argument('-loss', type=str, default='margin', metavar='L',
                    help='loss function (default: margin, others: ce, mse)')
parser.add_argument('-name', type=str, metavar='N',
                    help='If no name then it gets the name of the parameters')
parser.add_argument('-a1', type=float, default=2, metavar='a1',
                    help='squash (default: 2)')
parser.add_argument('-a2', type=float, default=1, metavar='a2',
                    help='squash (default: 1)')
parser.add_argument('-k1', type=float, metavar='k1',
                    help='sig: ex:1')
parser.add_argument('-k2', type=float, metavar='k2',
                    help='sig: ex:0')
parser.add_argument('-k3', type=float, metavar='k3',
                    help='sig: ex:1')
parser.add_argument('-t1', type=float, metavar='t1',
                    help='tanh: ex:1')
parser.add_argument('-d', type=str, default='Omniglot', metavar='dataset_used',
                    help='Dataset used. Possible datasets: Omniglot, MNIST')
parser.add_argument('-c', type=int, metavar='num_classes',
                    help='If dataset Omniglot and MNIST one can define number of classes')
parser.add_argument('-e', type=int, default=100, metavar='NUM_EPOCHS',
                    help='The number of times to train through the whole dataset')
parser.add_argument('-b', type=int, default=100, metavar='BATCH_SIZE',
                    help='The batch-size')
parser.add_argument('-smax', type=int, metavar='SAMPLE_PER_CLASS',
                    help='Limit the training samples per class')

global args
args = vars(parser.parse_args())
BATCH_SIZE = args.get('b') #100
if args.get('c') is not None:
    NUM_CLASSES = args.get('c')
else:
    NUM_CLASSES = 1623 if args.get('d') == 'Omniglot' else 10
NUM_EPOCHS = args.get('e')
NUM_ROUTING_ITERATIONS = 3



def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


def augmentation(x, max_shift=2):
    _, _, height, width = x.size()

    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)

    shifted_image = torch.zeros(*x.size())
    shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_route_nodes, in_channels, out_channels, kernel_size=None, stride=None,
                 num_iterations=NUM_ROUTING_ITERATIONS, **kwargs):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        self.squash = Squash(**kwargs)

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(*priors.size())).cuda()
            for i in range(self.num_iterations):
                probs = softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs

    def define_activation_function(self, code_word, **kwargs):
        if code_word == 'squash':
            return Squash(code_word, **kwargs)
        elif code_word == 'sig':
            return Squash(code_word, **kwargs) #NOT CORRECT
        elif code_word == 'tanh':
            return self.tanh
        elif code_word == 'relu':
            return self.relu
        else:
            raise Exception(' Activation function not valid. The code_word used was : {}'.format(code_word))

    def squash(self, tensor, a1=2, a2=1, dim=-1):
        l2 = torch.sqrt((tensor ** 2).sum(dim=dim, keepdim=True))
        scale = self.squash_function(l2, a1, a2)
        return scale * tensor / l2

    def sigmoid(self, tensor, s1=3, s2=3, s3=3, a1=2, a2=1, dim=-1):
        l2 = torch.sqrt((tensor ** 2).sum(dim=dim, keepdim=True))
        scale = 1 / (1 + s1*torch.exp(s2-s3*l2)) * self.squash_function(l2, a1, a2)
        return scale * tensor / l2

    def tanh(self, tensor, t1=3, t2=2, a1=2, a2=1, dim=-1):
        l2 = torch.sqrt((tensor ** 2).sum(dim=dim, keepdim=True))
        scale = (0.5 + 0.5*torch.tanh(t1*l2 - t2)) * self.squash_function(l2, a1, a2)
        return scale * tensor / l2

    def squash_function(self, l2, a1=1, a2=0.2):
        return abs(l2)**a1 / (abs(l2)**a1 + a1)


class Squash(torch.nn.Module):

    def __init__(self, **kwargs):
        super(Squash, self).__init__()
        if 'act' in kwargs:
            act = kwargs.pop('act', False)
        else:
            raise Exception('To run function we need to define act (is parameter in --act)')

        if act == 'squash':
            self.scaling_function = Squash_default(**kwargs)
        elif act == 'sig':
            self.scaling_function = Sigmoid_scaling(**kwargs)
        elif act == 'tanh':
            return self.tanh
        else:
            raise Exception(' Activation function not valid. The code_word used was : {}'.format(code_word))


    def forward(self, tensor):
        l2 = torch.sqrt((tensor ** 2).sum(dim=dim, keepdim=True))
        scale = self.scaling_function(l2)
        return scale * tensor / l2


class Squash_default(torch.nn.Module):
    """
    increase a1 to get a steeper curve inplace
    Increase a2 to get a flattening elongated effect on the curve of the activationfunction, widening the area of effect 
    """

    def __init__(self, **kwargs):
        super(Squash_default, self).__init__()
        self.a1 = kwargs.pop('a1', False)
        self.a2 = kwargs.pop('a2', False)

    def forward(self, l2):
        return abs(l2)**self.a1 / (abs(l2)**self.a1 + self.a2)


class Sigmoid_scaling(torch.nn.Module):
    """
    Increase s1 to ...
    Increase s2 to
    Increase s3 to
    """

    def __init__(self, **kwargs):
        super(Sigmoid_scaling, self).__init__()
        self.__dict__.update(kwargs)

    def forward(self, l2):
        return  (1 / (1 + self.k1*torch.exp(self.k2-self.k3*l2))) #* (abs(l2)**self.a1 / (abs(l2)**self.a1 + self.a2))



class CapsuleNet_mnist(nn.Module):
    def __init__(self, **kwargs):
        super(CapsuleNet_mnist, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2, **kwargs)
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes= 32 * 6 * 6, in_channels=8,
                                           out_channels=16, **kwargs)

        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()   # hmmm
        )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)

        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data)

        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))

        return classes, reconstructions


class CapsuleNet_omniglot(nn.Module):
    def __init__(self, **kwargs):
        super(CapsuleNet_omniglot, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=9, stride=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=11, stride=2)

        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2, **kwargs)
        self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes= 32 * 6 * 6, in_channels=8,
                                           out_channels=16, **kwargs)

        self.decoder = nn.Sequential(
            nn.Linear(16 * NUM_CLASSES, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 11025),
            nn.Sigmoid()   # hmmm
        )

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze().transpose(0, 1)

        # fix for last_batch beeing only one sample
        if len(list(x.size())) < 3:
            x = x.transpose(0,1).unsqueeze_(0)
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)
        if y is None:
            # In all batches, get the most active capsule.
            _, max_length_indices = classes.max(dim=1)
            y = Variable(torch.eye(NUM_CLASSES)).cuda().index_select(dim=0, index=max_length_indices.data)

        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))

        return classes, reconstructions


class CapsuleLoss(nn.Module):
    def __init__(self, loss_func='margin'):
        super(CapsuleLoss, self).__init__()
        self.reconstruction_loss = nn.MSELoss(size_average=True)
        self.loss_function = self.define_loss_function(loss_func)

    def forward(self, images, labels, classes, reconstructions):

        assert torch.numel(images) == torch.numel(reconstructions)
        images = images.view(reconstructions.size()[0], -1)
        reconstruction_loss = self.reconstruction_loss(reconstructions, images)

        return self.loss_function(images, labels, classes, reconstruction_loss)

    def define_loss_function(self, code_word):
        if code_word == 'margin':
            return self.margin_loss
        elif code_word == 'ce':
            return self.crossentropy
        elif code_word == 'mse':
            return self.mse
        else:
            raise Exception(' Loss function not valid. The code_word used was : {}'.format(code_word))

    def margin_loss(self, images, labels, classes, reconstruction_loss):
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.mean()

        return (margin_loss + 0.0005 * reconstruction_loss) / images.size(0)

    def crossentropy(self, images, labels, classes, reconstruction_loss):
        # Coeff scales the loss to the approx. the same magnitude as orig. -> reconstruction effect is approx. the same
        s_coeff = 20.85
        _, labels = labels.max(dim=1)
        return (s_coeff*nn.functional.cross_entropy(classes, labels) + 0.0005 * reconstruction_loss) / images.size(0)

    def mse(self, images, labels, classes, reconstruction_loss):
        s_coeff = 679.8
        return (s_coeff*nn.functional.mse_loss(classes, labels) + 0.0005 * reconstruction_loss) / images.size(0)


def is_valid_args(**kwargs):
    try:
        if kwargs['act'] == 'squash': 
                needed_params = ('a1', 'a2')
                for key in needed_params:
                    temp = kwargs[key]
        elif kwargs['act'] == 'sig':
                needed_params = ('k1', 'k2', 'k3')
                for key in needed_params:
                    temp = kwargs[key]
        else:
            raise Exception(' Activation function not valid. The code_word used was : {}'.format(kwargs['act']))
    except:
        raise Exception('Arguments for {}-function not valid. Need numeric variables for {l}'.format(kwargs['act'], l=needed_params))

#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    from torch.autograd import Variable
    from torch.optim import Adam
    
    import torchnet as tnt
    from torchnet.engine import Engine
    from torchnet.logger import VisdomPlotLogger, VisdomLogger
    
    from torchvision.utils import make_grid
    from torchvision.datasets import MNIST, ImageFolder
    from tqdm import tqdm
    import torchvision
    from torchvision.datasets.utils import makedir_exist_ok

    from create_symbols_dataset import create_symbols_dataset

    #global args
    #args = vars(parser.parse_args())
    print(args)
    args = {k: v for k, v in args.items() if v is not None} # removes unused arguments
    is_valid_args(**args)
    if 'name' in args:
        visdom_env = '-'.join([args['act'][:3], args['loss'][:3], args.pop('name', False)])
    else:
        visdom_env = '-'.join(['%s' % value[:3] if type(value) is str else '%s:%s' % (key, int(value)) for (key, value) in args.items()])
    arg_loss = args.pop('loss', False)
    if args.get('d') == "Omniglot":
        model = CapsuleNet_omniglot(**args)
    else:
        model = CapsuleNet_mnist(**args)
    # model.load_state_dict(torch.load('epochs/epoch_327.pt'))
    if torch.cuda.is_available():
        model.cuda()

    print("# parameters:", sum(param.numel() for param in model.parameters()))

    optimizer = Adam(model.parameters())

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(NUM_CLASSES, normalized=True)
    AP_meter = tnt.meter.APMeter()
    layoutoptions = {'plotly': {'legend': dict(x=0.8, y=0.5, traceorder='normal', font=dict(family='sans-serif',size=9,color='#000'), bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)', borderwidth=2)}}

    train_loss_logger = VisdomPlotLogger('line', env=visdom_env, opts={'title': 'Train Loss', 
                                                                                'xlabel': 'Epochs', 
                                                                                'ylabel': 'Train loss', 
                                                                                'legend': [visdom_env],
                                                                                'layoutopts': layoutoptions})
    train_error_logger = VisdomPlotLogger('line', env=visdom_env, opts={'title': 'Train Accuracy', 
                                                                                'xlabel': 'Epochs', 
                                                                                'ylabel': 'Train accuracy', 
                                                                                'legend': [visdom_env],
                                                                                'layoutopts': layoutoptions})
    test_loss_logger = VisdomPlotLogger('line', env=visdom_env, opts={'title': 'Test Loss', 
                                                                                'xlabel': 'Epochs', 
                                                                                'ylabel': 'Test loss', 
                                                                                'legend': [visdom_env],
                                                                                'layoutopts': layoutoptions})
    test_accuracy_logger = VisdomPlotLogger('line', env=visdom_env, opts={'title': 'Test Accuracy', 
                                                                                'xlabel': 'Epochs', 
                                                                                'ylabel': 'Test accuracy', 
                                                                                'legend': [visdom_env],
                                                                                'layoutopts': layoutoptions})
    #AP_legend = ["{:04d}".format(c) for c in range(NUM_CLASSES)]
    average_precision_logger = VisdomPlotLogger('line', env=visdom_env, opts={'title': 'Average Precision per class',
                                                                                'xlabel': 'Class',
                                                                                'ylabel': 'Average precision',
                                                                                'layoutopts': layoutoptions})
    confusion_logger = VisdomLogger('heatmap', env=visdom_env, opts={'title': 'Confusion matrix', 'columnnames': list(range(NUM_CLASSES)), 'rownames': list(range(NUM_CLASSES))})
    ground_truth_logger = VisdomLogger('image', env=visdom_env, opts={'title': 'Ground Truth'})
    reconstruction_logger = VisdomLogger('image', env=visdom_env, opts={'title': 'Reconstruction ' + visdom_env})
    reconstruction_error_logger = VisdomLogger('image', env=visdom_env, opts={'title': 'Reconstruction Error ' + visdom_env})
    reconstruction_loss_logger = VisdomPlotLogger('line', env=visdom_env, opts={'title': 'Reconstruction Loss', 
                                                                                'xlabel': 'Epochs', 
                                                                                'ylabel': 'Reconstruction Loss', 
                                                                                'legend': [visdom_env],
                                                                                'layoutopts': layoutoptions})

    capsule_loss = CapsuleLoss(loss_func=arg_loss)


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

    class Omniglot_dataset(torchvision.datasets.vision.VisionDataset):
        training_file = 'omniglot_training.pt'
        test_file = 'omniglot_test.pt'

        @property
        def train_labels(self):
            return self.targets

        @property
        def test_labels(self):
            return self.targets

        @property
        def train_data(self):
            return self.data

        @property
        def test_data(self):
            return self.data

        def __init__(self, root, train=True, transform=None, download=False):
            self.root = root
            self.transform = transform
            self.train = train
            self.raw_training_folder = 'symbol_dataset/training_set'
            self.raw_testing_folder = 'symbol_dataset/testing_set'

            if download:
                if not os.path.exists(os.path.join('omniglot_raw_data')):
                    subprocess.call("./download_omniglot.sh", shell=True, executable='/bin/bash')
                if not os.path.exists('./symbol_dataset'):
                    print("importing symbols")
                    # Override this function as selection of num_classes is selected later
                    create_symbols_dataset(num_classes=1623)
                self.process()

            if not self._check_exists():
                raise RuntimeError('Dataset not found.' +
                                   ' You can use download=True to download it')

            if self.train:
                data_file = self.training_file
            else:
                data_file = self.test_file
            self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

        def __getitem__(self, index):
            print("get_item__________________________")
            return 0

        def __len__(self):
            return len(self.data)

        @property
        def processed_folder(self):
            return os.path.join(self.root, self.__class__.__name__, 'processed')

        def _check_exists(self):
            return (os.path.exists(os.path.join(self.processed_folder,
                                                self.training_file)) and
                    os.path.exists(os.path.join(self.processed_folder,
                                                self.test_file)))

        def process(self):

            if self._check_exists():
                return

            makedir_exist_ok(self.processed_folder)

            # process and save as torch files
            print('Processing...')

            training_set = ImageFolder(root=self.raw_training_folder, transform=None)
            testing_set = ImageFolder(root=self.raw_testing_folder, transform=None)
            for i, dataset in enumerate([training_set, testing_set]):
                data_orig = getattr(dataset, 'samples')
                data_list = []
                print("Unpacking images into tensor")
                for j, tuple in enumerate(data_orig):
                    pil = dataset.loader(tuple[0])
                    gray_pil = transforms.functional.to_grayscale(pil, num_output_channels=1)
                    tensor_img = transforms.functional.to_tensor(gray_pil)
                    data_list.append(tensor_img)
                    if j % 5000 == 0:
                        print("{} images processed out of {}".format(j, len(data_orig)))
                print("Finished: {} images processed out of {}".format(j + 1, len(data_orig)))
                data = torch.cat(data_list, out=torch.Tensor(len(data_orig), 105, 105))

                labels = getattr(dataset, 'targets')
                labels = torch.LongTensor(labels)

                if i == 0 : # training_set
                    with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
                        torch.save((data, labels), f)
                else:
                    with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
                        torch.save((data, labels), f)

            print('Done!')

        def extra_repr(self):
            return "Split: {}".format("Train" if self.train is True else "Test")


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


    def limit_dataset_by_max_samples_per_class(data, labels, sample_per_class, num_classes):
        counter_by_class_list = [0]*num_classes
        new_data = []
        new_labels = []

        for i in range(labels.shape[0]):
            try :
                if counter_by_class_list[labels[i]] < sample_per_class:
                    new_data.append(data[i].unsqueeze_(0))
                    new_labels.append(labels[i].unsqueeze_(0))
                    counter_by_class_list[labels[i]] += 1
                elif sum(counter_by_class_list) == num_classes*sample_per_class:
                    break
            except IndexError:
                pass
        if sum(counter_by_class_list) < num_classes*sample_per_class:
            print("WARNING: Less samples per class than defined in parameters.")
            print("counter_by_class_list: ", counter_by_class_list)
        return torch.cat(new_data, dim=0), torch.cat(new_labels, dim=0)


    def get_iterator(mode, dataset_used='Omniglot'):
        if dataset_used == 'Omniglot':
            dataset = Omniglot_dataset(root='./data', download=True, train=mode)
        else:
            dataset = MNIST(root='./data', download=True, train=mode)
        data = getattr(dataset, 'train_data' if mode else 'test_data')
        labels = getattr(dataset, 'train_labels' if mode else 'test_labels')

        if mode:
            data, labels = limit_dataset_by_max_samples_per_class(data, labels,
                                args.get('smax'), NUM_CLASSES)
            print('(Mode: %s) Number of samples in dataset: %d' % (
                    mode, data.shape[0]))
        tensor_dataset = tnt.dataset.TensorDataset([data, labels])
        """
        print("-----------------------------------------")
        print(dataset)
        print(data[0][14])
        print("- - - - - - - - - - - - - - - - - - - - -")
        print("data.size(): ", data.size())
        print('type(data): ', type(data))
        print('type(data[0][0][0]): ', type(data[0][0][0]))
        print('data[0][0][0]: ', data[0][0][0])
        #print("data.mean(): ", data.mean())
        print("data.min(): ", data.min())
        print("data.max(): ", data.max())
        print("- - - - - - - - - - - - - - - - - - - - -")
        print("labels.size(): ", labels.size())
        print("type(labels): ", type(labels))
        print("type(labels[0]): ", type(labels[0]))
        print("labels[0]: ", labels[0])
        print("labels.min(): ", labels.min())
        print("labels.max(): ", labels.max())
        print("-----------------------------------------")
        """
        if not mode and dataset_used == 'MNIST':
            return tensor_dataset.parallel(batch_size=50, num_workers=4, shuffle=mode)
        else:
            return tensor_dataset.parallel(batch_size=BATCH_SIZE, num_workers=4, shuffle=mode)

    def processor_omniglot(sample):
        data, labels, training = sample
        data = augmentation(data.unsqueeze(1).float())
        labels = torch.LongTensor(labels)
        labels = torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)

        data = Variable(data).cuda()
        labels = Variable(labels).cuda()

        if training:
            classes, reconstructions = model(data, labels)
        else:
            classes, reconstructions = model(data)

        loss = capsule_loss(data, labels, classes, reconstructions)

        return loss, classes

    def processor_mnist(sample):
        data, labels, training = sample

        data = augmentation(data.unsqueeze(1).float() / 255.0)
        labels = torch.LongTensor(labels)

        labels = torch.eye(NUM_CLASSES).index_select(dim=0, index=labels)

        data = Variable(data).cuda()
        labels = Variable(labels).cuda()

        if training:
            classes, reconstructions = model(data, labels)
        else:
            classes, reconstructions = model(data)

        loss = capsule_loss(data, labels, classes, reconstructions)

        return loss, classes


    def reset_meters():
        meter_accuracy.reset()
        meter_loss.reset()
        confusion_meter.reset()
        AP_meter.reset()


    def on_sample(state):
        state['sample'].append(state['train'])


    def on_forward(state):
        meter_accuracy.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        confusion_meter.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        AP_meter.add(state['output'].data,
                torch.nn.functional.one_hot(torch.LongTensor(state['sample'][1]), num_classes=NUM_CLASSES))
        meter_loss.add(state['loss'].item())


    def on_start_epoch(state):
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])


    def on_end_epoch(state):
        dataset_used = args.get('d')
        print('[Epoch %d] Training Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        train_loss_logger.log(state['epoch'], meter_loss.value()[0])
        train_error_logger.log(state['epoch'], meter_accuracy.value()[0])

        reset_meters()
        if dataset_used == 'Omniglot':
            engine.test(processor_omniglot, get_iterator(False, dataset_used=dataset_used))
        else:
            engine.test(processor_mnist, get_iterator(False, dataset_used=dataset_used))
        test_loss_logger.log(state['epoch'], meter_loss.value()[0])
        test_accuracy_logger.log(state['epoch'], meter_accuracy.value()[0])
        confusion_logger.log(confusion_meter.value())
        for index, value in enumerate(AP_meter.value()):
            average_precision_logger.log(state['epoch'], value, name=str(index))

        print('[Epoch %d] Testing Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], meter_loss.value()[0], meter_accuracy.value()[0]))

        torch.save(model.state_dict(), 'epochs/epoch_%d.pt' % state['epoch'])

        # Reconstruction visualization.
        if dataset_used == "Omniglot":
            test_sample = next(iter(get_iterator(False, dataset_used)))
            ground_truth = test_sample[0].unsqueeze(1).float()
        else:
            test_sample = next(iter(get_iterator(False, dataset_used)))
            ground_truth = (test_sample[0].unsqueeze(1).float() / 255.0)

        _, reconstructions = model(Variable(ground_truth).cuda())
        reconstruction = reconstructions.cpu().view_as(ground_truth).data

        ground_truth_logger.log(
            make_grid(ground_truth, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())
        reconstruction_logger.log(
            make_grid(reconstruction, nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())

        reconstruction_error_logger.log(make_grid(abs(reconstruction - ground_truth), nrow=int(BATCH_SIZE ** 0.5), normalize=True, range=(0, 1)).numpy())

        #assert torch.numel(images) == torch.numel(reconstructions)
        #images = images.view(reconstructions.size()[0], -1)

        reconstruction_loss_logger.log(state['epoch'],
                0.0005*nn.MSELoss(size_average=True)(reconstruction, ground_truth)) # log reconstruction loss on unseen photos

    # def on_start(state):
    #     state['epoch'] = 327
    #
    # engine.hooks['on_start'] = on_start
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    if args.get('d') == "Omniglot":
        engine.train(processor_omniglot, get_iterator(True, dataset_used='Omniglot'), maxepoch=NUM_EPOCHS, optimizer=optimizer)
    else:
        engine.train(processor_mnist, get_iterator(True, dataset_used='MNIST'), maxepoch=NUM_EPOCHS, optimizer=optimizer)


    # delete processed files
    if os.path.exists('data/Omniglot_dataset/processed/omniglot_training.pt'):
        os.remove('data/Omniglot_dataset/processed/omniglot_training.pt')
    if os.path.exists('data/Omniglot_dataset/processed/omniglot_test.pt'):
        os.remove('data/Omniglot_dataset/processed/omniglot_test.pt')
    if os.path.exists('symbol_dataset/'):
        shutil.rmtree('symbol_dataset/')
