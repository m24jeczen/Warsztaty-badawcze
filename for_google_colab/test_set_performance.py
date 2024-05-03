import numpy as np
import sklearn.metrics as metrics
import torch
import torch.utils.data
import pandas as pd

from dataset import Dataset
from model import instantiate_sparseconvmil
import argparse

def _define_args():
    parser = argparse.ArgumentParser(description='SparseConvMIL: Sparse Convolutional Context-Aware Multiple Instance '
                                                 'Learning for Whole Slide Image Classification')

    parser.add_argument('--slide-parent-folder', type=str, default='Warsztaty-badawcze/Data/Selected_Test_5', metavar='PATH',
                        help='path of parent folder containing preprocessed slides data')
    parser.add_argument('--slide-labels-filepath', type=str, default='Warsztaty-badawcze/Data/Selected_Test_5/labels.csv',
                        metavar='PATH',
                        help='path of CSV-file containing slide labels')

    parser.add_argument('--epochs', type=int, default=300, metavar='N', help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate')
    parser.add_argument('--reg', type=float, default=1e-6, metavar='R', help='weight decay')

    # Model parameters
    parser.add_argument('--tile-embedder', type=str, default='resnet18', metavar='MODEL', nargs='*',
                        help='type of resnet architecture for the tile embedder')
    parser.add_argument('--tile-embedder-pretrained', action='store_true', default=False,
                        help='use Imagenet-pretrained tile embedder architecture')
    parser.add_argument('--sparse-conv-n-channels-conv1', type=int, default=36,
                        help='number of channels of first convolution of the sparse-input CNN pooling')
    parser.add_argument('--sparse-conv-n-channels-conv2', type=int, default=36,
                        help='number of channels of first convolution of the sparse-input CNN pooling')
    parser.add_argument('--sparse-map-downsample', type=int, default=10, help='downsampling factor of the sparse map')
    parser.add_argument('--wsi-embedding-classifier-n-inner-neurons', type=int, default=32,
                        help='number of inner neurons for the WSI embedding classifier')

    # Dataset parameters
    parser.add_argument('--batch-size', type=int, default=2, metavar='SIZE',
                        help='number of slides sampled per iteration')
    parser.add_argument('--n-tiles-per-wsi', type=int, default=5, metavar='SIZE',
                        help='number of tiles to be sampled per WSI')

    # Miscellaneous parameters
    parser.add_argument('--j', type=int, default=2, metavar='N_WORKERS', help='number of workers for dataloader')

    args = parser.parse_args()
    hyper_parameters = {
        'slide_parent_folder': args.slide_parent_folder,
        'slide_labels_filepath': args.slide_labels_filepath,
        'epochs': args.epochs,
        'lr': args.lr,
        'reg': args.reg,
        'tile_embedder': args.tile_embedder,
        'tile_embedder_pretrained': args.tile_embedder_pretrained,
        'sparse_conv_n_channels_conv1': args.sparse_conv_n_channels_conv1,
        'sparse_conv_n_channels_conv2': args.sparse_conv_n_channels_conv2,
        'sparse_map_downsample': args.sparse_map_downsample,
        'wsi_embedding_classifier_n_inner_neurons': args.wsi_embedding_classifier_n_inner_neurons,
        'batch_size': args.batch_size,
        'n_tiles_per_wsi': args.n_tiles_per_wsi,
        'j': args.j,
    }

    return hyper_parameters

def get_dataloader(dataset, batch_size, shuffle, num_workers):
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle, num_workers=num_workers)

def main_test(hyper_parameters):
    # Loads dataset and dataloader for test data
    print('Loading test data')
    test_dataset = Dataset(hyper_parameters['slide_parent_folder'], hyper_parameters['slide_labels_filepath'],
                           hyper_parameters['n_tiles_per_wsi'])
    n_classes = test_dataset.n_classes
    test_dataloader = get_dataloader(test_dataset, hyper_parameters['batch_size'], False, hyper_parameters['j'])
    print('  done')

    # Loads trained MIL model
    print('Loading trained SparseConvMIL model')
    trained_model = instantiate_sparseconvmil(hyper_parameters['tile_embedder'],
                                              hyper_parameters['tile_embedder_pretrained'],
                                              hyper_parameters['sparse_conv_n_channels_conv1'],
                                              hyper_parameters['sparse_conv_n_channels_conv2'],
                                              3, 3, hyper_parameters['sparse_map_downsample'],
                                              hyper_parameters['wsi_embedding_classifier_n_inner_neurons'],
                                              n_classes)
    trained_model = torch.nn.DataParallel(trained_model)
    trained_model.load_state_dict(torch.load('trained_model.pth'))
    trained_model.eval()
    print('  done')

    # Lists to store true and predicted labels
    true_labels = []
    predicted_labels = []

    # Perform predictions on test data
    print('Performing predictions on test data')
    with torch.no_grad():
        for batch in test_dataloader:
            data, _, _, labels = batch
            data = data.cuda()
            predictions = trained_model(data)
            predicted_classes = torch.argmax(predictions, dim=1)
            predicted_classes = predicted_classes.cpu().numpy()
            labels = labels.numpy()

            true_labels.extend(labels)
            predicted_labels.extend(predicted_classes)

    # Calculate evaluation metrics
    accuracy = metrics.accuracy_score(true_labels, predicted_labels)
    precision = metrics.precision_score(true_labels, predicted_labels, average='macro')
    recall = metrics.recall_score(true_labels, predicted_labels, average='macro')
    f1 = metrics.f1_score(true_labels, predicted_labels, average='macro')
    confusion_matrix = metrics.confusion_matrix(true_labels, predicted_labels)

    # Print the results
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-Score: {f1}')
    print('Confusion Matrix:')
    print(confusion_matrix)

if __name__ == '__main__':
    main_test(_define_args())
