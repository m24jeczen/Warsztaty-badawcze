__author__ = 'marvinler'

import argparse

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.utils.data
import pandas as pd

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from dataset import Dataset
from model import instantiate_sparseconvmil

import os 

from sklearn.metrics import recall_score


def _define_args():
    parser = argparse.ArgumentParser(description='SparseConvMIL: Sparse Convolutional Context-Aware Multiple Instance '
                                                 'Learning for Whole Slide Image Classification')
    
    parser.add_argument('--slide-parent-folder-train', type=str, default='Warsztaty-badawcze/Right_data/Selected_Train_5', metavar='PATH',
                        help='path of parent folder containing preprocessed slides data')
    parser.add_argument('--slide-labels-filepath-train', type=str, default='Warsztaty-badawcze/Right_data/Selected_Train_5/labels.csv',
                        metavar='PATH',
                        help='path of CSV-file containing slide labels')
    
    parser.add_argument('--slide-parent-folder-test', type=str, default='Warsztaty-badawcze/Right_data/Selected_Test_5', metavar='PATH',
                        help='path of parent folder containing preprocessed slides data')
    parser.add_argument('--slide-labels-filepath-test', type=str, default='Warsztaty-badawcze/Right_data/Selected_Test_5/labels.csv',
                        metavar='PATH',
                        help='path of CSV-file containing slide labels')

    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-3, metavar='LR', help='learning rate')
    parser.add_argument('--reg', type=float, default=1e-6, metavar='R', help='weight decay')

    # Model parameters
    parser.add_argument('--tile-embedder', type=str, default='resnet18', metavar='MODEL', nargs='*',
                        help='type of resnet architecture for the tile embedder')
    parser.add_argument('--tile-embedder-pretrained', action='store_true', default=False,
                        help='use Imagenet-pretrained tile embedder architecture')
    parser.add_argument('--sparse-conv-n-channels-conv1', type=int, default=32,
                        help='number of channels of first convolution of the sparse-input CNN pooling')
    parser.add_argument('--sparse-conv-n-channels-conv2', type=int, default=32,
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
    parser.add_argument('--j', type=int, default=10, metavar='N_WORKERS', help='number of workers for dataloader')


        # Save and resume training
    parser.add_argument('--save-model-path', type=str, default='Warsztaty-badawcze/sparseconvmil_model.pth',
                        help='Path to save the trained model')
    parser.add_argument('--load-model-path', type=str, default='Warsztaty-badawcze/sparseconvmil_model.pth',
                        help='Path to load the pre-trained model for continuing training')

    args = parser.parse_args()
    hyper_parameters = {
        'slide_parent_folder_train': args.slide_parent_folder_train,
        'slide_labels_filepath_train': args.slide_labels_filepath_train,
        'slide_parent_folder_test': args.slide_parent_folder_test,
        'slide_labels_filepath_test': args.slide_labels_filepath_test,
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
        'save_model_path': args.save_model_path,
        'load_model_path': args.load_model_path,
    }

    return hyper_parameters


def get_dataloader(dataset, batch_size, shuffle, num_workers):
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle, num_workers=num_workers)

def make_confusion_matrix(ground_truths, predicted_classes, ground_truths_test, predicted_classes_test, n_classes = 2):
    # Confusion matrix for train data
    conf_matrix_train = confusion_matrix(ground_truths, predicted_classes)

    # Confusion matrix for test data
    conf_matrix_test = confusion_matrix(ground_truths_test, predicted_classes_test)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues', xticklabels=range(n_classes), yticklabels=range(n_classes))
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix - Train Data')
    plt.savefig('Warsztaty-badawcze/conf_matrixes/confusion_matrix_wczytywanie_modelu_train.png')  # Saving the confusion matrix for train data as an image
    plt.close()  # Closing the plot to release memory

    # Plotting confusion matrix for test data
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', xticklabels=range(n_classes), yticklabels=range(n_classes))
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix - Test Data')
    plt.savefig('Warsztaty-badawcze/conf_matrixes/confusion_matrix_wczytywanie_modelu_test.png')  # Saving the confusion matrix for test data as an image
    plt.close()  # Closing the plot to release memory

def perform_epoch(mil_model, dataloader, optimizer, loss_function):
    """
    Perform a complete training epoch by looping through all data of the dataloader.
    :param mil_model: MIL model to be trained
    :param dataloader: loader of the dataset
    :param optimizer: pytorch optimizer
    :param loss_function: loss function to compute gradients
    :return: (mean of losses, balanced accuracy)
    """
    proba_predictions = []
    ground_truths = []
    losses = []

    for data, locations, slides_labels, slides_ids in dataloader:
        data = data.cuda()
        locations = locations.cuda()
        slides_labels_cuda = slides_labels.cuda()

        optimizer.zero_grad()
        predictions = mil_model(data, locations)

        loss = loss_function(predictions, slides_labels_cuda)
        loss.backward()
        optimizer.step()

        # Store data for finale epoch average measures
        losses.append(loss.detach().cpu().numpy())
        proba_predictions.extend(predictions.detach().cpu().numpy())
        ground_truths.extend(slides_labels.numpy())

    predicted_classes = np.argmax(proba_predictions, axis=1)
    bac = metrics.balanced_accuracy_score(ground_truths, predicted_classes)
    recall = recall_score(ground_truths, predicted_classes, pos_label=1)


    return np.mean(losses), bac, recall, ground_truths, predicted_classes


def testing(mil_model, dataloader, loss_function):
    proba_predictions = []
    ground_truths = []
    losses = []
    
    for data, locations, slides_labels, slides_ids in dataloader:
        data = data.cuda()
        locations = locations.cuda()
        slides_labels_cuda = slides_labels.cuda()

        #optimizer.zero_grad()
        predictions = mil_model(data, locations)

        loss = loss_function(predictions, slides_labels_cuda)
        #loss.backward()
        #optimizer.step()

        losses.append(loss.detach().cpu().numpy())
        proba_predictions.extend(predictions.detach().cpu().numpy())
        ground_truths.extend(slides_labels.numpy())

    predicted_classes = np.argmax(proba_predictions, axis=1)
    bac = metrics.balanced_accuracy_score(ground_truths, predicted_classes)
    recall = recall_score(ground_truths, predicted_classes, pos_label=1)

    return np.mean(losses), bac, recall, ground_truths, predicted_classes


def save_model(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Model saved to {path}")


def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Model loaded from {path}, resuming from epoch {epoch}")
    return epoch


def main(hyper_parameters):
    # Loads dataset and dataloader
    print('Loading train data')
    dataset_train = Dataset(hyper_parameters['slide_parent_folder_train'], hyper_parameters['slide_labels_filepath_train'],
                      hyper_parameters['n_tiles_per_wsi'])
    n_classes = dataset_train.n_classes
    dataloader_train = get_dataloader(dataset_train, hyper_parameters['batch_size'], True, hyper_parameters['j'])
    print('  done')

    print('Loading test data')
    dataset_test = Dataset(hyper_parameters['slide_parent_folder_test'], hyper_parameters['slide_labels_filepath_test'],
                      hyper_parameters['n_tiles_per_wsi'])
    n_classes = dataset_test.n_classes
    dataloader_test = get_dataloader(dataset_test, hyper_parameters['batch_size'], True, hyper_parameters['j'])
    print('  done')



    # Loads MIL model, optimizer and loss function
    print('Loading SparseConvMIL model')
    sparseconvmil_model = instantiate_sparseconvmil(hyper_parameters['tile_embedder'],
                                                    hyper_parameters['tile_embedder_pretrained'],
                                                    hyper_parameters['sparse_conv_n_channels_conv1'],
                                                    hyper_parameters['sparse_conv_n_channels_conv2'],
                                                    3, 3, hyper_parameters['sparse_map_downsample'],
                                                    hyper_parameters['wsi_embedding_classifier_n_inner_neurons'],
                                                    n_classes)
    sparseconvmil_model = torch.nn.DataParallel(sparseconvmil_model)
    print('  done')
    optimizer = torch.optim.Adam(sparseconvmil_model.parameters(), hyper_parameters['lr'],
                                 weight_decay=hyper_parameters['reg'])
    loss_function = torch.nn.CrossEntropyLoss()

    print('Parsing arguments...')
    
    start_epoch = 0
    if hyper_parameters['load_model_path'] and os.path.exists(hyper_parameters['load_model_path']):
        start_epoch = load_model(sparseconvmil_model, optimizer, hyper_parameters['load_model_path'])

    else:
        print("nie ma modelu do załadowania")

    epoch_data = []

    bac_1_counter = 0

    # tutaj dajcie false, jeżeli chcecie tylko przetestować zapisany model
    training = True

    if (training):
        print('Starting training...')
        for epoch in range(start_epoch,hyper_parameters["epochs"]+start_epoch):
            loss, bac, recall,ground_truths, predicted_classes = perform_epoch(sparseconvmil_model, dataloader_train, optimizer, loss_function)
            if bac == 1:
                bac_1_counter += 1

            # Zapisywanie co 10 epok, jak nie chcecie to zakomentujcie
            if (epoch + 1) % 10 == 0:
                save_model(sparseconvmil_model, optimizer, epoch, hyper_parameters['save_model_path'])

            epoch_data.append([epoch + 1, loss, bac, recall])
            print('Epoch', f'{epoch:3d}/{hyper_parameters["epochs"] +start_epoch}', f'    loss={loss:.3f}', f'    bac={bac:.3f}', f'    recall={recall:.3f}')
            if bac_1_counter == 3:
                break

        save_model(sparseconvmil_model, optimizer, epoch, hyper_parameters['save_model_path'])

        print("Ground truths: ", ground_truths)
        print("Predicted classes: ", predicted_classes)
    

    print('---training  done---')
    # Loop through all epochs
    

        


    print('Checking on test data ')
    loss_test, bac_test, recall_test, ground_truths_test, predicted_classes_test = testing(sparseconvmil_model, dataloader_test, loss_function)
    print("Test results:", f'    loss={loss_test:.3f}', f'    bac={bac_test:.3f}', f'    recall={recall_test:.3f}')
    print("Ground truths test: ", ground_truths_test)
    print("Predicted classes test: ", predicted_classes_test)

    epoch_df = pd.DataFrame(epoch_data, columns=['Epoch', 'Loss', 'BAC','Recall'])


    # Zapisywanie ramki z danymi do wykresów
    csv_path = 'Warsztaty-badawcze/train_epochs_with_recall.csv'
    epoch_df.to_csv(csv_path, index=False)
    print(f"Epoch data saved to {csv_path}")

    print("Making confusion matrix")
    make_confusion_matrix(ground_truths, predicted_classes, ground_truths_test, predicted_classes_test)




if __name__ == '__main__':
    main(_define_args())