#!/usr/bin/env python3

# Imports
import os
import argparse
import model_nw

def parse_input_args():
    parser = argparse.ArgumentParser(description='Train arguments')
    parser.add_argument('data_dir', type=str, help='Data directory')
    parser.add_argument('--save_dir', default='', help='Directory to save checkpoints')
    parser.add_argument('--save_file', default='saved_model.pth', help='Checkpoint filename')
    parser.add_argument('--arch', default='resnet', help='Architecture (resnet, densenet)')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate' )
    parser.add_argument('--hidden_units', default='512', type=str, help='Hidden units' )
    parser.add_argument('--epochs', default=5, type=int, help='Number of epochs')
    parser.add_argument('--gpu', default=False, action='store_true', help='Use GPU')
    
    return parser.parse_args()

def main():
    args = parse_input_args()

    # Setup data loaders
    if os.path.isdir(args.data_dir):
        train_loader, valid_loader, test_loader, class_to_idx = \
                                            model_nw.get_data_loaders(args.data_dir)
    else:
        print("Error: data_dir not available")
        exit(1)

    # Setup pretrained model and train it
    hidden_units = list(map(int, args.hidden_units.split(',')))
    model, classifier = model_nw.setup_pretrained_model(args.arch, hidden_units, len(class_to_idx))
    criterion, optimizer = model_nw.get_criterion_optimizer(classifier, args.learning_rate)
    model.class_to_idx = class_to_idx

    # Train model
    print_every = 20
    device = 'cpu'
    if args.gpu == True:
        if model_nw.is_gpu_available():
            device = 'cuda'
        else:
            print("Error: GPU unavailable")
            exit(1)
 
    print ("Training model for arch {} on {}".format(args.arch, device))
    model_nw.train_model(model, train_loader, valid_loader, args.epochs, print_every, \
                         criterion, optimizer, device)

    # Check test accuracy
    total, correct = model_nw.check_accuracy(model, test_loader, device)
    print('Test Accuracy for {} images: {:.2f}%%'.format(total, 100 * correct / total))
    
    # Save the trained model
    if len(args.save_dir) > 0:
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        save_path = args.save_dir + '/'
    else:
        save_path = ''

    save_path = save_path + args.save_file
    model_nw.save_checkpoint(save_path, model, args.arch, args.epochs, hidden_units)
    print("Saved trained model to {}".format(save_path))

if __name__ == '__main__':
    main()
