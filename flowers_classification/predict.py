#!/usr/bin/env python3

# Imports
import os
import argparse
import model_nw
import utils

def parse_input_args():
    parser = argparse.ArgumentParser(description='Predict arguments')
    parser.add_argument('input', type=str, help='Image filename')
    parser.add_argument('checkpoint', type=str, help='Checkpoint filename')
    parser.add_argument('--top_k', type=int, default=5, help='Top k classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Category names')
    parser.add_argument('--gpu', default=False, action='store_true', help='Use GPU')
    
    return parser.parse_args()

def main():
    args = parse_input_args()

    # Validate input file
    if not os.path.isfile(args.input):
        print("Error: input file not found")
        exit(1)

    # Set device
    device = 'cpu'
    if args.gpu == True:
        if model_nw.is_gpu_available():
            device = 'cuda'
        else:
            print("Error: GPU unavailable")
            exit(1)

    # Load checkpoint
    if not os.path.isfile(args.checkpoint):
        print("Error: checkpoint file not found")
        exit(1)
    model = model_nw.load_checkpoint(args.checkpoint)

    # Predict using model
    print("Predict for {} on {}".format(args.input, device))
    probs, classes = model_nw.predict(model, args.input, args.top_k, device)

    # Load category names
    cat_to_name = []
    if len(args.category_names) > 0:
        cat_to_name = utils.get_cat_to_name(args.category_names)

    # Print probability and classes
    for cat, prob in zip(classes, probs):
        if cat in cat_to_name:
            category = cat_to_name[cat]
        else:
            category = cat
        print ("Category {}: Probability {:.2f}".format(category, prob))

if __name__ == '__main__':
    main()
