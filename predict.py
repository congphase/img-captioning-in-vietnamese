import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os,json
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        print("using " + args.vocab_path)
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))
    
    with open(args.caption_path) as f:
        val_data = json.load(f)

    results_data = []
    curr_id = 0
    for index, image_data in enumerate(val_data['images']):
        # Prepare an image
        image_path = args.image_dir + "/" + image_data['file_name']
        image = load_image(image_path, transform)
        image_tensor = image.to(device)
        
        # Generate an caption from the image
        feature = encoder(image_tensor)
        sampled_ids = decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
        
        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)
        sentence = sentence.replace('<start> ','')
        sentence = sentence.replace(' <end>', '')
        record = {
            'image_id': int(image_data['id']),
            'caption': sentence.replace('_',' '),
            'id': curr_id
        }
        curr_id+=1
        results_data.append(record)
        if index%10 == 0:
            print(f"Done image {index}/{len(val_data['images'])}")
        
    with open(args.result_path, 'w+') as f_results:
        f_results.write(json.dumps(results_data, ensure_ascii=False))
    print(f"The result saved at {args.result_path}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default='models/encoder-5-3000.pkl', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-5-3000.pkl', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocabUIT_v2.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default=None, help='path for resized validation images')
    parser.add_argument('--caption_path', type=str, default=None, help='path for validation captions')
    parser.add_argument('--result_path', type=str, default='./result.json', help='path for result')
    
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()

    main(args)
