import argparse
import os
from PIL import Image
import torch
import torchvision.transforms as transforms 
from torchvision.utils import save_image
import AdaIN_net as net

if __name__ == '__main__':

	image_size = 512
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	parser = argparse.ArgumentParser()
	parser.add_argument('-content_image', type=str, help='test image')
	parser.add_argument('-style_image', type=str, help='style image')
	parser.add_argument('-encoder_file', type=str, help='encoder weight file')
	parser.add_argument('-decoder_file', type=str, help='decoder weight file')
	parser.add_argument('-output_path' , type=str, help='output path', default='./output/')
	parser.add_argument('-alpha', type=float, default=1.0, help='Level of style transfer, value between 0 and 1')
	parser.add_argument('-cuda', type=str, help='[Y/N]')

	opt = parser.parse_args()
	# load image are rgb
	content_image = Image.open(opt.content_image, mode='r').convert('RGB')
	style_image = Image.open(opt.style_image, mode='r').convert('RGB')
	output_format = opt.content_image[opt.content_image.find('.'):]
	decoder_file = opt.decoder_file
	encoder_file = opt.encoder_file
	alpha = opt.alpha
	use_cuda = False
	if opt.cuda == 'y' or opt.cuda == 'Y':
		use_cuda = True
	out_dir = opt.output_path
	os.makedirs(out_dir, exist_ok=True)

	encoder = net.encoder_decoder.encoder
	encoder.load_state_dict(torch.load(encoder_file, map_location='cpu'))
	decoder = net.encoder_decoder.decoder
	decoder.load_state_dict(torch.load(decoder_file, map_location='cpu'))
	model = net.AdaIN_net(encoder, decoder)

	model.to(device=device)
	model.eval()

	

	content_image = transforms.Resize(size=image_size)(content_image)
	style_image = transforms.Resize(size=image_size)(style_image)

	input_tensor = transforms.ToTensor()(content_image).unsqueeze(0)
	style_tensor = transforms.ToTensor()(style_image).unsqueeze(0)

	if torch.cuda.is_available() and use_cuda:
		model.cuda()
		input_tensor = input_tensor.cuda()
		style_tensor = style_tensor.cuda()

		

	out_tensor = None
	with torch.no_grad():
		out_tensor = model(input_tensor, style_tensor, alpha)

	save_file = out_dir + opt.content_image[opt.content_image.rfind('/')+1: opt.content_image.find('.')] \
							+"_style_"+ opt.style_image[opt.style_image.rfind('/')+1: opt.style_image.find('.')] \
							+ "_alpha_" + str(alpha) \
							+ "_10k_decoder" + output_format 
	print('saving output file: ', save_file)
	save_image(out_tensor, save_file)