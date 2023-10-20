@echo off

set CONTENT_IMG_PATH="images/content/baboon.jpg"
set STYLE_IMG_PATH_IDK="images/style/0a585acb9d7134c0b39656a588527385c.jpg"
set STYLE_IMG_PATH_WARHOL="images/style/Andy_Warhol_97.jpg"
set STYLE_IMG_PATH_BRUSH="images/style/brushstrokes.jpg"
set STYLE_IMG_PATH_MARC="images/style/chagall_marc_1.jpg"
set STYLE_IMG_PATH_PERS="images/style/the-persistence-of-memory-1931.jpg"
set ENCODER_PATH="models/encoder.pth"
set DECODER_PATH="models/decoder_g130_10k.pth"
set OUTPUT_PATH="output/10k/gamma1.30/baboon/"
set CUDA="N"

echo "Running test.py"

set ALPHA=0.1
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%

set ALPHA=0.5
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%


set ALPHA=0.9
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%


