@echo off

set CONTENT_IMG_PATH="images/content/Lena.png"
set STYLE_IMG_PATH_IDK="images/style/0a585acb9d7134c0b39656a588527385c.jpg"
set STYLE_IMG_PATH_WARHOL="images/style/Andy_Warhol_97.jpg"
set STYLE_IMG_PATH_BRUSH="images/style/brushstrokes.jpg"
set STYLE_IMG_PATH_MARC="images/style/chagall_marc_1.jpg"
set STYLE_IMG_PATH_PERS="images/style/the-persistence-of-memory-1931.jpg"
set STYLE_IMG_PATH_NIGHT="images/style/starynight.jpg"
set ENCODER_PATH="models/encoder.pth"
set DECODER_PATH="models/decoder_g015_10k.pth"
set OUTPUT_PATH="output/10k/gamma0.15/Lena/"
set CUDA="N"

echo "Running test.py with gamma=0.15"

set ALPHA=0.1
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%

set ALPHA=0.5
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%

set ALPHA=0.9
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%

echo "Running test.py with gamma=0.30"
set DECODER_PATH="models/decoder_g030_10k.pth"
set OUTPUT_PATH="output/10k/gamma0.30/Lena/"

set ALPHA=0.1
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%

set ALPHA=0.5
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%

set ALPHA=0.9
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%

echo "Running test.py with gamma=0.75"
set DECODER_PATH="models/decoder_g075_10k.pth"
set OUTPUT_PATH="output/10k/gamma0.75/Lena/"

set ALPHA=0.1
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%

set ALPHA=0.5
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%

set ALPHA=0.9
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%


echo "Running test.py with gamma=1.20"
set DECODER_PATH="models/decoder_g120_10k.pth"
set OUTPUT_PATH="output/10k/gamma1.20/Lena/"

set ALPHA=0.1
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%

set ALPHA=0.5
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%

set ALPHA=0.9
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%


echo "Running test.py with gamma=1.30"
set DECODER_PATH="models/decoder_g130_10k.pth"
set OUTPUT_PATH="output/10k/gamma1.30/Lena/"

set ALPHA=0.1
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%

set ALPHA=0.5
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%

set ALPHA=0.9
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%


echo "Running test.py with gamma=1.60"
set DECODER_PATH="models/decoder_g160_10k.pth"
set OUTPUT_PATH="output/10k/gamma1.60/Lena/"

set ALPHA=0.1
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%

set ALPHA=0.5
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%

set ALPHA=0.9
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%


echo "Running test.py with gamma=2.00"
set DECODER_PATH="models/decoder_g20_10k.pth"
set OUTPUT_PATH="output/10k/gamma2.00/Lena/"

set ALPHA=0.1
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%

set ALPHA=0.5
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%

set ALPHA=0.9
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%


echo "Running test.py with gamma=2.50"
set DECODER_PATH="models/decoder_g250_10k.pth"
set OUTPUT_PATH="output/10k/gamma2.50/Lena/"

set ALPHA=0.1
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%

set ALPHA=0.5
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%

set ALPHA=0.9
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%


echo "Running test.py with gamma=10.0
set DECODER_PATH="models/decoder_g10.0_10k.pth"
set OUTPUT_PATH="output/10k/gamma10.0/Lena/"

set ALPHA=0.1
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%

set ALPHA=0.5
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%

set ALPHA=0.9
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_NIGHT% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_IDK% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_WARHOL% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_BRUSH% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_MARC% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
python test.py -content_image %CONTENT_IMG_PATH% -style_image %STYLE_IMG_PATH_PERS% -encoder_file %ENCODER_PATH% -decoder_file %DECODER_PATH% -output_path %OUTPUT_PATH% -alpha %ALPHA% -cuda %CUDA%
