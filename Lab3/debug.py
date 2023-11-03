# Assuming the output of the encoder is Batch x 512 x 4 x 4
# Note: This is just a hypothesis for illustrative purposes
import torch
import torchvision
import torchvision.transforms as transforms

from vanilla import VanillaFrontend, VGG, ModFrontend
#from model import VGG, ModFrontend, VisualTransformerDecoder
import argparse

encoder_output = torch.rand(1, 512, 4, 4)  # Example tensor output of the encoder

# Initialize ModFrontend with dummy encoder (since we are manually providing encoder output)
mod_frontend = ModFrontend(encoder=None, num_classes=100)

# Simulate forwarding through ModFrontend, printing sizes at each step
current_input = encoder_output
print(f"Input to ModFrontend: {current_input.size()}")

current_input = mod_frontend.pool(current_input)
print(f"After AdaptiveAvgPool2d: {current_input.size()}")

current_input = current_input.view(current_input.size(0), -1)  # Flatten
print(f"After Flatten: {current_input.size()}")

current_input = mod_frontend.intermediate(current_input)
print(f"After Intermediate: {current_input.size()}")

# Note: SqueezeExcitationBlock needs to be fixed as per previous discussion

# The residual connection
residual = mod_frontend.match_dimension(current_input)
print(f"After Match Dimension (Residual): {residual.size()}")

current_input = current_input + residual
print(f"After Residual Addition: {current_input.size()}")

output = mod_frontend.classifier(current_input)
print(f"Final Output: {output.size()}")
