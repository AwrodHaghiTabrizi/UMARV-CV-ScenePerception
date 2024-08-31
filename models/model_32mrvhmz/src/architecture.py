import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import yolov5

class lane_model(nn.Module):
    def __init__(self, lookback):
        super(lane_model, self).__init__()
        self.yolov5_model = yolov5.load('yolov5s')  # Using yolov5s as an example

        # Replace the YOLOv5 head with a segmentation head
        self.yolov5_model.model[-1] = nn.Conv2d(256, 4, kernel_size=1)

    def forward(self, x):
        # Forward pass through YOLOv5
        features = self.yolov5_model.model[:-1](x)  # Forward pass excluding the last layer
        output = self.yolov5_model.model[-1](features)  # Forward pass through the new segmentation head
        return output

# class lane_model(nn.Module):
#     def __init__(self, lookback):
#         super(lane_model, self).__init__()
        
#         # Initialize the feature extractor and model
#         self.feature_extractor = SegformerFeatureExtractor()
#         self.model = SegformerForSemanticSegmentation(num_labels=4)
        
#         # Set device (optional, if you plan to use CUDA)
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.model.to(self.device)
        
#     def forward(self, x):
#         # Preprocess the input (resizing and normalization)
#         inputs = self.feature_extractor(x, return_tensors="pt")
        
#         # Ensure inputs are moved to the correct device
#         inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
#         # Perform the forward pass
#         outputs = self.model(**inputs)
        
#         return outputs

# class lane_model(nn.Module):
#   def __init__(self, lookback):
#     super(lane_model, self).__init__()
#     self.model = nn.Sequential(
#       nn.Conv2d( in_channels=3+3*lookback['count'], out_channels=20 , kernel_size=15 , padding=7 , stride=1 ),
#       nn.BatchNorm2d(20),
#       nn.LeakyReLU(),
#       nn.Conv2d( in_channels=20 , out_channels=20 , kernel_size=15 , padding=7 , stride=1 ),
#       nn.BatchNorm2d(20),
#       nn.LeakyReLU(),
#       nn.Conv2d( in_channels=20 , out_channels=20 , kernel_size=15 , padding=7 , stride=1 ),
#       nn.BatchNorm2d(20),
#       nn.LeakyReLU(),
#       nn.Conv2d( in_channels=20 , out_channels=10 , kernel_size=15 , padding=7 , stride=1 ),
#       nn.BatchNorm2d(10),
#       nn.LeakyReLU(),
#       nn.Conv2d( in_channels=10 , out_channels=4 , kernel_size=15 , padding=7 , stride=1 ),
#     )
#   def forward(self, input):
#     output = self.model(input)
#     return output

# class lane_model(nn.Module):
#   def __init__(self, lookback):
#     super(lane_model, self).__init__()

#     in_channels = 3 + 3 * lookback['count']
#     out_channels = 4

#     # Contracting path
#     self.enc1 = self.conv_block(in_channels, 64)
#     self.enc2 = self.conv_block(64, 128)
#     self.enc3 = self.conv_block(128, 256)
#     self.enc4 = self.conv_block(256, 512)
    
#     # Bottleneck
#     self.bottleneck = self.conv_block(512, 1024)
    
#     # Expansive path
#     self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
#     self.dec4 = self.conv_block(1024, 512)
#     self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
#     self.dec3 = self.conv_block(512, 256)
#     self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
#     self.dec2 = self.conv_block(256, 128)
#     self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
#     self.dec1 = self.conv_block(128, 64)
    
#     # Final layer
#     self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

#   def conv_block(self, in_channels, out_channels):
#     return nn.Sequential(
#       nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#       nn.BatchNorm2d(out_channels),
#       nn.LeakyReLU(),
#       nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#       nn.BatchNorm2d(out_channels),
#       nn.LeakyReLU()
#     )

#   def forward(self, x):
#     # Contracting path
#     enc1 = self.enc1(x)
#     enc2 = self.enc2(F.max_pool2d(enc1, kernel_size=2, stride=2))
#     enc3 = self.enc3(F.max_pool2d(enc2, kernel_size=2, stride=2))
#     enc4 = self.enc4(F.max_pool2d(enc3, kernel_size=2, stride=2))
    
#     # Bottleneck
#     bottleneck = self.bottleneck(F.max_pool2d(enc4, kernel_size=2, stride=2))
    
#     # Expansive path
#     dec4 = self.upconv4(bottleneck)
#     dec4 = torch.cat((dec4, enc4), dim=1)
#     dec4 = self.dec4(dec4)
#     dec3 = self.upconv3(dec4)
#     dec3 = torch.cat((dec3, enc3), dim=1)
#     dec3 = self.dec3(dec3)
#     dec2 = self.upconv2(dec3)
#     dec2 = torch.cat((dec2, enc2), dim=1)
#     dec2 = self.dec2(dec2)
#     dec1 = self.upconv1(dec2)
#     dec1 = torch.cat((dec1, enc1), dim=1)
#     dec1 = self.dec1(dec1)
    
#     # Final layer
#     return self.final_conv(dec1)