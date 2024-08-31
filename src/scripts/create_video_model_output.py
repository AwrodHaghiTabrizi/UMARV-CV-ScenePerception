import os
import shutil
import cv2
import torch
import torch.nn as nn
import dropbox
from getpass import getpass
from tqdm.notebook import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

def initialize_model(model_id, lookback):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = lane_model(lookback=lookback).to(device)
    dbx_access_token = ""
    dbx = dropbox.Dropbox(dbx_access_token)
    dbx_model_weight_dir = f'/UMARV/ComputerVision/ScenePerception/model_weights/model_{model_id}_weights.pth'
    local_model_weights_dir = f'{os.getcwd()}/parameters/output/weights.pth'
    try:
        metadata = dbx.files_get_metadata(dbx_model_weight_dir)
    except dropbox.exceptions.ApiError as e:
        if e.error.is_path() and e.error.get_path().is_not_found():
            print("No model weights found in Dropbox.")
            return model
    file_metadata, res = dbx.files_download(dbx_model_weight_dir)
    with open(local_model_weights_dir, 'wb') as file:
        file.write(res.content)
    model.load_state_dict(torch.load(local_model_weights_dir))
    os.remove(local_model_weights_dir)
    return model

def get_model_output(model, lookback, raw_frames, i):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    default_data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128), antialias=None)
    ])
    

    frames = []
    for j in range(0, lookback['count'] + 1):
        if i - (j * lookback['stride']) < 0:
            frame = frames[-1]
        else:
            frame = raw_frames[i - (j * lookback['stride'])]
            frame = default_data_transform(frame)
            frame = frame.to(device)
        frames.append(frame)
    
    data = torch.cat(frames, dim=0)
    # data = data.unsqueeze(0)

    output = model(data)

    output_display = torch.zeros(raw_frames[0].shape, device=device)
    output_display[0][output == 0], output_display[1][output == 0], output_display[2][output == 0] = 190, 190, 190
    output_display[0][output == 1], output_display[1][output == 1], output_display[2][output == 1] = 220, 0, 0
    output_display[0][output == 2], output_display[1][output == 2], output_display[2][output == 2] = 155, 240, 160
    output_display[0][output == 3], output_display[1][output == 3], output_display[2][output == 3] = 230, 140, 0
    output_display = output_display.permute(1,2,0).byte().cpu().numpy()

    return output_display

    # img = raw_frames[i]
    
    
    # img = default_data_transform(img)
    # img = img.to(device)
    # img = img.unsqueeze(0)
    # model_output = model(img)
    # soft = nn.Softmax(dim=1)
    # soft_output = soft(model_output)
    # soft_ones_output = soft_output[0,1,:,:]
    # ones_output = torch.zeros(soft_ones_output.shape, device=device)
    # ones_output[soft_ones_output > .5] = 1
    # soft_ones_output[0,0], soft_ones_output[0,1] = 1, 0 # Prevent imshow from normalizing the image
    # return soft_ones_output.detach().squeeze().clamp(0,1).cpu().numpy()

class lane_model(nn.Module):
  def __init__(self, lookback):
    super(lane_model, self).__init__()
    self.model = nn.Sequential(
      nn.Conv2d( in_channels=3+3*lookback['count'], out_channels=20 , kernel_size=15 , padding=7 , stride=1 ),
      nn.BatchNorm2d(20),
      nn.LeakyReLU(),
      nn.Conv2d( in_channels=20 , out_channels=10 , kernel_size=15 , padding=7 , stride=1 ),
      nn.BatchNorm2d(10),
      nn.LeakyReLU(),
      nn.Conv2d( in_channels=10 , out_channels=4 , kernel_size=15 , padding=7 , stride=1 ),
    )
  def forward(self, input):
    output = self.model(input)
    return output

def main():

    repo_dir = os.getcwd()
    input_dir = f"{repo_dir}/parameters/input"
    output_dir = f"{repo_dir}/parameters/output"

    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mp4')]
    if len(video_files) == 0:
        print("Input directory does not contain video of file type .mp4 .")
        return
    elif len(video_files) > 1:
        print("Input directory contains multiple videos. Should only contain one video.")
        return

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Capture the frames from the input video
    video_dir = f"{input_dir}/{video_files[0]}"
    cap = cv2.VideoCapture(video_dir)
    if not cap.isOpened():
        print("Error opening video file.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_step = 1
    frame_index = 0
    data_index = 0
    raw_frames = []
    frame_size = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_step == 0:
            raw_frames.append(frame)
            data_index += 1
            if not frame_size:
                frame_size = (frame.shape[1], frame.shape[0])
        frame_index += 1
    cap.release()

    # Create the segmented model output
    lookback = {'count': 5, 'stride': 1}
    model = initialize_model(model_id="32j3pqu5", lookback=lookback)
    segmented_outputs = []
    # lane_outputs_binary = []
    # overlayed_frames = []
    # composite_frames = []
    # half_size = lambda image: cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    for i, img in enumerate(raw_frames):
        segmented_output = get_model_output(model, lookback, raw_frames, i)  # get_lane_output is a placeholder for your actual function
        segmented_output = cv2.resize(segmented_output, frame_size)
        # segmented_output = np.stack((lane_output_soft, lane_output_soft, lane_output_soft), axis=2)
        # segmented_output = (lane_output_soft * 255).astype(np.uint8)
        segmented_outputs.append(segmented_output)
        
    #     # Creating a binary mask where detected lanes are
    #     binary = np.zeros_like(lane_output_soft[:,:,0])
    #     binary[lane_output_soft[:,:,0] > 25] = 1
    #     lane_output_binary = (binary * 255).astype(np.uint8)
    #     lane_output_binary = np.stack((lane_output_binary, lane_output_binary, lane_output_binary), axis=2)
    #     lane_outputs_binary.append(lane_output_binary)
        
    #     # Create an overlay where the binary mask is true
    #     overlayed_frame = np.copy(img)
    #     overlayed_frame[binary == 1] = [0, 255, 255]  # Setting pixels to bright yellow where mask is true
    #     overlayed_frames.append(overlayed_frame)
    
    #     # Resize each quadrant to half the original dimensions
    #     height, width = img.shape[:2]
    #     half_size = (width // 2, height // 2)
    #     img_small = cv2.resize(img, half_size)
    #     overlayed_frame_small = cv2.resize(overlayed_frame, half_size)
    #     lane_output_soft_small = cv2.resize(lane_output_soft, half_size)
    #     lane_output_binary_small = cv2.resize(lane_output_binary, half_size)
    #     # Creating a blank canvas for the composite image
    #     composite_image = np.zeros((height, width, 3), dtype=np.uint8)
    #     # Placing each small image in its respective quadrant
    #     composite_image[:height//2, :width//2] = img_small  # Top-left
    #     composite_image[:height//2, width//2:] = overlayed_frame_small  # Top-right
    #     composite_image[height//2:, :width//2] = lane_output_soft_small  # Bottom-left
    #     composite_image[height//2:, width//2:] = lane_output_binary_small  # Bottom-right
    #     composite_frames.append(composite_image)

    # Create the raw video
    output_video_raw_dir = f"{output_dir}/raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_raw_dir, fourcc, fps, frame_size)
    for img in raw_frames:
        video_writer.write(img)
    video_writer.release()

    # Create the video of segmented model output
    output_video_lanes_dir = f"{output_dir}/lanes_soft.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_lanes_dir, fourcc, fps, frame_size)
    for img in segmented_outputs:
        video_writer.write(img)
    video_writer.release() 

    # # Create the video of binary output of the captured lanes
    # output_video_lanes_dir = f"{output_dir}/lanes_binary.mp4"
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_writer = cv2.VideoWriter(output_video_lanes_dir, fourcc, fps, frame_size)
    # for img in lane_outputs_binary:
    #     video_writer.write(img)
    # video_writer.release()

    # # Create the video of binary output of the captured lanes overlayed on the original frames
    # output_video_lanes_dir = f"{output_dir}/overlay.mp4"
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_writer = cv2.VideoWriter(output_video_lanes_dir, fourcc, fps, frame_size)
    # for img in overlayed_frames:
    #     video_writer.write(img)
    # video_writer.release()

    # # Create the matrix video combining all 4 outputs
    # output_video_lanes_dir = f"{output_dir}/matrix.mp4"
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # video_writer = cv2.VideoWriter(output_video_lanes_dir, fourcc, fps, frame_size)
    # for img in composite_frames:
    #     video_writer.write(img)
    # video_writer.release()

    print("ALL DONE")

if __name__ == "__main__":
    main()