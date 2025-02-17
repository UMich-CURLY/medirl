import os
from PIL import Image
from IPython import embed
def extract_gif_frames(gif_path, output_dir):
    """
    Extract frames from a GIF, resize them to 60x60, and save them into separate folders.

    :param gif_path: Path to the GIF file.
    :param output_dir: Path to the output directory where frames will be saved.
    """
    # Open the GIF file
    with Image.open(gif_path) as gif:
        num_frames = gif.n_frames
        print("Number of frames: ", num_frames)
    # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        for frame_index in range(num_frames):
            # Seek to the current frame
            gif.seek(frame_index)

            # Create a folder for the current frame
            frame_folder = os.path.join(output_dir, str(frame_index))
            os.makedirs(frame_folder, exist_ok=True)

            # Resize the current frame to 60x60
            # resized_frame = gif.resize((60, 60))
            resized_frame = gif.copy()
            # Save the current frame as an image
            resized_frame.save(frame_folder+"/grid_map.png")



# Example usage
gif_path = "episode_0.gif"  # Path to the GIF
output_dir = "test_irl/demo_0"       # Path to the output directory
extract_gif_frames(gif_path, output_dir)
