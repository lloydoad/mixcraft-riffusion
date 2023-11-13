# from riffusion.cli import audio_to_image, image_to_audio

# test_sample = "testing_samples/sample_01.wav"
# test_image_output = "testing_samples/created_sample_01.png"
# test_audio_output = "testing_samples/created_sample_01.wav"
# audio_to_image(audio=test_sample, image=test_image_output)

# image_to_audio(image=test_image_output, audio=test_audio_output)

from dataset_processor import split_wav_files

input = "/home/nass/Desktop/workspace/first-diffusion-model/groove/drummer1/session1/"
output = "./drumming_samples/"
split_wav_files(directory=input, output_directory=output)