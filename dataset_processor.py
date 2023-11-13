import csv
import os
import shutil
from pydub import AudioSegment
from riffusion.cli import audio_to_images_batch
from datasets import load_dataset

class DatasetProcessorParams:
    input_directory: str = ""
    output_directory: str = ""
    huggingface_dataset_name: str = ""
    huggingface_token: str = ""
    limit: int = -1
    device: str = "cuda"
    mono: bool = True

class DatasetProcessor:
    def __init__(self, params: DatasetProcessorParams):
        self.params = params
    
    def setup_dataset_folder(self):
        dataset_path= os.path.join(self.params.output_directory, 'dataset/train/')
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
        os.makedirs(dataset_path, exist_ok=True)
        return dataset_path
    
    def split_wav_files(self, directory: str, output_directory: str) -> None:
        """
        This function splits .wav files in a given directory into chunks of 5 seconds each.
        :param directory: The directory where the .wav files are located.
        :param output_directory: The directory where the processed .wav files are being put
        """
        for filename in os.listdir(directory):
            if filename.endswith(".wav"):
                audio = AudioSegment.from_wav(os.path.join(directory, filename))
                length_audio = len(audio)
                start = 0

                # In milliseconds, 5000ms = 5s
                threshold = 5000
                while start < length_audio:
                    chunk = audio[start:start+threshold]
                    # Only save chunks that are 5 seconds long
                    if len(chunk) == threshold:
                        start_time = (start // 1000)
                        end_time = ((start + threshold) // 1000)
                        chunk.export(os.path.join(output_directory, f"{start_time}-to-{end_time}-seconds_{filename}"), format="wav")
                    start += threshold
    
    def convert_audio_to_image_batch(self, dataset_directory: str):
        audio_to_images_batch(audio_dir=dataset_directory, 
                              output_dir=dataset_directory,
                              mono=self.params.mono, 
                              device=self.params.device,
                              limit=self.params.limit)

        for filename in os.listdir(dataset_directory):
            if filename.endswith(".wav"):
                os.remove(os.path.join(dataset_directory, filename))

    def create_metedata(self, dataset_directory: str):
        metadata_file = os.path.join(dataset_directory, 'metadata.csv')

        with open(metadata_file, 'w', newline='') as csvfile:
            # write header
            fieldnames = ['file_name', 'text']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # write headers
            for filename in os.listdir(dataset_directory):
                if not filename.endswith('.jpg'):
                    continue

                full_split = filename.split('_')
                genre_split = ' '.join(full_split[2].split('-'))
                alternative = full_split[-1].split('.')[0]
                text_description = f'{full_split[0]} of drumming in {genre_split} with {full_split[3]} beat rythm, version {full_split[1]}, {alternative}'
                writer.writerow({'file_name': filename, 'text': text_description})

    def run(self):
        print("Reset Dataset folder")
        dataset_directory = self.setup_dataset_folder()

        print("Splitting audio into 5 second clips")
        self.split_wav_files(directory=self.params.input_directory,
                             output_directory=dataset_directory)

        print("Converting audio clips into spectrograms")
        self.convert_audio_to_image_batch(dataset_directory=dataset_directory)

        print("Attaching dataset metadata")
        self.create_metedata(dataset_directory=dataset_directory)

        print("Uploading to huggingface")
        train_data = load_dataset('imagefolder', data_dir='./dataset')
        train_data.push_to_hub(self.params.huggingface_dataset_name,
                               token=self.params.huggingface_token)

        print(f'Updated dataset {self.params.huggingface_dataset_name}')

params = DatasetProcessorParams()
params.input_directory = "/home/nass/Desktop/workspace/first-diffusion-model/groove/drummer1/session1/"
params.output_directory = "."
params.huggingface_dataset_name = "lloydoad/drum-spectrogram-dataset"
params.huggingface_token = "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"
DatasetProcessor(params=params).run()