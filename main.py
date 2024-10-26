import os
from preprocessing import preprocess_main
from model_VGG import train_vgg
from model_resnet50 import train_resnet
from model_inceptionV3 import train_inceptionv3
from model_efficientnet import train_efficientnet
from model_densenet import train_densenet
from post_processing import post_processing


class ModelPipeline:
    def __init__(self, directory, target_name):
        self.directory = directory
        self.target_name = target_name
        self.new_directory_path = os.path.join(directory, target_name)

    def find_target_folder(self):
        """Finds the target folder in the directory."""
        try:
            entries = os.listdir(self.directory)
            folders = [
                entry for entry in entries 
                if os.path.isdir(os.path.join(self.directory, entry)) and entry == self.target_name
            ]
            return folders
        except Exception as e:
            print(f"Error listing folders in {self.directory}: {e}")
            return []

    def preprocess_data(self):
        """Runs the preprocessing step if the target folder exists."""
        if os.path.exists(self.new_directory_path):
            preprocess_main(self.new_directory_path, img_size=224)
            print("Preprocessing complete.")

    def train_models(self):
        """Trains the models based on the new directory path."""
        # Uncomment the models you want to train
        # train_vgg(self.new_directory_path, self.directory)
        train_resnet(self.new_directory_path, self.directory)
        # train_inceptionv3(self.new_directory_path, self.directory)
        # train_efficientnet(self.new_directory_path, self.directory)
        # train_densenet(self.new_directory_path, self.directory)
        print("Model training complete.")

    def post_process(self):
        """Performs post-processing on the trained models' output."""
        post_processing(self.new_directory_path, self.directory)
        print("Post-processing complete.")

    def run_pipeline(self):
        """Executes the entire pipeline."""
        folders = self.find_target_folder()
        if folders:
            print(f"Found target folder: {folders}")
            self.preprocess_data()
            self.train_models()
            self.post_process()
        else:
            print("Target folder not found.")
        return folders


# Example usage
if __name__ == "__main__":
    directory = os.getcwd()
    target_name = 'data'
    pipeline = ModelPipeline(directory, target_name)
    folders = pipeline.run_pipeline()
    print("Pipeline execution complete.")
