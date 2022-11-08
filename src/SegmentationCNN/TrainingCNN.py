from CNNData import CNNData 
from DataPreprocessing import * 

wavs, tsvs, names = get_wavs_and_tsvs("/Users/serenahuston/GitRepos/python-classifier-2022/physionet.org/files/circor-heart-sound/1.0.3/training_data",
                                return_names=True)


dp = DataPreprocessing(wavs[0], tsvs[0], names[0])
in_patches = dp.extract_env_patches()

out_patches = dp.extract_segmentation_patches()
print(out_patches.shape)

# train_data = CNNData()
# trainloader = DataLoader(dataset=train_data, batch_size=100)
# validationloader = DataLoader(dataset=validation_data, batch_size=5000)


# model = UNet()

# tensor = torch.from_numpy(dp.input_patches[0]).type(torch.float32)

# print(tensor.dtype)
# print(model.forward(tensor).shape)