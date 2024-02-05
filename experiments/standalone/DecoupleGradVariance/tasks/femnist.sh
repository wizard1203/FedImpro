export dataset=${dataset:-"femnist"}
export model_input_channels=${model_input_channels:-1}
# export model=${model:-"mnistflnet"}
export model=${model:-"resnet18_v2"}
export num_classes=${num_classes:-62}
export dataset_load_image_size=28
export model_output_dim=${model_output_dim:-62}
export lr=${lr:-0.01}