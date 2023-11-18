# Perona-Malik with PyTorch and CNNs

Perona-Malik diffusion is a well known regularization technique that preserves edges. Although we normally use a constant kernel to calculate the gradients (i.e. an edge detector such as Laplacian), using CNNs and PyTorch, it is possible to explore operators other than edge detectors. When combined with an appropriate loss function and the right diffusion rate vs learning rate, this allows exploring aesthetically pleasing diffusion reaction transformations which still temporarily preserve edges to some extent. Feel free to see my [blog](https://gozepolat.github.io/posts/Art_with_math/) for more details.

## Usage

Just run `python demo.py --image <path/to/image> --out-folder <optional/path/to/save/folder>`! For camera input, try `demo_video.py` after making sure that you have GPU acceleration enabled and `opencv` installed.

### Example

`python demo.py --image images/star.jpg --out-folder transformations` will create transformed versions of the image `star.jpg` and save them in the output folder `transformations`.

<img src="transformations/star_0.jpg" width="20%" height="20%"><img src="transformations/star_1.jpg" width="20%" height="20%"><img src="transformations/star_2.jpg" width="20%" height="20%"><img src="transformations/star_3.jpg" width="20%" height="20%"><img src="transformations/star_4.jpg" width="20%" height="20%"><img src="transformations/star_5.jpg" width="20%" height="20%"><img src="transformations/star_6.jpg" width="20%" height="20%"><img src="transformations/star_7.jpg" width="20%" height="20%"><img src="transformations/star_8.jpg" width="20%" height="20%"><img src="transformations/star_9.jpg" width="20%" height="20%"><img src="transformations/star_10.jpg" width="20%" height="20%"><img src="transformations/star_11.jpg" width="20%" height="20%"><img src="transformations/star_12.jpg" width="20%" height="20%"><img src="transformations/star_13.jpg" width="20%" height="20%"><img src="transformations/star_14.jpg" width="20%" height="20%"><img src="transformations/star_15.jpg" width="20%" height="20%"><img src="transformations/star_16.jpg" width="20%" height="20%"><img src="transformations/star_17.jpg" width="20%" height="20%"><img src="transformations/star_18.jpg" width="20%" height="20%"><img src="transformations/star_19.jpg" width="20%" height="20%"><img src="transformations/star_20.jpg" width="20%" height="20%"><img src="transformations/star_21.jpg" width="20%" height="20%"><img src="transformations/star_22.jpg" width="20%" height="20%"><img src="transformations/star_23.jpg" width="20%" height="20%"><img src="transformations/star_24.jpg" width="20%" height="20%"><img src="transformations/star_25.jpg" width="20%" height="20%"><img src="transformations/star_26.jpg" width="20%" height="20%"><img src="transformations/star_27.jpg" width="20%" height="20%"><img src="transformations/star_28.jpg" width="20%" height="20%"><img src="transformations/star_29.jpg" width="20%" height="20%"><img src="transformations/star_30.jpg" width="20%" height="20%"><img src="transformations/star_31.jpg" width="20%" height="20%"><img src="transformations/star_32.jpg" width="20%" height="20%"><img src="transformations/star_33.jpg" width="20%" height="20%"><img src="transformations/star_34.jpg" width="20%" height="20%">

#### Sample Video
[profile.webm](https://github.com/gozepolat/fast-perona-malik/assets/25344752/5191e484-b693-4a8a-af07-6974191afe02)


