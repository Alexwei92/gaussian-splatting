# Dependencies
pip install plyfile
pip install torch
pip install torchaudio
pip install torchvision
pip install tqdm
pip install opencv-python
pip install joblib

# Training speed acceleration
cd submodules/diff-gaussian-rasterization
rm -r build
git checkout 3dgs_accel
cd ../..

# Submodules
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install submodules/fused-ssim

# SIBR Viewer Dependencies
sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev
cd SIBR_viewers
rm -r build
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release # add -G Ninja to build faster
cmake --build build -j24 --target install
cd ..