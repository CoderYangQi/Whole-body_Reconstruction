# 🌐 Whole-body_Reconstruction

This repository is part of the work based on the [Volume-reconstruction](https://github.com/SMART-pipeline/Volume-reconstruction) project, focusing on advanced imaging and reconstruction techniques.

## 🖥️ System Requirements

- **Graphics Card:** Nvidia graphic card with over 8 GB of memory
- **Operating System:** Windows

## 🛠 Installation

### 🌟 Strongly Recommended (Avoid installing dependencies individually)

1. **Download Environment and Anaconda-5.3.1:** Get the necessary files via this [Link](https://rec.ustc.edu.cn/share/609a7520-2d6c-11ef-b3a9-8556057b7c72).
2. **Install Anaconda-5.3.1:** Follow the instructions to install and unzip the venv.7z.
3. **Verify the Environment:**
   ```
   click Whole-body_Reconstruction\VISoR_Reconstruction\run_visor_reconstruction.bat
   ```
   
If a GUI interface appears, the environment is correctly set up and functional.

### 🛠️ Manual Installation by Requirements

For custom setups, you might prefer to install each requirement individually. Below is the list of packages that need to be installed:

```
opencv-python~=4.1.2.30
numpy~=1.18.1
tifffile~=2019.7.26.2
Pillow~=5.3.0
PyQt5~=5.14.1
SimpleITK~=1.2.0rc2.dev1166+ga27d6
torch~=1.4.0
PyYAML~=3.13
torchvision~=0.5.0
```


**Note**: ['SimpleElastix'](https://github.com/SuperElastix/SimpleElastix) must be installed separately. Instructions can be found [here](https://simpleelastix.readthedocs.io/GettingStarted.html#Windows). I initially used an earlier version of SimpleElastix; however, experimenting with the latest version may yield better results.

# 🔍 Usage
Detailed usage instructions will be updated soon. Stay tuned for comprehensive guidance on how to leverage this project effectively.
