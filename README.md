# DroidRacingChallenge 2021

## Teammates:

- **Cameron Coombe**: Machine Learning & Computer Vision
- **Callum Hays**: AI & ROS Integration
- **Phoenix Seybold**: Electrical, Mechanical & Manufacturing

## Setting Up

```bash
sudo apt install ros-noetic-desktop-full
pip install -r requirements.txt
rosdep update
rosdep install --from-paths ./src/ --ignore-src
catin_make
```
