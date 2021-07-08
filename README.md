# DroidRacingChallenge 2021

## Teammates:

- **Cameron Coombe**: Machine Learning & Computer Vision
- **Callum Hays**: AI & ROS Integration
- **Phoenix Seybold**: Electrical, Mechanical & Manufacturing

## Setting Up

```bash
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add - && \
sudo apt update && \
sudo apt install -y ros-melodic-desktop-full python3.7 python3.7-dev python3-pip python3.7-venv python3-yaml && \

source /opt/ros/melodic/setup.sh && \
rosdep update && \
rosdep install --from-paths ./src/ --ignore-src -y && \

source ./venv/bin/activate && \
pip install --upgrade pip && \
python3 -m pip install --force-reinstall numpy Cython && \
python3 -m pip install -r requirements.txt && \

export ROS_PYTHON_VERSION=3
```

If you get a "fatal error: xlocale.h: No such file or directory" (happened to me on an arm64 device), run this:
```bash
sudo ln -s /usr/include/locale.h /usr/include/xlocale.h
```

Then build it:

```bash
catkin_make
```

## Jetson Nano

You need to ensure that the PWM outputs are enabled. To do this (on the latest supported image based on 18.04), run:
```bash
sudo /opt/nvidia/jetson-io/jetson-io.py
```

**NOTE**: You will have to **RESTART** to do this, so be prepared to.

Select "Configure 40-pin expansion header" and then toggle both pwm pins. Then ensure you "Save and reboot to reconfigure pins".