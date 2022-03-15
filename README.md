# DroidRacingChallenge2021

Team Chimera:
- Callum Hays
- Cameron Coombe
- Phoenix Seybold

## Setup

Make sure the pinouts coming from the redboard are plugged into the drive ESC and steering Servo. (blue should go into the slightly longer lead, orange to the other. Both reds and blacks are just 5V and GND as per usual).

Log into the Jetson somehow. Via HDMI: Connect the HDMI cable and restart the jetson. A login terminal prompt should show. The login credentials are:

```
username: champ
password: grindyboy
```

The program relies on being connected to a Wifi network. To edit the wifi network config, run `sudo vim /etc/NetworkManager/system-connections` and then `sudo service NetworkManager restart` to load the config. If successful, running `ip address` and looking at the last one (`wlan0`) will show your IP address on the network (if you're connected).


## Running

Get the wifi address of the robot by running `ip address` and looking for `wlan0`.

Run the program with:

```
cd ~/DroidRacingChallenge2021
zsh run.sh
```

You should then be able to access the web interface at `<IP ADDR>:8080`. After a little while the interface should show with live-streams coming from the robot. The wheels should be turning as it detects various obstacles.

You can tune any parameter in the algorithm by modifying the sliders/checkboxes/dropdowns on the left.

The robot starts with the drives turned off. To turn them on, check the `diffsteer.go` checkbox in the parameter pane on the left.

Now the robot should be running. You may need to tune the thresholding values in order for the bot to recognize lines and obstacles properly. Good luck!
