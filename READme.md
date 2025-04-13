# Live Control for Go1 in Mujoco Playground

This repository enables you to load a pre-trained policy for the Unitree Go1's locomotion joystick enviornment and activate a live simulation of the robot using either a controller or a keyboard. This project is entirely an extension of google-deepmind's **Mujoco Playground**.

---

## Setup Instructions

**1. Follow the Instructions on Mujoco Playground**  
Visit the [Mujoco Playground GitHub page](https://github.com/google-deepmind/mujoco_playground/) and follow the setup instructions.

**2. Clone the Live Control Repository and Change to `live_control`**  
After setting up Mujoco Playground, navigate to its main directory and clone this repository:

**3. Brax==0.12.1**
Downgrade Brax if needed `pip install brax==0.12.1`

```bash
cd path/to/mujoco_playground
git clone https://github.com/yourusername/live_control_go1.git
```

**4. Mac Compatibility:**
See bottom

## To load the trained policy and start the live viewer:

**a. To run live control with a connected controller:**

```bash
python controller_live_control.py --env_name=Go1JoystickFlatTerrain --load_checkpoint_path=load_me/checkpoints --play_only
```

**b. To run live control with a connected keyboard:**

```bash
python keyboard_live_control.py --env_name=Go1JoystickFlatTerrain --load_checkpoint_path=load_me/checkpoints --play_only
```

_using a controller is recommended as Mujoco's keybinds intefere with keyboard controls!!_

## NOTES:

**1.** The pre-trained policy is saved in load_me directory

**2.** Go1JoystickFlatTerrain loads this xml: /mujoco_playground/mujoco_playground/\_src/locomotion/go1/xmls/scene_mjx_feetonly_flat_terrain.xml

**3.** The rough terrain works for the same policy, to use simply change "--env_name=Go1JoystickFlatTerrain" to "--env_name=Go1JoystickRoughTerrain" in initial call

**4a.** To use controller, make sure to `pip install pygame` in your environment
**4b.** On Mac, to use controller, make sure to `pip install pysdl2` in your environment

## For Mac Users

Make sure to have PySDL2 installed, pygame will _NOT_ work!

On macOS, use `mjpython` instead of `python` to run the simulation.

Also, use the checkpoint directory designated for Mac (e.g., `mac_load_me/checkpoints`).

## To load the trained policy and start the live viewer on Mac:

**a. To run live control with a connected controller:**

```bash
mjpython mac_controller_live_control.py --env_name=Go1JoystickFlatTerrain --load_checkpoint_path=mac_load_me/checkpoints --play_only
```

**b. To run live control with a connected keyboard:**

```bash
mjpython keyboard_live_control.py --env_name=Go1JoystickFlatTerrain --load_checkpoint_path=mac_load_me/checkpoints --play_only
```

_using a controller is recommended as Mujoco's keybinds intefere with keyboard controls!!_
