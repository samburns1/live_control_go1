# Live Control for Go1 in Mujoco Playground

Welcome to the Live Control repository! This repository enables you to load a trained policy and start a live simulation of the Go1 robot using either a controller or a keyboard.

---

## Setup Instructions

**1. Follow the Instructions on Mujoco Playground**  
Visit the [Mujoco Playground GitHub page](https://github.com/google-deepmind/mujoco_playground/) and follow the setup instructions.

**2. Clone the Live Control Repository**  
After setting up Mujoco Playground, navigate to its main directory and clone this repository:

```bash
cd path/to/mujoco_playground
git clone https://github.com/yourusername/live_control_go1.git
```


# To load the trained policy and start the live viewer:

**a. Change to ```live_control```**  
```bash
cd path/to/mujoco_playground
git clone https://github.com/yourusername/live_control_go1.git
```

2. To run live control with a connected controller:
python controller_live_control.py --env_name=Go1JoystickFlatTerrain --load_checkpoint_path=load_me/checkpoints --play_only

3. To run live control with a connected keyboard:
python keyboard_live_control.py --env_name=Go1JoystickFlatTerrain --load_checkpoint_path=load_me/checkpoints --play_only

NOTES:
1. the pre-trained policy is saved in load_me directory
2. Go1JoystickFlatTerrain loads this xml: /mujoco_playground/mujoco_playground/_src/locomotion/go1/xmls/scene_mjx_feetonly_flat_terrain.xml
3. The rough terrain works for the same policy, to use simply change "--env_name=Go1JoystickFlatTerrain" to "--env_name=Go1JoystickRoughTerrain" in initial call
4. to use controller, make sure to pip install pygame