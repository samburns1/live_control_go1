to load the trained policy and start the live viewer:

1. cd in live_control

2. To run live control with a connected controller:
python controller_live_control.py --env_name=Go1JoystickFlatTerrain --load_checkpoint_path=load_me/checkpoints --play_only

3. To run live control with a connected keyboard:
python keyboard_live_control.py --env_name=Go1JoystickFlatTerrain --load_checkpoint_path=load_me/checkpoints --play_only

NOTES:
1. the pre-trained policy is saved in load_me directory
2. Go1JoystickFlatTerrain loads this xml: /mujoco_playground/mujoco_playground/_src/locomotion/go1/xmls/scene_mjx_feetonly_flat_terrain.xml
3. The rough terrain works for the same policy, to use simply change "--env_name=Go1JoystickFlatTerrain" to "--env_name=Go1JoystickRoughTerrain" in initial call