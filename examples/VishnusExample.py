import visii
import noise
import random
import numpy as np 

opt = lambda: None
opt.spp = 400 
opt.width = 1920
opt.height = 1080 
opt.noise = False
opt.out = '15_camera_motion_car_blur.png'
opt.control = True

# # # # # # # # # # # # # # # # # # # # # # # # #
visii.initialize()
visii.set_dome_light_intensity(.8)
visii.resize_window(int(opt.width), int(opt.height))
# # # # # # # # # # # # # # # # # # # # # # # # #

# load the textures
dome = visii.texture.create_from_file("dome", "content/teatro_massimo_2k.hdr")

# we can add HDR images to act as dome
visii.set_dome_light_texture(dome)
visii.set_dome_light_rotation(visii.angleAxis(visii.pi() * .5, visii.vec3(0, 0, 1)))

car_speed = 0
car_speed_x = car_speed
car_speed_y = -2 * car_speed

camera_height = 80
# # # # # # # # # # # # # # # # # # # # # # # # #

if not opt.noise is True: 
    visii.enable_denoiser()

camera = visii.entity.create(
    name = "camera",
    transform = visii.transform.create("camera"),
    camera = visii.camera.create(
        name = "camera", 
        aspect = float(opt.width)/float(opt.height)
    )
)