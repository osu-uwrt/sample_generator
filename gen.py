# 01.simple_scene.py
#
# This shows a sphere on top of a mirror floor. 
# You should see how to set up a simple scene with visii, where light is
# provided by the dome.

import visii
from random import *

opt = lambda: None
opt.spp = 50 
opt.width = 512
opt.height = 512 
opt.nb_frames = 10
opt.out = '01_simple_scene.png'

# headless - no window
# verbose - output number of frames rendered, etc..
visii.initialize(headless = True, verbose = True)

# Use a neural network to denoise ray traced
visii.enable_denoiser()

# First, lets create an entity that will serve as our camera.
camera = visii.entity.create(name = "camera")

# To place the camera into our scene, we'll add a "transform" component.
# (All visii objects have a "name" that can be used for easy lookup later.)
camera.set_transform(visii.transform.create(name = "camera_transform"))

# To make our camera entity act like a "camera", we'll add a camera component
camera.set_camera(
    visii.camera.create_from_fov(
        name = "camera_camera", 
        field_of_view = 0.785398, # note, this is in radians
        aspect = opt.width / float(opt.height)
    )
)

# Finally, we'll select this entity to be the current camera entity.
# (visii can only use one camera at the time)
visii.set_camera_entity(camera)

# Lets set the camera to look at an object. 
# We'll do this by editing the transform component.
camera.get_transform().look_at(at = (0, 0, .9), up = (0, 0, 1), eye = (0, 5, 1))

# Next, lets at an object (a floor).
floor = visii.entity.create(
    name = "floor",
    mesh = visii.mesh.create_plane("mesh_floor"),
    transform = visii.transform.create("transform_floor"),
    material = visii.material.create("material_floor")
)

# This function loads a signle obj mesh. It ignores 
# the associated .mtl file
mesh = visii.mesh.create_from_file("obj", "./garlic.obj")

obj_entity = visii.entity.create(
    name="obj_entity",
    mesh = mesh,
    transform = visii.transform.create("obj_entity"),
    material = visii.material.create("obj_entity")
)

# lets set the obj_entity up
obj_entity.get_transform().set_rotation( 
    (0.7071, 0, 0, 0.7071)
)
obj_entity.get_material().set_base_color(
    (0.9,0.12,0.08)
)  
obj_entity.get_material().set_roughness(0.7)   
obj_entity.get_material().set_specular(1)   
obj_entity.get_material().set_sheen(1)

for i in range(opt.nb_frames):
    obj_entity.get_transform().set_position((
        uniform(-5,5),
        uniform(0,-60),
        uniform(-1,3)
    ))
    obj_entity.get_transform().set_rotation((
        uniform(0,1), # X 
        uniform(0,1), # Y
        uniform(0,1), # Z
        uniform(0,1)  # W
    ))
    # Now that we have a simple scene, let's render it 
    visii.render_to_file(
        width = opt.width, 
        height = opt.height, 
        samples_per_pixel = opt.spp,   
        file_path = 'data/garlic' + str(i) + '.png'
    )

visii.deinitialize()
