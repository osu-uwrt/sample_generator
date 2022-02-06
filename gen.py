from numpy.core.defchararray import array
import visii
import json
import glob
import random
import numpy as np
from tqdm import tqdm
from squaternion import Quaternion
import cv2
import shutil
import os
from multiprocessing import Pool
from math import ceil

import colorsys



opt = lambda: None
opt.spp = 50                  # How many rendering iterations per frame
opt.width = 512                 # Size of frames
opt.height = 512 
opt.debug = False               # Output corners of the cuboid
opt.nb_frames = 3          # Number of frames
# opt.out = "/mnt/Data/visii_data/gman/" # Where to output data
opt.out = "./temp/" # Where to output data
opt.entity = "gman"           # Name of entity to output
opt.model = "./models/gman.obj" # Path to object file
opt.test_percent = 5             # percent chance frame data exports to test folder
opt.texture = './models/gman.PNG' # path to the object texture


opt.nb_objs = 10 
# # # # # # # # # # # # # # # # # # # # # # # # #

#visii.mesh.create_sphere('rm_0')



# converts 360 degree angles to Quaternions
def euler_to_quaternion(yaw, pitch, roll):

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]

# Function to add cuboid information to an object using 
def add_cuboid(name, debug=opt.debug):
    """
    Add cuboid children to the transform tree to a given object for exporting

    :param name: string name of the visii entity to add a cuboid
    :param debug:   bool - add sphere on the visii entity to make sure the  
                    cuboid is located at the right place. 

    :return: return a list of cuboid in canonical space of the object. 
    """

    obj = visii.entity.get(name)

    min_obj = obj.get_mesh().get_min_aabb_corner()
    max_obj = obj.get_mesh().get_max_aabb_corner()
    centroid_obj = obj.get_mesh().get_aabb_center()

    cuboid = [
        visii.vec3(max_obj[0], max_obj[1], max_obj[2]),
        visii.vec3(min_obj[0], max_obj[1], max_obj[2]),
        visii.vec3(max_obj[0], min_obj[1], max_obj[2]),
        visii.vec3(max_obj[0], max_obj[1], min_obj[2]),
        visii.vec3(min_obj[0], min_obj[1], max_obj[2]),
        visii.vec3(max_obj[0], min_obj[1], min_obj[2]),
        visii.vec3(min_obj[0], max_obj[1], min_obj[2]),
        visii.vec3(min_obj[0], min_obj[1], min_obj[2]),
        visii.vec3(centroid_obj[0], centroid_obj[1], centroid_obj[2]), 
    ]

    dimensions_dict = {
        'width': max_obj[0] - min_obj[0],
        'height': max_obj[1] - min_obj[1],
        'length': max_obj[2] - min_obj[2]
    }


    with open(opt.out + "dimensions.json", "w+") as fp:
        json.dump(dimensions_dict, fp, indent=4, sort_keys=True)

    # change the ids to be like ndds / DOPE
    cuboid = [  cuboid[2],cuboid[0],cuboid[3],
                cuboid[5],cuboid[4],cuboid[1],
                cuboid[6],cuboid[7],cuboid[-1]]

    cuboid.append(visii.vec3(centroid_obj[0], centroid_obj[1], centroid_obj[2]))
        
    for i_p, p in enumerate(cuboid):
        child_transform = visii.transform.create(f"{name}_cuboid_{i_p}")
        child_transform.set_position(p)
        child_transform.set_scale(visii.vec3(0.1))
        child_transform.set_parent(obj.get_transform())            
        if debug: 
            visii.entity.create(
                name = f"{name}_cuboid_{i_p}",
                mesh = visii.mesh.create_sphere(f"{name}_cuboid_{i_p}"),
                transform = child_transform, 
                material = visii.material.create(f"{name}_cuboid_{i_p}")
            )
    
    for i_v, v in enumerate(cuboid):
        cuboid[i_v]=[v[0], v[1], v[2]]

    return cuboid

def get_cuboid_image_space(obj_id, camera_name = 'camera'):
    """
    reproject the 3d points into the image space for a given object. 
    It assumes you already added the cuboid to the object 

    :obj_id: string for the name of the object of interest
    :camera_name: string representing the camera name in visii

    :return: cubdoid + centroid projected to the image, values [0..1]
    """

    cam_matrix = visii.entity.get(camera_name).get_transform().get_world_to_local_matrix()
    cam_proj_matrix = visii.entity.get(camera_name).get_camera().get_projection()

    points = []
    points_cam = []
    for i_t in range(9):
        trans = visii.transform.get(f"{obj_id}_cuboid_{i_t}")
        mat_trans = trans.get_local_to_world_matrix()
        pos_m = visii.vec4(
            mat_trans[3][0],
            mat_trans[3][1],
            mat_trans[3][2],
            1)
        
        p_cam = cam_matrix * pos_m 

        p_image = cam_proj_matrix * (cam_matrix * pos_m) 
        p_image = visii.vec2(p_image) / p_image.w
        p_image = p_image * visii.vec2(1,-1)
        p_image = (p_image + visii.vec2(1,1)) * 0.5

        points.append([p_image[0],p_image[1]])
        points_cam.append([p_cam[0],p_cam[1],p_cam[2]])
    return points, points_cam


# function to export meta data about the scene and about the objects 
# of interest. Everything gets saved into a json file.
def export_to_ndds_file(
    filename = "tmp.json", #this has to include path as well
    obj_names = [], # this is a list of ids to load and export
    height = 500, 
    width = 500,
    camera_name = 'camera',
    camera_struct = None,
    visibility_percentage = False, 
    ):
    """
    Method that exports the meta data like NDDS. This includes all the scene information in one 
    scene. 

    :filename: string for the json file you want to export, you have to include the extension
    :obj_names: [string] each entry is a visii entity that has the cuboids attached to, these
                are the objects that are going to be exported. 
    :height: int height of the image size 
    :width: int width of the image size 
    :camera_name: string for the camera name visii entity
    :camera_struct: dictionary of the camera look at information. Expecting the following 
                    entries: 'at','eye','up'. All three has to be floating arrays of three entries.
                    This is an optional export. 
    :visibility_percentage: bool if you want to export the visibility percentage of the object. 
                            Careful this can be costly on a scene with a lot of objects. 

    :return nothing: 
    """


    import simplejson as json

    # assume we only use the view camera
    cam_matrix = visii.entity.get(camera_name).get_transform().get_world_to_local_matrix()
    
    cam_matrix_export = []
    for row in cam_matrix:
        cam_matrix_export.append([row[0],row[1],row[2],row[3]])
    
    cam_world_location = visii.entity.get(camera_name).get_transform().get_position()
    cam_world_quaternion = visii.entity.get(camera_name).get_transform().get_rotation()
    # cam_world_quaternion = visii.quat_cast(cam_matrix)

    cam_intrinsics = visii.entity.get(camera_name).get_camera().get_intrinsic_matrix(width, height)

    if camera_struct is None:
        camera_struct = {
            'at': [0,0,0,],
            'eye': [0,0,0,],
            'up': [0,0,0,]
        }

    dict_out = {
                "camera_data" : {
                    "width" : width,
                    'height' : height,
                    'camera_look_at':
                    {
                        'at': [
                            camera_struct['at'][0],
                            camera_struct['at'][1],
                            camera_struct['at'][2],
                        ],
                        'eye': [
                            camera_struct['eye'][0],
                            camera_struct['eye'][1],
                            camera_struct['eye'][2],
                        ],
                        'up': [
                            camera_struct['up'][0],
                            camera_struct['up'][1],
                            camera_struct['up'][2],
                        ]
                    },
                    'camera_view_matrix':cam_matrix_export,
                    'location_world':
                    [
                        cam_world_location[0],
                        cam_world_location[1],
                        cam_world_location[2],
                    ],
                    'quaternion_world_xyzw':[
                        cam_world_quaternion[0],
                        cam_world_quaternion[1],
                        cam_world_quaternion[2],
                        cam_world_quaternion[3],
                    ],
                    'intrinsics':{
                        'fx':cam_intrinsics[0][0],
                        'fy':cam_intrinsics[1][1],
                        'cx':cam_intrinsics[2][0],
                        'cy':cam_intrinsics[2][1]
                    }
                }, 
                "objects" : []
            }

    # Segmentation id to export
    id_keys_map = visii.entity.get_name_to_id_map()

    for obj_name in obj_names: 

        projected_keypoints, _ = get_cuboid_image_space(obj_name, camera_name=camera_name)

        # put them in the image space. 
        for i_p, p in enumerate(projected_keypoints):
            projected_keypoints[i_p] = [p[0]*width, p[1]*height]

        # Get the location and rotation of the object in the camera frame 

        trans = visii.transform.get(obj_name)
        quaternion_xyzw = visii.inverse(cam_world_quaternion) * trans.get_rotation()

        object_world = visii.vec4(
            trans.get_position()[0],
            trans.get_position()[1],
            trans.get_position()[2],
            1
        ) 
        pos_camera_frame = cam_matrix * object_world

        #check if the object is visible
        visibility = -1
        bounding_box = [-1,-1,-1,-1]

        segmentation_mask = visii.render_data(
            width=int(width), 
            height=int(height), 
            start_frame=0,
            frame_count=1,
            bounce=int(0),
            options="entity_id",
        )
        segmentation_mask = np.array(segmentation_mask).reshape(width,height,4)[:,:,0]
        
        if visibility_percentage == True and int(id_keys_map [obj_name]) in np.unique(segmentation_mask.astype(int)): 
            transforms_to_keep = {}
            
            for name in id_keys_map.keys():
                if 'camera' in name.lower() or obj_name in name:
                    continue
                trans_to_keep = visii.entity.get(name).get_transform()
                transforms_to_keep[name]=trans_to_keep
                visii.entity.get(name).clear_transform()

            # Percentage visibility through full segmentation mask. 
            segmentation_unique_mask = visii.render_data(
                width=int(width), 
                height=int(height), 
                start_frame=0,
                frame_count=1,
                bounce=int(0),
                options="entity_id",
            )

            segmentation_unique_mask = np.array(segmentation_unique_mask).reshape(width,height,4)[:,:,0]

            values_segmentation = np.where(segmentation_mask == int(id_keys_map[obj_name]))[0]
            values_segmentation_full = np.where(segmentation_unique_mask == int(id_keys_map[obj_name]))[0]
            visibility = len(values_segmentation)/float(len(values_segmentation_full))
            
            # set back the objects from remove
            for entity_name in transforms_to_keep.keys():
                visii.entity.get(entity_name).set_transform(transforms_to_keep[entity_name])
        else:

            if int(id_keys_map[obj_name]) in np.unique(segmentation_mask.astype(int)): 
                #
                visibility = 1
                y,x = np.where(segmentation_mask == int(id_keys_map[obj_name]))
                bounding_box = [int(min(x)),int(max(x)),height-int(max(y)),height-int(min(y))]
            else:
                visibility = 0

        # Final export
        dict_out['objects'].append({
            'class':obj_name.split('_')[0],
            'name':obj_name,
            'provenance':'visii',
            # TODO check the location
            'location': [
                pos_camera_frame[0],
                pos_camera_frame[1],
                pos_camera_frame[2]
            ],
            'quaternion_xyzw':[
                quaternion_xyzw[0],
                quaternion_xyzw[1],
                quaternion_xyzw[2],
                quaternion_xyzw[3],
            ],
            'quaternion_xyzw_world':[
                trans.get_rotation()[0],
                trans.get_rotation()[1],
                trans.get_rotation()[2],
                trans.get_rotation()[3]
            ],


            'projected_cuboid': projected_keypoints[0:8],
            'projected_cuboid_centroid': projected_keypoints[8],
            'segmentation_id':id_keys_map[obj_name],
            'visibility_image':visibility,
            'bounding_box': {
                'top_left':[
                    bounding_box[0],
                    bounding_box[2],
                ], 
                'bottom_right':[
                    bounding_box[1],
                    bounding_box[3],
                ],
                


            },

        })
        
    with open(filename, 'w+') as fp:
        json.dump(dict_out, fp, indent=4, sort_keys=True)




def add_random_obj(name_r = "name", pos=[float]):
    # this function adds a random object that uses one of the pre-loaded mesh
    # components, assigning a random pose and random material to that object.

    

    obj_random = visii.entity.create(
        name = name_r,
        transform = visii.transform.create(name_r),
        material = visii.material.create(name_r)
    )

    mesh_id = random.randint(0,15)

    # set the mesh. (Note that meshes can be shared, saving memory)
    mesh_r = visii.mesh.get(f'rm_{mesh_id}')
    obj_random.set_mesh(mesh_r)

    # obj_random.get_transform().set_position((
    #     random.uniform(-25,10),
    #     random.uniform(-1,1),
    #     random.uniform(-1,1)
    # ))

    x_not =pos[0]
    y_not=pos[1]
    z_not=pos[2]

    
   
    position = [
        random.uniform(-.25,10),
        random.uniform(-1,1),
        random.uniform(-1,1)
  

    ]

    # Scale the position based on depth into image to make sure it remains in frame
    position[1] *= position[0]
    position[2] *= position[0]

    obj_random.get_transform().set_position(tuple(position))

    obj_random.get_transform().set_rotation((
        random.uniform(0,1), # X 
        random.uniform(0,1), # Y
        random.uniform(0,1), # Z
        random.uniform(0,1)  # W
    ))

    s = random.uniform(0.15, 0.2)
    obj_random.get_transform().set_scale((
        s,s,s
    ))  

    rgb = colorsys.hsv_to_rgb(
        random.uniform(0,1),
        random.uniform(0.7,1),
        random.uniform(0.7,1)
    )

    obj_random.get_material().set_base_color(rgb)

    mat = obj_random.get_material()
    
    # Some logic to generate "natural" random materials
    material_type = random.randint(0,2)
    
    # Glossy / Matte Plastic
    if material_type == 0:  
        if random.randint(0,2): mat.set_roughness(random.uniform(.9, 1))
        else           : mat.set_roughness(random.uniform(.0,.1))
    
    # Metallic
    if material_type == 1:  
        mat.set_metallic(random.uniform(0.9,1))
        if random.randint(0,2): mat.set_roughness(random.uniform(.9, 1))
        else           : mat.set_roughness(random.uniform(.0,.1))
    
    # Glass
    if material_type == 2:  
        mat.set_transmission(random.uniform(0.9,1))
        
        # controls outside roughness
        if random.randint(0,2): mat.set_roughness(random.uniform(.9, 1))
        else           : mat.set_roughness(random.uniform(.0,.1))
        
        # controls inside roughness
        if random.randint(0,2): mat.set_transmission_roughness(random.uniform(.9, 1))
        else           : mat.set_transmission_roughness(random.uniform(.0,.1))

    mat.set_sheen(random.uniform(0,1)) # <- soft velvet like reflection near edges
    mat.set_clearcoat(random.uniform(0,1)) # <- Extra, white, shiny layer. Good for car paint.    
    if random.randint(0,1): mat.set_anisotropic(random.uniform(0.9,1)) # elongates highlights 

    

    # (lots of other material parameters are listed in the docs)






# # # # # # # # # # # # # # # # # # # # # # # # #



def make_background(image):
    # Shrink negative to 400x400
    SIZE = 400
    image = cv2.resize(image, (SIZE, SIZE))

    # Make background image that will wrap over dome
    WIDTH = 1600
    HEIGHT = 800
    background = np.ones((HEIGHT, WIDTH, 3))

    # Randomize the color and brightness
    average_color = np.mean(image, axis=(0, 1)) 
    average_color *= np.random.uniform(.8, 1.2, size=3)
    desired_brightness = np.random.uniform(84, 192)

    # Fill in the empty space with the average color of the negative image
    background *= average_color
    # Normalize the color to be the set brightness
    background *= desired_brightness / (np.mean(image)+.0001)
    # Drop our negative in the center
    background[
        HEIGHT // 2 - SIZE // 2 : HEIGHT // 2 + SIZE // 2,
        WIDTH // 2 - SIZE // 2 : WIDTH // 2 + SIZE // 2
    ] = image

    return background.astype(np.uint8)
            
# take list of frame ids and makes a frame in visii with that frame id
def f(frame_ids):
    # headless - no window
    # verbose - output number of frames rendered, etc..
    visii.initialize(headless = True, verbose = False)

   
    # Use a neural network to denoise ray traced
    visii.enable_denoiser()

    # set up dome background
    negatives = list(glob.glob("negatives/*.jpg"))
    visii.set_dome_light_intensity(1)

    # create an entity that will serve as our camera.
    camera = visii.entity.create(name = "camera")

    # To place the camera into our scene, we'll add a "transform" component.
    # (All visii objects have a "name" that can be used for easy lookup later.)
    camera.set_transform(visii.transform.create(name = "camera_transform"))

    # To make our camera entity act like a "camera", we'll add a camera component
    camera.set_camera(
        visii.camera.create_from_fov(
            name = "camera_camera", 
            field_of_view = 1.4, # note, this is in radians
            aspect = opt.width / float(opt.height)
        )
    )

    # Finally, we'll select this entity to be the current camera entity.
    # (visii can only use one camera at the time)
    visii.set_camera_entity(camera)


    #First lets pre-load some mesh components.
    visii.mesh.create_sphere('rm_0')
    visii.mesh.create_torus_knot('rm_1')
    visii.mesh.create_teapotahedron('rm_2') 
    visii.mesh.create_box('rm_3')
    visii.mesh.create_capped_cone('rm_4')    
    visii.mesh.create_capped_cylinder('rm_5')
    visii.mesh.create_capsule('rm_6')
    visii.mesh.create_cylinder('rm_7')
    visii.mesh.create_disk('rm_8')  
    visii.mesh.create_dodecahedron('rm_9')
    visii.mesh.create_icosahedron('rm_10')
    visii.mesh.create_icosphere('rm_11')
    visii.mesh.create_rounded_box('rm_12')
    visii.mesh.create_spring('rm_13')
    visii.mesh.create_torus('rm_14')
    visii.mesh.create_tube('rm_15')

    # lets store the camera look at information so we can export it
    camera_struct_look_at = {
        'at':[0, 0, 0],
        'up':[0, 0, 1],
        'eye':[-1, 0, 0]
    }

    # Lets set the camera to look at an object. 
    # We'll do this by editing the transform component.
    camera.get_transform().look_at(
        at = camera_struct_look_at['at'],
        up = camera_struct_look_at['up'],
        eye = camera_struct_look_at['eye']
    )


    # This function loads a mesh ignoring .mtl
    global mesh 
    mesh = visii.mesh.create_from_file(opt.entity, opt.model)

    # creates visii entity using loaded mesh
    obj_entity = visii.entity.create(
        name= opt.entity + "_entity",
        mesh = mesh,
        transform = visii.transform.create(opt.entity + "_entity"),
        material = visii.material.create(opt.entity + "_entity"),
    )

    # obj_entity.get_light().set_intensity(0.05)

    # you can also set the light color manually
    # obj_entity.get_light().set_color((1,0,0))

    # Add texture to the material
    material = visii.material.get(opt.entity + "_entity")
    texture = visii.texture.create_from_file(opt.entity, opt.texture)
    material.set_base_color_texture(texture)

    # Lets add the cuboid to the object we want to export
    add_cuboid(opt.entity + "_entity", opt.debug)

    # lets keep track of the entities we want to export
    entities_to_export = [opt.entity + "_entity"]

    # Loop where we change and render each frame
    for i in tqdm(frame_ids):
        # load a random negtive onto the dome
        negative = cv2.imread(random.choice(negatives))

        # Skip dark backgrounds (20/255)
        if np.mean(negative) < 20:
            continue

        # Fix lighting of background and make it small within the FOV
        background = make_background(negative)
        cv2.imwrite("test" + str(i) + ".png", background)
        dome = visii.texture.create_from_file("dome", "test" + str(i) + ".png")
        visii.set_dome_light_texture(dome)
        visii.set_dome_light_rotation(visii.angleAxis(visii.pi() * .5, visii.vec3(0, 0, 1)))

        stretch_factor = 2
        scale = [
            random.uniform(1,stretch_factor),  # width
            random.uniform(1,stretch_factor),   # length
            random.uniform(1,stretch_factor)   # height 
        ]
        obj_entity.get_transform().set_scale(scale)

        # create random rotation while making usre the entity is facing forward in each frame
        rot = [
            random.uniform(0, 359), # Roll
            random.uniform(-35, 35), # Pitch
            random.uniform(-60,60) # Yaw
        ]
        q = Quaternion.from_euler(rot[0], rot[1], rot[2], degrees=True)
        # q = Quaternion.from_euler(0, 0, 0, degrees=True)

        position = [
            random.uniform(-.25, 10), # X Depth
            random.uniform(-1, 1),# Y 
            random.uniform(-1, 1) # Z
        ]
        # Scale the position based on depth into image to make sure it remains in frame
        position[1] *= position[0]
        position[2] *= position[0]
        
        obj_entity.get_transform().set_position(tuple(position))


        obj_entity.get_transform().set_rotation((
            q.x, q.y, q.z, q.w       
        ))

        # TODO: Populate random shpaes in the world (We might have to check and make sure we are not generating a shape inside of gman or bootlegger)

       
        for k in range(3):
            add_random_obj(str(k), position)
            print("\rcreating random object ", 0, end="")
        #add_random_obj(str(1),position)
        #print("\rcreating random object ", 1, end="")
        #print(" - done!")

        
       

        # use random to make 95 % probability the frame data goes into training and
        # 5% chance it goes in test folder
        folder = ''
        if random.randint(0,100) < opt.test_percent:
            folder = opt.entity + '_test/'
        else:
            folder = opt.entity + '_training/'

        # Render the scene
        visii.render_to_file(
            width = opt.width, 
            height = opt.height, 
            samples_per_pixel = opt.spp,   
            file_path = opt.out + folder + opt.entity + str(i) + '.png'
        )

        # set up JSON    
        export_to_ndds_file(
            filename = opt.out + folder + opt.entity + str(i) + '.json',
            obj_names = entities_to_export,
            width=opt.width, 
            height=opt.height, 
            camera_struct = camera_struct_look_at
        )

        # remove current negative from the dome
        visii.clear_dome_light_texture()
        visii.texture.remove("dome")

        # TODO: Remove all the random shapes that just got loaded in

        os.remove("test" + str(i) + ".png")

    visii.deinitialize()

if __name__ == "__main__":
    # clear the contents of training and test folders
    dir_path = opt.out + opt.entity + "_test"

    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))

    try:
        os.mkdir(dir_path)
    except OSError:
        print ("Creation of the directory %s failed" % path)

    dir_path = opt.out + opt.entity + "_training"

    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print("Error: %s : %s" % (dir_path, e.strerror))

    try:
        os.mkdir(dir_path)
    except OSError:
        print ("Creation of the directory %s failed" % path)

    # Start processes

    l = list(range(opt.nb_frames))
    nb_process = 6 
    
    # How many frames each process should have 
    n = ceil(len(l) / nb_process)
    
    # divide frames between processes
    frame_groups = [l[i:i + n] for i in range(0, len(l), n)]

    with Pool(nb_process) as p:
        p.map(f,frame_groups)
