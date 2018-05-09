#!/usr/bin/env python3
"""
Python OpenGL practical application.
"""
# Python built-in modules
import os                           # os function, i.e. checking file status
import sys

# External, non built-in modules
import OpenGL.GL as GL              # standard Python OpenGL wrapper
import glfw                         # lean window system wrapper for OpenGL
import numpy as np
from random import random               # all matrix manipulations & OpenGL args
from transform import translate, rotate, scale, vec
from transform import frustum, perspective
from transform import Trackball, identity
import pyassimp                     # 3D ressource loader
import pyassimp.errors              # assimp error management + exceptions

from PIL import Image               # load images for textures
from itertools import cycle

from transform import lerp, vec, sincos
from transform import (quaternion_slerp, quaternion_matrix, quaternion,
                       quaternion_from_euler)

import bisect
from bisect import bisect_left      # search sorted keyframe lists

from DiamondSquare import diamondSquare

# ------------ low level OpenGL object wrappers ----------------------------
class Shader:
    """ Helper class to create and automatically destroy shader program """
    @staticmethod
    def _compile_shader(src, shader_type):
        src = open(src, 'r').read() if os.path.exists(src) else src
        src = src.decode('ascii') if isinstance(src, bytes) else src
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, src)
        GL.glCompileShader(shader)
        status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
        src = ('%3d: %s' % (i+1, l) for i, l in enumerate(src.splitlines()))
        if not status:
            log = GL.glGetShaderInfoLog(shader).decode('ascii')
            GL.glDeleteShader(shader)
            src = '\n'.join(src)
            #print('Compile failed for %s\n%s\n%s' % (shader_type, log, src))
            return None
        return shader

    def __init__(self, vertex_source, fragment_source):
        """ Shader can be initialized with raw strings or source file names """
        self.glid = None
        vert = self._compile_shader(vertex_source, GL.GL_VERTEX_SHADER)
        frag = self._compile_shader(fragment_source, GL.GL_FRAGMENT_SHADER)
        if vert and frag:
            self.glid = GL.glCreateProgram()  # pylint: disable=E1111
            GL.glAttachShader(self.glid, vert)
            GL.glAttachShader(self.glid, frag)
            GL.glLinkProgram(self.glid)
            GL.glDeleteShader(vert)
            GL.glDeleteShader(frag)
            status = GL.glGetProgramiv(self.glid, GL.GL_LINK_STATUS)
            if not status:
                #print(GL.glGetProgramInfoLog(self.glid).decode('ascii'))
                GL.glDeleteProgram(self.glid)
                self.glid = None

    def __del__(self):
        GL.glUseProgram(0)
        if self.glid:                      # if this is a valid shader object
            GL.glDeleteProgram(self.glid)  # object dies => destroy GL object

#perspective(fovy, aspect, near, far):
# ------------  simple color fragment shader demonstrated in Practical 1 ------
COLOR_VERT = """#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out vec3 fragColor;

void main() {
    gl_Position = projection * view * model * vec4(position, 1);
    fragColor = color;
}"""


COLOR_FRAG = """#version 330 core
in vec3 fragColor;
out vec4 outColor;
void main() {
    outColor = vec4(fragColor, 1);
}"""

# new shader for skinned meshes, fully compatible with previous color fragment
TEXT_VERT = """#version 330 core
// ---- vertex attributes
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 text_coord;

uniform mat4 modelviewprojection;
// ----- interpolated attribute variables to be passed to fragment shader
out vec2 fragTexCoord;

void main() {
    // ------ compute world and normalized eye coordinates of our vertex
    gl_Position =  modelviewprojection * vec4(position, 1);
    fragTexCoord = text_coord;
}
"""

TEXT_FRAG = """#version 330 core
uniform sampler2D diffuseMap;
in vec2 fragTexCoord;
out vec4 outColor;
void main() {
    outColor = texture(diffuseMap, fragTexCoord);
}"""


#------------- Buffer initialization -----------------------------------------

class VertexArray:
    """helper class to create and self destroy vertex array objects."""
    def __init__(self, attributes, index=None, usage=GL.GL_STATIC_DRAW):
        """ Vertex array from attributes and optional index array. Vertex
            attribs should be list of arrays with dim(0) indexed by vertex. """

        # create vertex array object, bind it
        self.glid = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.glid)
        self.buffers = []  # we will store buffers in a list
        nb_primitives, size = 0, 0

        # load a buffer per initialized vertex attribute (=dictionary)
        for loc, data in enumerate(attributes):
            if data is None:
                continue

            # bind a new vbo, upload its data to GPU, declare its size and type
            self.buffers += [GL.glGenBuffers(1)]
            data = np.array(data, np.float32, copy=False)
            nb_primitives, size = data.shape
            GL.glEnableVertexAttribArray(loc)  # activates for current vao only
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffers[-1])
            GL.glBufferData(GL.GL_ARRAY_BUFFER, data, usage)
            GL.glVertexAttribPointer(loc, size, GL.GL_FLOAT, False, 0, None)

        # optionally create and upload an index buffer for this object
        self.draw_command = GL.glDrawArrays
        self.arguments = (0, nb_primitives)
        if index is not None:
            self.buffers += [GL.glGenBuffers(1)]
            index_buffer = np.array(index, np.int32, copy=False)
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.buffers[-1])
            GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, index_buffer, usage)
            self.draw_command = GL.glDrawElements
            self.arguments = (index_buffer.size, GL.GL_UNSIGNED_INT, None)

        # cleanup and unbind so no accidental subsequent state update
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)

    def draw(self, primitive):
        """draw a vertex array, either as direct array or indexed array"""
        GL.glBindVertexArray(self.glid)
        self.draw_command(primitive, *self.arguments)
        GL.glBindVertexArray(0)

    def __del__(self):  # object dies => kill GL array and buffers from GPU
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(len(self.buffers), self.buffers)

# ------------  Ground modelling ----------------------------------------------

class sol:
    """Sol object"""

    def getIndex(self, point):
        if not self.pointIndex.__contains__(point):
            self.position.append(point)
            self.textcoord.append((point[0]/self.size, point[2]/self.size))
            self.color.append((0.25, 0.5 + point[1]/2048 + (point[1]%64)/1024, 0.25))
            # self.color.append((random(), random(), random()))
            self.pointIndex[point] = self.indexCount
            self.indexCount += 1
        return self.pointIndex[point]

    def addSquare(self, p1, p2, p3, p4):
        self.index.append(self.getIndex(p1))
        self.index.append(self.getIndex(p2))
        self.index.append(self.getIndex(p4))
        self.index.append(self.getIndex(p2))
        self.index.append(self.getIndex(p3))
        self.index.append(self.getIndex(p4))

    def __init__(self, size = 512):
        height = diamondSquare(size + 1, "ProjetGraphique3D")
        self.indexCount = 0
        self.pointIndex = dict()
        self.position = []
        self.index = []
        self.color = []
        self.textcoord = []
        self.file = "text_ground.jpg"
        self.size = size

        # interactive toggles
        self.wrap = cycle([GL.GL_REPEAT, GL.GL_MIRRORED_REPEAT,
                           GL.GL_CLAMP_TO_BORDER, GL.GL_CLAMP_TO_EDGE])
        self.filter = cycle([(GL.GL_NEAREST, GL.GL_NEAREST),
                             (GL.GL_LINEAR, GL.GL_LINEAR),
                             (GL.GL_LINEAR, GL.GL_LINEAR_MIPMAP_LINEAR)])
        self.wrap_mode, self.filter_mode = next(self.wrap), next(self.filter)

        # setup texture and upload it to GPU
        self.texture = Texture("text_ground.jpg", self.wrap_mode, *self.filter_mode)


        self.shader = Shader(TEXT_VERT, TEXT_FRAG)

        for x in range(size):
            for y in range(size):
                self.addSquare((x, height[x][y], y), (x, height[x][y + 1], y + 1), (x + 1, height[x + 1][y + 1], y + 1), (x + 1, height[x + 1][y], y))

        # triangle position buffer
        self.position = np.array(self.position, np.float32)
        self.index = np.array(self.index, np.uint32)

        self.vertex_array = VertexArray([self.position, self.textcoord], self.index)



    def draw(self, projection, view, model, win=None, **_kwargs):

        GL.glUseProgram(self.shader.glid)

        # projection geometry
        loc = GL.glGetUniformLocation(self.shader.glid, 'modelviewprojection')
        GL.glUniformMatrix4fv(loc, 1, True, projection @ view @ model)

        # texture access setups
        loc = GL.glGetUniformLocation(self.shader.glid, 'diffuseMap')
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(loc, 0)
        self.vertex_array.draw(GL.GL_TRIANGLES)

        # leave clean state for easier debugging
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)

    def __del__(self):
        GL.glDeleteVertexArrays(1, [self.glid])
        GL.glDeleteBuffers(1, self.buffers)

# ------------  Hierarchical Modeling -----------------------------------------

class Node:
    """ Scene graph transform and parameter broadcast node """
    def __init__(self, name='', children=(), transform=identity(), **param):
        self.transform, self.param, self.name = transform, param, name
        self.children = list(iter(children))

    def add(self, *drawables):
        """ Add drawables to this node, simply updating children list """
        self.children.extend(drawables)

    def draw(self, projection, view, model, **param):
        """ Recursive draw, passing down named parameters & model matrix. """
        # merge named parameters given at initialization with those given here
        param = dict(param, **self.param)
        model = model@self.transform
        for child in self.children:
            child.draw(projection, view, model, **param)

# -------------- 3D ressource loader -----------------------------------------

def load(file):
    """ load resources from file using pyassimp, return list of ColorMesh """
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scene = pyassimp.load(file, option)
    except pyassimp.errors.AssimpError:
        #print('ERROR: pyassimp unable to load', file)
        return []     # error reading => return empty list

    meshes = [ColorMesh([m.vertices, m.normals], m.faces) for m in scene.meshes]
    size = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    #print('Loaded %s\t(%d meshes, %d faces)' % (file, len(scene.meshes), size))

    pyassimp.release(scene)
    return meshes

class ColorMesh:

    def __init__(self, attributes, index=None):
        self.vertex_array = VertexArray(attributes, index)

    def draw(self, projection, view, model, color_shader=None, color=(1,1,1,1), **_kwargs):
        names = ['view', 'projection', 'model']
        loc = {n: GL.glGetUniformLocation(color_shader.glid, n) for n in names}
        GL.glUseProgram(color_shader.glid)

        GL.glUniformMatrix4fv(loc['view'], 1, True, view)
        GL.glUniformMatrix4fv(loc['projection'], 1, True, projection)
        GL.glUniformMatrix4fv(loc['model'], 1, True, model)

        self.vertex_array.draw(GL.GL_TRIANGLES)

    def __del__(self):
        del(self.vertex_array)

# -------------- OpenGL Texture Wrapper ---------------------------------------

class Texture:
    """ Helper class to create and automatically destroy textures """
    def __init__(self, file, wrap_mode=GL.GL_REPEAT, min_filter=GL.GL_LINEAR,
                 mag_filter=GL.GL_LINEAR_MIPMAP_LINEAR):
        self.glid = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.glid)
        # helper array stores texture format for every pixel size 1..4
        format = [GL.GL_LUMINANCE, GL.GL_LUMINANCE_ALPHA, GL.GL_RGB, GL.GL_RGBA]
        try:
            # imports image as a numpy array in exactly right format
            tex = np.array(Image.open(file))
            format = format[0 if len(tex.shape) == 2 else tex.shape[2] - 1]
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, tex.shape[1],
                            tex.shape[0], 0, format, GL.GL_UNSIGNED_BYTE, tex)

            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, wrap_mode)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, wrap_mode)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, min_filter)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, mag_filter)
            GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
            message = 'Loaded texture %s\t(%s, %s, %s, %s)'
            #print(message % (file, tex.shape, wrap_mode, min_filter, mag_filter))
        except FileNotFoundError:
            print("ERROR: unable to load texture file %s" % file)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def __del__(self):  # delete GL texture from GPU when object dies
        GL.glDeleteTextures(self.glid)

class TexturedMesh:
    def __init__(self, texture, vert_and_uv, faces):
        vertices, tex_uv = vert_and_uv
        self.vertex_array = VertexArray([vertices, tex_uv], faces)
        self.texture = Texture(texture)


    def draw(self, projection, view, model, color=(1,1,1,1), **_kwargs):

        color_shader = Shader(TEXT_VERT, TEXT_FRAG)

        locmat = GL.glGetUniformLocation(color_shader.glid, 'modelviewprojection')
        GL.glUseProgram(color_shader.glid)
        GL.glUniformMatrix4fv(locmat, 1, True, projection @ view @ model)

        GL.glUseProgram(color_shader.glid)

        #Display texture
        loc = GL.glGetUniformLocation(color_shader.glid, 'diffuseMap')
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(loc, 0)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        self.vertex_array.draw(GL.GL_TRIANGLES)
        GL.glUseProgram(0)


# -------------- 3D textured mesh loader ---------------------------------------

def load_textured(file):
    """ load resources using pyassimp, return list of TexturedMeshes """
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scene = pyassimp.load(file, option)
    except pyassimp.errors.AssimpError:
        #print('ERROR: pyassimp unable to load', file)
        return []  # error reading => return empty list

    # Note: embedded textures not supported at the moment
    path = os.path.dirname(file)
    for mat in scene.materials:
        mat.tokens = dict(reversed(list(mat.properties.items())))
        if 'file' in mat.tokens:  # texture file token
            tname = mat.tokens['file'].split('/')[-1].split('\\')[-1]
            # search texture in file's whole subdir since path often screwed up
            tname = [os.path.join(d[0], f) for d in os.walk(path) for f in d[2]
                     if tname.startswith(f) or f.startswith(tname)]
            if tname:
                mat.texture = tname[0]
            else:
                print('Failed to find texture:', tname)

    # prepare textured mesh
    meshes = []
    for mesh in scene.meshes:
        texture = scene.materials[mesh.materialindex].texture

        # tex coords in raster order: compute 1 - y to follow OpenGL convention
        tex_uv = ((0, 1) + mesh.texturecoords[0][:, :2] * (1, -1)
                  if mesh.texturecoords.size else None)

        # create the textured mesh object from texture, attributes, and indices
        meshes.append(TexturedMesh(texture, [mesh.vertices, tex_uv], mesh.faces))
        #print(0)

    size = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    #print('Loaded %s\t(%d meshes, %d faces)' % (file, len(scene.meshes), size))

    pyassimp.release(scene)
    return meshes

# ------------  KeyFrames Animation -------------------------------------------

class KeyFrames:
    """ Stores keyframe pairs for any value type with interpolation_function"""
    def __init__(self, time_value_pairs, interpolation_function=lerp):
        if isinstance(time_value_pairs, dict):  # convert to list of pairs
            time_value_pairs = time_value_pairs.items()
        keyframes = sorted(((key[0], key[1]) for key in time_value_pairs))
        self.times, self.values = zip(*keyframes)  # pairs list -> 2 lists
        self.interpolate = interpolation_function

    def value(self, time):
        """ Computes interpolated value from keyframes, for a given time """

        # 1. ensure time is within bounds else return boundary keyframe
        if (time <= self.times[0]):
            return self.values[0]
        elif (time >= self.times[-1]):
            return self.values[-1]
        # 2. search for closest index entry in self.times, using bisect_left function
        index = bisect_left(self.times, time)
        # 3. using the retrieved index, interpolate between the two neighboring values
        # in self.values, using the initially stored self.interpolate function
        quot = 1.0*(time - self.times[index - 1])/(self.times[index] - self.times[index - 1])
        return self.interpolate(self.values[index-1], self.values[index], quot)

def get_list_pairs(l):
    if isinstance(l, dict):
        return l.items()
    else:
        return l

class TransformKeyFrames:
    """ KeyFrames-like object dedicated to 3D transforms """
    def __init__(self, translate_keys, rotate_keys, scale_keys):
        """ stores 3 keyframe sets for translation, rotation, scale """
        translate_keys = get_list_pairs(translate_keys)
        rotate_keys = get_list_pairs(rotate_keys)
        scale_keys = get_list_pairs(scale_keys)
        keyf_t = sorted(((key[0], key[1]) for key in translate_keys))
        self.ttimes, self.trans = zip(*keyf_t)
        keyf_r = sorted(((key[0], key[1]) for key in rotate_keys))
        self.rtimes, self.rotate = zip(*keyf_r)
        keyf_s = sorted(((key[0], key[1]) for key in scale_keys))
        self.stimes, self.scale = zip(*keyf_s)

    def _linterp(self, time, times, vals):
        if time <= times[0]:
            return vals[0]
        elif time >= times[-1]:
            return vals[-1]
        #Time is in range
        index = bisect_left(times, time)
        quot = 1.0*(time - times[index - 1])/(times[index] - times[index - 1])
        return lerp(vals[index-1], vals[index], quot)

    def _qinterp(self, time):
        if time <= self.rtimes[0]:
            return self.rotate[0]
        elif time >= self.rtimes[-1]:
            return self.rotate[-1]
        index = bisect_left(self.rtimes, time)
        quot = 1.0*(time - self.rtimes[index - 1])/(self.rtimes[index] - self.rtimes[index - 1])
        return quaternion_slerp(self.rotate[index-1], self.rotate[index], quot)

    def value(self, time):
        """ Compute each component's interpolation and compose TRS matrix """
        tlerp = self._linterp(time, self.ttimes, self.trans)
        slerp = self._linterp(time, self.stimes, self.scale)
        rlerp = self._qinterp(time)
        rmat = quaternion_matrix(rlerp)
        tmat = translate(*tlerp)
        smat = scale(slerp)
        transformation = tmat @ rmat @ smat
        return transformation

class KeyFrameControlNode(Node):
    """ Place node with transform keys above a controlled subtree """
    def __init__(self, translate_keys, rotate_keys, scale_keys, **kwargs):
        super().__init__(**kwargs)
        self.keyframes = TransformKeyFrames(translate_keys, rotate_keys, scale_keys)

    def draw(self, projection, view, model, **param):
        """ When redraw requested, interpolate our node transform from keys """
        self.transform = self.keyframes.value(glfw.get_time())
        super().draw(projection, view, model, **param)

# -------------- Linear Blend Skinning : TP7 ---------------------------------

MAX_VERTEX_BONES = 4
MAX_BONES = 128

# Skinning Shader with textures
SKINNING_VERT_T = """#version 330 core
// ---- camera geometry
uniform mat4 projection, view;

// ---- Texture ambiante
uniform sampler2D diffuseMap;

// ---- normal&specMap
uniform sampler2D normalMap;
uniform sampler2D specMap;


// ---- Illumination de Phong
uniform vec3 light_position;
uniform vec3 K_d;
uniform vec3 K_s;
uniform float s;
uniform float normal_mapping;

// ---- skinning globals and attributes
const int MAX_VERTEX_BONES=%d, MAX_BONES=%d;
uniform mat4 boneMatrix[MAX_BONES];

// ---- vertex attributes
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normals;
layout(location = 2) in vec4 bone_ids;
layout(location = 3) in vec4 bone_weights;
layout(location = 4) in vec2 text_coord;
layout(location = 5) in vec3 tangent;
layout(location = 6) in vec3 bitangent;
// ----- couleur calculée dans le vertex shader même
// passage vec COLOR_FRAG ensuite
out vec3 fragColor;

vec3 normalization(vec3 t) {
    float som = t.x + t.y + t.z;
    return t/som;
}
vec3 normal_map;


void main() {

    // ------ calcul de la skinning matrix
    int id;
    mat4 skinMatrix = mat4(0.0);
    for (int i = 0; i < MAX_VERTEX_BONES; i++) {
        id = int(bone_ids[i]);
        skinMatrix += boneMatrix[id]*bone_weights[i];
    }

    // ------ compute world and normalized eye coordinates of our vertex
    vec4 wPosition4 = skinMatrix * vec4(position, 1.0);
    gl_Position = projection * view * wPosition4;

    // ---- matrice TBN et calcul ne normal_map
    if (normal_mapping > 0) {
        mat3 TBN = mat3(tangent, bitangent, normals);
        normal_map = texture(normalMap, text_coord).xyz;
        normal_map = normalization(normal_map * 2.0 - 1.0);
        normal_map = normalization(TBN * normal_map);
    } else {
        normal_map = normals;
    }

    vec3 K_a = texture(diffuseMap, text_coord).xyz;
    vec3 ref = reflect(position - light_position, normal_map);
    // valeur arbitraire
    vec3 view_vertex = vec3(0, 0, 0);

    // ---- Illumination de Phong
    fragColor = K_a + K_d*dot(position - light_position, normal_map) + K_s*pow(dot(ref, view_vertex), s);
}
""" % (MAX_VERTEX_BONES, MAX_BONES)

# Skinning shader without textures
SKINNING_VERT = """#version 330 core
// ---- camera geometry
uniform mat4 projection, view;

// ---- skinning globals and attributes
const int MAX_VERTEX_BONES=%d, MAX_BONES=%d;
uniform mat4 boneMatrix[MAX_BONES];

// ---- vertex attributes
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec4 bone_ids;
layout(location = 3) in vec4 bone_weights;
layout(location = 4) in vec2 text_coord;

// ----- interpolated attribute variables to be passed to fragment shader
out vec3 fragColor;

void main() {

    // ------ calcul de la skinning matrix
    int id;
    mat4 skinMatrix = mat4(0.0);
    for (int i=0; i<MAX_VERTEX_BONES; i++) {
        id = int(bone_ids[i]);
        skinMatrix += boneMatrix[id]*bone_weights[i];
    }


    // ------ compute world and normalized eye coordinates of our vertex
    vec4 wPosition4 = skinMatrix * vec4(position, 1.0);
    gl_Position = projection * view * wPosition4;

    fragColor = color;
}
""" % (MAX_VERTEX_BONES, MAX_BONES)

class SkinnedMesh:
    """class of skinned mesh nodes in scene graph """
    # dents du dinosaure non texturées
    def __init__(self, attributes, bone_nodes, bone_offsets, index=None):
        # setup shader attributes for linear blend skinning shader
        self.vertex_array = VertexArray(attributes, index)

        # feel free to move this up in Viewer as shown in previous practicals
        self.skinning_shader = Shader(SKINNING_VERT, COLOR_FRAG)

        # store skinning data
        self.bone_nodes = bone_nodes
        self.bone_offsets = bone_offsets

    def draw(self, projection, view, _model, **_kwargs):
        """ skinning object draw method """

        shid = self.skinning_shader.glid
        GL.glUseProgram(shid)

        # setup camera geometry parameters
        loc = GL.glGetUniformLocation(shid, 'projection')
        GL.glUniformMatrix4fv(loc, 1, True, projection)
        loc = GL.glGetUniformLocation(shid, 'view')
        GL.glUniformMatrix4fv(loc, 1, True, view)
        # bone world transform matrices need to be passed for skinning
        for bone_id, node in enumerate(self.bone_nodes):
            bone_matrix = node.world_transform @ self.bone_offsets[bone_id]

            bone_loc = GL.glGetUniformLocation(shid, 'boneMatrix[%d]' % bone_id)
            GL.glUniformMatrix4fv(bone_loc, 1, True, bone_matrix)

        # draw mesh vertex array
        self.vertex_array.draw(GL.GL_TRIANGLES)

        # leave with clean OpenGL state, to make it easier to detect problems
        GL.glUseProgram(0)


class SkinnedTextMesh:
    # corps du raptor texturé
    """class of skinned mesh nodes in scene graph """
    def __init__(self, attributes, bone_nodes, bone_offsets, texture, index=None):
        # setup shader attributes for linear blend skinning shader
        self.vertex_array = VertexArray(attributes, index)

        # feel free to move this up in Viewer as shown in previous practicals
        self.skinning_shader = Shader(SKINNING_VERT_T, COLOR_FRAG)

        # store skinning data
        self.bone_nodes = bone_nodes
        self.bone_offsets = bone_offsets

        # Texture ambiante
        self.texture = Texture(texture)

        # Textures normales et spéculaires
        self.normalMap = Texture("dino/textures/Dino_normal.png")
        self.specMap = Texture("dino/textures/Dino_spec.png")

    def draw(self, projection, view, _model, K_s=(0.0000000007, 0.0000000007, 0.0000000007), K_d=(0.00010, 0.00006, 0), light_position=(256, 0, 0), s=1.1, normal_mapping = 1.0, **_kwargs):
        """ skinning object draw method """

        shid = self.skinning_shader.glid
        GL.glUseProgram(shid)

        # setup camera geometry parameters
        loc = GL.glGetUniformLocation(shid, 'projection')
        GL.glUniformMatrix4fv(loc, 1, True, projection)
        loc = GL.glGetUniformLocation(shid, 'view')
        GL.glUniformMatrix4fv(loc, 1, True, view)

        #Display texture
        loc = GL.glGetUniformLocation(shid, 'diffuseMap')
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(loc, 0)

        #Normals texture
        loc = GL.glGetUniformLocation(shid, 'normalMap')
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.normalMap.glid)
        GL.glUniform1i(loc, 0)

        #Specular texutures texture
        loc = GL.glGetUniformLocation(shid, 'specMap')
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.specMap.glid)
        GL.glUniform1i(loc, 0)

        #setup phong parameters
        names = ['light_position', 'K_d', 'K_s', 's', 'normal_mapping']
        loc = {n: GL.glGetUniformLocation(self.skinning_shader.glid, n) for n in names}
        GL.glUniform3fv(loc["light_position"], 1, light_position)
        GL.glUniform3fv(loc["K_d"], 1, K_d)
        GL.glUniform3fv(loc["K_s"], 1, K_s)
        GL.glUniform1f(loc["s"], s)
        GL.glUniform1f(loc["normal_mapping"], normal_mapping)


        # bone world transform matrices need to be passed for skinning
        for bone_id, node in enumerate(self.bone_nodes):
            bone_matrix = node.world_transform @ self.bone_offsets[bone_id]

            bone_loc = GL.glGetUniformLocation(shid, 'boneMatrix[%d]' % bone_id)
            GL.glUniformMatrix4fv(bone_loc, 1, True, bone_matrix)

        # draw mesh vertex array
        self.vertex_array.draw(GL.GL_TRIANGLES)

        # leave with clean OpenGL state, to make it easier to detect problems
        GL.glUseProgram(0)


# -------- Skinning Control for Keyframing Skinning Mesh Bone Transforms ------
class SkinningControlNode(Node):
    """ Place node with transform keys above a controlled subtree """
    def __init__(self, *keys, **kwargs):
        super().__init__(**kwargs)
        self.keyframes = TransformKeyFrames(*keys) if keys[0] else None
        self.world_transform = identity()

    def draw(self, projection, view, model, **param):
        """ When redraw requested, interpolate our node transform from keys """
        if self.keyframes:  # no keyframe update should happens if no keyframes
            self.transform = self.keyframes.value(glfw.get_time())

        # store world transform for skinned meshes using this node as bone
        self.world_transform = model @ self.transform

        # default node behaviour (call children's draw method)
        super().draw(projection, view, model, **param)

# -------------- 3D resource loader -------------------------------------------
"""
calcul des tangentes et bitangentes dans le try juste avant le return
"""
def load_skinned(file):
    """load resources from file using pyassimp, return node hierarchy """
    try:
        option = pyassimp.postprocess.aiProcessPreset_TargetRealtime_MaxQuality
        scene = pyassimp.load(file, option)
    except pyassimp.errors.AssimpError:
        #print('ERROR: pyassimp unable to load', file)
        return []

    # ----- load animations
    def conv(assimp_keys, ticks_per_second):
        """ Conversion from assimp key struct to our dict representation """
        return {key.time / ticks_per_second: key.value for key in assimp_keys}

    # load first animation in scene file (could be a loop over all animations)
    transform_keyframes = {}
    if scene.animations:
        anim = scene.animations[0]
        for channel in anim.channels:
            # for each animation bone, store trs dict with {times: transforms}
            # (pyassimp name storage bug, bytes instead of str => convert it)
            transform_keyframes[channel.nodename.data.decode('utf-8')] = (
                conv(channel.positionkeys, anim.tickspersecond),
                conv(channel.rotationkeys, anim.tickspersecond),
                conv(channel.scalingkeys, anim.tickspersecond)
            )

    # Note: embedded textures not supported at the moment
    path = os.path.dirname(file)
    for mat in scene.materials:
        mat.tokens = dict(reversed(list(mat.properties.items())))
        if 'file' in mat.tokens:  # texture file token
            tname = mat.tokens['file'].split('/')[-1].split('\\')[-1]
            # search texture in file's whole subdir since path often screwed up
            tname = [os.path.join(d[0], f) for d in os.walk(path) for f in d[2]
                     if tname.startswith(f) or f.startswith(tname)]
            if tname:
                mat.texture = tname[0]
            else:
                print('Failed to find texture:', tname)

    # ---- prepare scene graph nodes
    # create SkinningControlNode for each assimp node.
    # node creation needs to happen first as SkinnedMeshes store an array of
    # these nodes that represent their bone transforms
    nodes = {}  # nodes: string name -> node dictionary

    def make_nodes(pyassimp_node):
        """ Recursively builds nodes for our graph, matching pyassimp nodes """
        trs_keyframes = transform_keyframes.get(pyassimp_node.name, (None,))

        node = SkinningControlNode(*trs_keyframes, name=pyassimp_node.name,
                                   transform=pyassimp_node.transformation)
        nodes[pyassimp_node.name] = node, pyassimp_node
        node.add(*(make_nodes(child) for child in pyassimp_node.children))
        return node

    root_node = make_nodes(scene.rootnode)

    # ---- create SkinnedMesh objects
    for mesh in scene.meshes:
        # -- skinned mesh: weights given per bone => convert per vertex for GPU
        # first, populate an array with MAX_BONES entries per vertex
        v_bone = np.array([[(0, 0)]*MAX_BONES] * mesh.vertices.shape[0],
                          dtype=[('weight', 'f4'), ('id', 'u4')])
        for bone_id, bone in enumerate(mesh.bones[:MAX_BONES]):
            for entry in bone.weights:  # weight,id pairs necessary for sorting
                v_bone[entry.vertexid][bone_id] = (entry.weight, bone_id)

        v_bone.sort(order='weight')             # sort rows, high weights last
        v_bone = v_bone[:, -MAX_VERTEX_BONES:]  # limit bone size, keep highest

        # prepare bone lookup array & offset matrix, indexed by bone index (id)
        bone_nodes = [nodes[bone.name][0] for bone in mesh.bones]
        bone_offsets = [bone.offsetmatrix for bone in mesh.bones]

        try :
            # Si les textures sont définies : corp principal du dinosaure
            texture = scene.materials[mesh.materialindex].texture
            # tex coords in raster order: compute 1 - y to follow OpenGL convention
            if mesh.texturecoords.size:
                tex_uv = np.array((0, 1) + mesh.texturecoords[0][:, :2] * (1, -1), dtype=np.float32)
            else:
                tex_uv = None

            tangents = []
            bitangents = []

            for face in mesh.faces:
                # Calcul des tangentes et bitangentes pour chaque face de la figure
                v0 = mesh.vertices[face[0]]
                v1 = mesh.vertices[face[1]]
                v2 = mesh.vertices[face[2]]

                uv0 = tex_uv[face[0]]
                uv1 = tex_uv[face[1]]
                uv2 = tex_uv[face[2]]

                deltaPos1 = [v1[i] - v0[i] for i in range(3)]
                deltaPos2 = [v2[i] - v0[i] for i in range(3)]

                deltaUV1 = [uv1[i] - uv0[i] for i in range(2)]
                deltaUV2 = [uv2[i] - uv0[i] for i in range(2)]

                r = 1 / ((deltaUV1[0]*deltaUV2[1]) - (deltaUV2[0]*deltaUV1[1]))
                tangent = [(deltaPos1[i]*deltaUV2[1])-(deltaPos2[i]*deltaUV1[1]) for i in range(3)]
                bitangent = [(deltaPos2[i]*deltaUV2[1])-(deltaPos1[i]*deltaUV1[1]) for i in range(3)]

                tangents.append(tangent)
                bitangents.append(bitangent)


            # initialize skinned mesh and store in pyassimp_mesh for node addition
            # ajout des coordonnées uv de texture, des tangentes et bitangentes
            mesh.skinned_mesh = SkinnedTextMesh(
                    [mesh.vertices, mesh.normals, v_bone['id'], v_bone['weight'], tex_uv, tangents, bitangents],
                    bone_nodes, bone_offsets, texture, mesh.faces
            )
        except AttributeError:
            # cas sans textures définies (dents du dinosaure)
            mesh.skinned_mesh = SkinnedMesh(
                    [mesh.vertices, mesh.normals, v_bone['id'], v_bone['weight']],
                    bone_nodes, bone_offsets, mesh.faces
            )


    # ------ add each mesh to its intended nodes as indicated by assimp
    for final_node, assimp_node in nodes.values():
        final_node.add(*(_mesh.skinned_mesh for _mesh in assimp_node.meshes))

    nb_triangles = sum((mesh.faces.shape[0] for mesh in scene.meshes))
    # print('Loaded', file, '\t(%d meshes, %d faces, %d nodes, %d animations)' %
    #       (len(scene.meshes), nb_triangles, len(nodes), len(scene.animations)))
    pyassimp.release(scene)
    return [root_node]

# ------------  Viewer class & window management ------------------------------

class GLFWTrackball(Trackball):
    """ Use in Viewer for interactive viewpoint control """

    def __init__(self, win):
        """ Init needs a GLFW window handler 'win' to register callbacks """
        super().__init__()
        self.mouse = (0, 0)
        glfw.set_cursor_pos_callback(win, self.on_mouse_move)
        glfw.set_scroll_callback(win, self.on_scroll)

    def on_mouse_move(self, win, xpos, ypos):
        """ Rotate on left-click & drag, pan on right-click & drag """
        old = self.mouse
        self.mouse = (xpos, glfw.get_window_size(win)[1] - ypos)
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT):
            self.drag(old, self.mouse, glfw.get_window_size(win))
        if glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT):
            self.pan(old, self.mouse)

    def on_scroll(self, win, _deltax, deltay):
        """ Scroll controls the camera distance to trackball center """
        self.zoom(deltay, glfw.get_window_size(win)[1])

class Viewer:
    """ GLFW viewer window, with classic initialization & graphics loop """

    def __init__(self, width=640, height=480):

        # version hints: create GL window with >= OpenGL 3.3 and core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, False)
        self.win = glfw.create_window(width, height, 'Projet G3D - R pour faire rugir le raptor - N pour le normal mapping', None, None)
        # make win's OpenGL context current; no OpenGL calls can happen before
        glfw.make_context_current(self.win)

        # register event handlers
        glfw.set_key_callback(self.win, self.on_key)

        # useful message to check OpenGL renderer characteristics
        print('OpenGL', GL.glGetString(GL.GL_VERSION).decode() + ', GLSL',
              GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION).decode() +
              ', Renderer', GL.glGetString(GL.GL_RENDERER).decode())

        # initialize GL by setting viewport and default render characteristics
        GL.glClearColor(0.1, 0.1, 0.1, 0.1)
        GL.glEnable(GL.GL_DEPTH_TEST)         # depth test now enabled (TP2)
        GL.glEnable(GL.GL_CULL_FACE)          # backface culling enabled (TP2)

        # compile and initialize shader programs once globally
        self.color_shader = Shader(COLOR_VERT, COLOR_FRAG)

        #determine if the normal_mapping is use or not
        self.normal_mapping = 1

        # initially empty list of object to draw
        self.drawables = []

        # initialize trackball
        self.trackball = GLFWTrackball(self.win)

        # cyclic iterator to easily toggle polygon rendering modes
        self.fill_modes = cycle([GL.GL_LINE, GL.GL_POINT, GL.GL_FILL])

    def run(self):
        """ Main render loop for this OpenGL window """
        while not glfw.window_should_close(self.win):
            # clear draw buffer and depth buffer (<-TP2)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            winsize = glfw.get_window_size(self.win)
            view = self.trackball.view_matrix()
            projection = self.trackball.projection_matrix(winsize)

            # draw our scene objects
            for drawable in self.drawables:
                drawable.draw(projection, view, identity(), win = self.win,
                              color_shader=self.color_shader, normal_mapping = self.normal_mapping)

            # flush render commands, and swap draw buffers
            glfw.swap_buffers(self.win)

            # Poll for and process events
            glfw.poll_events()

    def add(self, *drawables):
        """ add objects to draw in this window """
        self.drawables.extend(drawables)

    def on_key(self, _win, key, _scancode, action, _mods):
        """ 'Q' or 'Escape' quits """
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)
            if key == glfw.KEY_W:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))
            if key == glfw.KEY_R:
                os.system("pkill aplay")
                os.system("aplay T-Rex.wav &")
                glfw.set_time(0)
            if key == glfw.KEY_N:
                self.normal_mapping = 1 - self.normal_mapping

# -------------- main program and scene setup --------------------------------
def main():
    """ create a window, add scene objects, then run rendering loop """
    viewer = Viewer()

    # paramètre de transformation des paramètres
    #sol
    ground_size = 512
    ground_offset = 20

    #dinosaure
    characters_offset_x = 0
    characters_offset_y = -20
    characters_offset_z = 0
    characters_scale = 15
    characters_rotate_deg = 180

    #forêt
    forest_offset = -15
    forest_scale = 1.5

    #skybox
    Skysphere_scale = 3

    characters = Node(transform = translate(characters_offset_x, characters_offset_y, characters_offset_z) @ scale(characters_scale) @ rotate(axis=(0, 1, 0), angle = characters_rotate_deg))
    characters.add(*load_skinned("dino/Dinosaurus_roar.dae"))

    forest = Node(transform = translate(0, forest_offset, 0) @ scale(forest_scale))
    forest.add(*load_textured("trees9/forest.obj"))

    ground = Node(transform = translate(-ground_size>>1, ground_offset, -ground_size>>1))
    ground.add(sol(ground_size))

    Skysphere = Node(transform = scale(Skysphere_scale))
    Skysphere.add(*load_textured("Skysphere/skysphere.obj"))

    scene = Node(transform = identity(), children = [characters, forest, ground, Skysphere])

    viewer.add(scene)

    viewer.run()

if __name__ == '__main__':
    glfw.init()                # initialize window system glfw
    main()                     # main function keeps variables locally scoped
    glfw.terminate()           # destroy all glfw windows and GL contexts
