from scipy.spatial.distance import euclidean as euc
from skimage.transform import resize
from sklearn.cluster import KMeans
import mpl_toolkits.mplot3d.axes3d
import matplotlib.pyplot as plt
from scipy.misc import imread
import pywavefront
import numpy as np
import copy
import math

class ImgParser:
  def __init__(self, *args, **kwargs):
    self.img_path = args[0]
    self.img = imread(self.img_path, mode='L')
    self.input_img = copy.deepcopy(self.img)
    self.vertices = self.get_vertices(self.img)
    self.input_vertices = copy.deepcopy(self.vertices)

  def get_vertices(self, img):
    '''Return the vertices from in image using Floyd-Steinberg dithering'''
    img = copy.deepcopy(img)
    img = img.T # col-major order
    w, h = img.shape
    for y in range(h):
      for x in range(w):
        old = img[x, y]
        new = 0 if old < 127 else 255
        img[x, y] = new
        quant_error = old - new
        if x < w - 1:
          img[x + 1, y] += quant_error * 7 // 16
        if x > 0 and y < h - 1:
          img[x - 1, y + 1] += quant_error * 3 // 16
        if y < h - 1:
          img[x, y + 1] += quant_error * 5 // 16
        if x < w - 1 and y < h - 1:
          img[x + 1, y + 1] += quant_error * 1 // 16
    img = img.T # row-major order
    self.img = img
    self.vertices = np.argwhere(img == 0)
    return self.vertices

  def get_n_vertices(self, n):
    '''Return a selection of `n` vertices from self.vertices'''
    y, x = self.get_resized_shape(n)
    resized = (resize(self.input_img, (y, x)) * 255).astype(int)
    replace = self.vertices.shape[0]<n
    return self.vertices[np.random.choice(len(self.vertices), size=n, replace=replace)]

  def get_resized_shape(self, n_verts):
    '''Return the size to reshape the input image to get n_verts'''
    # determine the proportion of pixels that are inked in the input image
    px_percent = self.input_vertices.shape[0] / self.input_img.size
    # determine the size the input image must be so p percent shading == n verts
    target_size = int(n_verts/px_percent)
    # determine the ratio of the input image's y dim vs x dim
    y_ratio = self.input_img.shape[0] / self.input_img.shape[1]
    x_ratio = self.input_img.shape[1] / self.input_img.shape[0]
    # identify the sizes of each side
    y = math.ceil(target_size**(1/2) * y_ratio)
    x = math.ceil(target_size**(1/2) * x_ratio)
    return y, x # size of each dimension

  def plot(self, *args, s=0.01, figscale=0.1):
    '''Plot self.vertces; s sets point size; figscale sets figsize'''
    if len(args):
      vertices = args[0]
    else:
      vertices = self.vertices
    h, w = self.img.shape
    plt.figure(figsize=(int(w*figscale), int(h*figscale)))
    y, x = [np.array(i) for i in zip(*vertices)]
    plt.scatter(x, 1-y, s=s)
    plt.show()


class ObjParser:
  def __init__(self, *args, **kwargs):
    self.obj_file_path = args[0]
    self.scene = pywavefront.Wavefront(self.obj_file_path, collect_faces=True,)
    self.vertices = np.array(self.scene.vertices) # array of vertex coords
    self.faces = np.array(self.scene.mesh_list[0].faces) # indices into vertices
    # store references to the original vert and face elements in this mesh
    self.input_vertices = copy.deepcopy(self.vertices)
    self.input_faces = copy.deepcopy(self.faces)

  def plot(self, *args, s=0.5):
    '''Plot self.vertices'''
    if len(args):
      vertices = args[0]
    else:
      vertices = self.vertices
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = zip(*vertices)
    c = range(vertices.shape[0])
    ax.scatter(x, y, z, marker='o', s=s, c=c)
    plt.show()

  def midpoint(self, p0, p1):
    '''Return the midpoint between two points'''
    return (p0 + p1) / 2

  def get_n_vertices(self, n):
    '''Given an obj file and a target number `n`, return the obj file represented with `n` vertices'''
    # trivial case
    vertices = copy.deepcopy(self.vertices)
    faces = copy.deepcopy(self.faces)
    if n == vertices.shape[0]:
      return vertices
    # add vertices
    elif vertices.shape[0] < n:
      print(' * adding vertices')
      while vertices.shape[0] < n:
        # face_positions.shape = n_faces, n_vertices per face, n_dims per vert
        fi = faces[0] # fi are face indices that index into vertices
        f = vertices[fi] # f is the set of vertices in a face
        p0, p1, p2 = [np.array(i) for i in zip(*f.T)]
        # find the longest side in ABC, find its midpoint D, and divide ABC into two triangles / faces
        max_d = np.argmax([euc(p0, p1), euc(p1, p2), euc(p0, p2)])
        new_vert = vertices.shape[0]
        if max_d == 0:
          point = self.midpoint(p0, p1)
          f1 = [fi[0], fi[2], new_vert]
          f2 = [fi[1], fi[2], new_vert]
        elif max_d == 1:
          point = self.midpoint(p1, p2)
          f1 = [fi[1], fi[0], new_vert]
          f2 = [fi[2], fi[0], new_vert]
        elif max_d == 2:
          point = self.midpoint(p0, p2)
          f1 = [fi[0], fi[1], new_vert]
          f2 = [fi[2], fi[1], new_vert]
        # add the new point to the list of points
        vertices = np.vstack([vertices, point])
        # remove the split face f and add the two new faces into which f was split
        faces = np.vstack([faces[1:], f1, f2])
    # remove vertices
    elif vertices.shape[0] > n:
      print(' * removing vertices')
      clf = KMeans(n_clusters=n)
      clf.fit_transform(vertices)
      vertices = clf.cluster_centers_
    return vertices
