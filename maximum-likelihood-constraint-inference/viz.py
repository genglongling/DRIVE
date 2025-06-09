import pyglet
import numpy as np

# Collection of shapes, and possibly other collections
class Group:
    
    def __init__(self, items = []):
        self.items = items
    
    def draw(self):
        for item in self.items:
            item.draw()

# Collection with a single image
class Image(Group):

    def __init__(self, url, x, y, w, h, rotation = 0, anchor_centered = False):
        assert(url[-3:] in ['png'])
        if url[-3:] == 'png': decoder = pyglet.image.codecs.png.PNGImageDecoder()
        image = pyglet.image.load(url, decoder = decoder)
        if anchor_centered: image.anchor_x, image.anchor_y = image.width//2, image.height//2
        scale_x, scale_y = w/image.width, h/image.height
        sprite = pyglet.sprite.Sprite(img = image)
        sprite.update(x = x, y = y, scale_x = scale_x, scale_y = scale_y,
            rotation = rotation)
        super().__init__(items = [sprite])

class Canvas(pyglet.window.Window):

    def __init__(self, w, h, id=None):
        super().__init__(w, h, visible=False)
        self.items = []
        if id:
            bg_url = "inD/%s_background.png" % id
            self.items += [Image(bg_url, 20, 60, w-30, h-60)]
    
    def on_draw(self):
        self.clear()
        for item in self.items:
            item.draw()

    def render(self):
        pyglet.app.run()


# Basic shape, dummy class
class Shape: pass

# Basic rectangle
class Rectangle(Shape):
    
    def __init__(self, pts, color = (0, 0, 0, 1), visible=True):
        self.pts = pts
        self.color = color
        self.visible = visible
    
    def draw(self):
        if not self.visible:
            return
        pyglet.gl.glBegin(pyglet.gl.GL_QUADS)
        pyglet.gl.glColor4f(*self.color)
        pyglet.gl.glVertex3f(*self.pts[0], 0)
        pyglet.gl.glVertex3f(*self.pts[1], 0)
        pyglet.gl.glVertex3f(*self.pts[2], 0)
        pyglet.gl.glVertex3f(*self.pts[3], 0)
        pyglet.gl.glEnd()

def calculate_rotated_bboxes(center_points_x, center_points_y, length, width, rotation=0):
    """
    Calculate bounding box vertices from centroid, width and length.
    :param centroid: center point of bbox
    :param length: length of bbox
    :param width: width of bbox
    :param rotation: rotation of main bbox axis (along length)
    :return:
    """

    centroid = np.array([center_points_x, center_points_y]).transpose()

    centroid = np.array(centroid)
    if centroid.shape == (2,):
        centroid = np.array([centroid])

    # Preallocate
    data_length = centroid.shape[0]
    rotated_bbox_vertices = np.empty((data_length, 4, 2))

    # Calculate rotated bounding box vertices
    rotated_bbox_vertices[:, 0, 0] = -length / 2
    rotated_bbox_vertices[:, 0, 1] = -width / 2

    rotated_bbox_vertices[:, 1, 0] = length / 2
    rotated_bbox_vertices[:, 1, 1] = -width / 2

    rotated_bbox_vertices[:, 2, 0] = length / 2
    rotated_bbox_vertices[:, 2, 1] = width / 2

    rotated_bbox_vertices[:, 3, 0] = -length / 2
    rotated_bbox_vertices[:, 3, 1] = width / 2

    for i in range(4):
        th, r = cart2pol(rotated_bbox_vertices[:, i, :])
        rotated_bbox_vertices[:, i, :] = pol2cart(th + rotation, r).squeeze()
        rotated_bbox_vertices[:, i, :] = rotated_bbox_vertices[:, i, :] + centroid

    return rotated_bbox_vertices

def cart2pol(cart):
    """
    Transform cartesian to polar coordinates.
    :param cart: Nx2 ndarray
    :return: 2 Nx1 ndarrays
    """
    if cart.shape == (2,):
        cart = np.array([cart])

    x = cart[:, 0]
    y = cart[:, 1]

    th = np.arctan2(y, x)
    r = np.sqrt(np.power(x, 2) + np.power(y, 2))
    return th, r


def pol2cart(th, r):
    """
    Transform polar to cartesian coordinates.
    :param th: Nx1 ndarray
    :param r: Nx1 ndarray
    :return: Nx2 ndarray
    """

    x = np.multiply(r, np.cos(th))
    y = np.multiply(r, np.sin(th))

    cart = np.array([x, y]).transpose()
    return cart