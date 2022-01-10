
import numpy as np
#import cmapy
import matplotlib.figure as mplfigure
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from skimage import io


COCO_PERSON_KEYPOINT_NAMES = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)

# rules for pairs of keypoints to draw a line between, and the line color to use.
KEYPOINT_CONNECTION_RULES = [
    # face
    ("left_ear", "left_eye", (102, 204, 255)),
    ("right_ear", "right_eye", (51, 153, 255)),
    ("left_eye", "nose", (102, 0, 204)),
    ("nose", "right_eye", (51, 102, 255)),
    # upper-body
    ("left_shoulder", "right_shoulder", (255, 128, 0)),
    ("left_shoulder", "left_elbow", (153, 255, 204)),
    ("right_shoulder", "right_elbow", (128, 229, 255)),
    ("left_elbow", "left_wrist", (153, 255, 153)),
    ("right_elbow", "right_wrist", (102, 255, 224)),
    # lower-body
    ("left_hip", "right_hip", (255, 102, 0)),
    ("left_hip", "left_knee", (255, 255, 77)),
    ("right_hip", "right_knee", (153, 255, 204)),
    ("left_knee", "left_ankle", (191, 255, 128)),
    ("right_knee", "right_ankle", (255, 195, 77)),
]

COLOR1 = [(179,0,0),(228,26,28),(255,255,51), (49,163,84), (0,109,45), (255,255,51), (240,2,127), (240,2,127),
          (240,2,127), (240,2,127), (240,2,127), (217,95,14), (254,153,41),(255,255,51), (44,127,184), (0,0,255)]


class VisImage:
    def __init__(self, img, scale=1.0):
        self.img = img
        self.scale = scale
        self.width, self.height = img.shape[1], img.shape[0]
        self._setup_figure(img)

    def _setup_figure(self, img):
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        # Need to imshow this first so that other patches can be drawn on top
        ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

        self.fig = fig
        self.ax = ax

    def save(self, filepath):
        self.fig.savefig(filepath)

    def get_image(self):
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")


class KeypointsVis:
    def __init__(self, image):
        self.keypoint_threshold = 0.05
        self.black = (0, 0, 0)
        self.red = (1.0, 0, 0)
        self.img = image
        scale = 1.0
        self.output = VisImage(self.img, scale=scale)
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 90, 10 // scale
        )
        
    def draw_and_connect_keypoints(self, keypoints):
        visible = {}
        for idx, keypoint in enumerate(keypoints):
            # draw keypoint
            x, y, prob = keypoint
            if prob > self.keypoint_threshold:
                self.draw_circle((x, y), color=self.black)
                keypoint_name = COCO_PERSON_KEYPOINT_NAMES[idx]
                visible[keypoint_name] = (x, y)

        for kp0, kp1, color in KEYPOINT_CONNECTION_RULES:
            if kp0 in visible and kp1 in visible:
                x0, y0 = visible[kp0]
                x1, y1 = visible[kp1]
                color = tuple(x / 255.0 for x in color)
                self.draw_line([x0, x1], [y0, y1], color=color)

        # draw lines from nose to mid-shoulder and mid-shoulder to mid-hip
        # Note that this strategy is specific to person keypoints.
        # For other keypoints, it should just do nothing
        try:
            ls_x, ls_y = visible["left_shoulder"]
            rs_x, rs_y = visible["right_shoulder"]
            mid_shoulder_x, mid_shoulder_y = (ls_x + rs_x) / 2, (ls_y + rs_y) / 2
        except KeyError:
            pass
        else:
            # draw line from nose to mid-shoulder
            nose_x, nose_y = visible.get("nose", (None, None))
            if nose_x is not None:
                self.draw_line([nose_x, mid_shoulder_x], [nose_y, mid_shoulder_y], color=self.red)

            try:
                # draw line from mid-shoulder to mid-hip
                lh_x, lh_y = visible["left_hip"]
                rh_x, rh_y = visible["right_hip"]
            except KeyError:
                pass
            else:
                mid_hip_x, mid_hip_y = (lh_x + rh_x) / 2, (lh_y + rh_y) / 2
                self.draw_line([mid_hip_x, mid_shoulder_x], [mid_hip_y, mid_shoulder_y], color=self.red)
        return self.output
        
    def draw_circle(self, circle_coord, color, radius=3):
        x, y = circle_coord
        self.output.ax.add_patch(
            mpl.patches.Circle(circle_coord, radius=radius, fill=True, color=color)
        )
        return self.output

    def draw_line(self, x_data, y_data, color, linestyle="-", linewidth=None):
        if linewidth is None:
            linewidth = self._default_font_size / 3
        linewidth = max(linewidth, 1)
        self.output.ax.add_line(
            mpl.lines.Line2D(
                x_data,
                y_data,
                linewidth=linewidth* self.output.scale,
                color=color,
                linestyle=linestyle,
            )
        )
        return self.output


class CenterVis:
    def __init__(self, image):
        self.keypoint_threshold = 0.05
        self.black = (0, 0, 0)
        self.red = (1.0, 0, 0)
        self.blue = (0.24, 0.64, 1.0)
        self.img = image
        scale = 1.0
        self.output = VisImage(self.img, scale=scale)
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 90, 10 // scale
        )
        
    def draw_and_connect_keypoints(self, keypoints):
        center = keypoints[:2]
        # draw center
        
        keypoints= keypoints[2:].reshape(-1,3)
        visible = {}
        for idx, keypoint in enumerate(keypoints):
            # draw keypoint
            x, y, prob = keypoint
            if prob > self.keypoint_threshold:
                self.draw_circle((x, y), color=self.blue)
                keypoint_name = COCO_PERSON_KEYPOINT_NAMES[idx]
                visible[keypoint_name] = (x, y)
                self.draw_line([center[0], x], [center[1], y], color=self.red)
                
        self.draw_circle((center[0], center[1]), color=self.red)

        return self.output
        
    def draw_circle(self, circle_coord, color, radius=3):
        x, y = circle_coord
        self.output.ax.add_patch(
            mpl.patches.Circle(circle_coord, radius=radius, fill=True, color=color)
        )
        return self.output

    def draw_line(self, x_data, y_data, color, linestyle="-", linewidth=None):
        if linewidth is None:
            linewidth = self._default_font_size / 3
        linewidth = max(linewidth, 1)
        self.output.ax.add_line(
            mpl.lines.Line2D(
                x_data,
                y_data,
                linewidth=(linewidth-2) * self.output.scale,
                color=color,
                linestyle=linestyle,
            )
        )
        return self.output


class VisibilityVis:
    def __init__(self, image):
        self.keypoint_threshold = 0.5
        self.black = (0, 0, 0)
        self.red = (1.0, 0, 0)
        self.img = image
        scale = 1.0
        self.output = VisImage(self.img, scale=scale)
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 90, 10 // scale
        )
        self.labels = ["+",".","x"]
        
    def draw_and_connect_keypoints(self, keypoints):
        for idx, keypoint in enumerate(keypoints):
            x, y, prob = keypoint
            color = tuple(x / 255.0 for x in COLOR1[idx])
            if prob > self.keypoint_threshold:
                self.draw_circle((x, y), color=color)
            else:
                self.draw_cross((x, y), color=color)
        return self.output
    
    def draw_groundtruth(self, keypoints):
        for idx, keypoint in enumerate(keypoints):
            x, y, prob = keypoint
            color = tuple(x / 255.0 for x in COLOR1[idx])
            if prob > self.keypoint_threshold:
                self.draw_plus((x, y), color=color)
        return self.output
        
    def draw_circle(self, circle_coord, color, radius=3):
        x, y = circle_coord
        self.output.ax.plot(
                            x,
                            y,
                            self.labels[1],
                            ms=25,
                            color=color,
                        )
        return self.output
    
    
    def draw_cross(self, circle_coord, color, radius=3):
        x, y = circle_coord
        self.output.ax.plot(
                            x,
                            y,
                            self.labels[2],
                            ms=25,
                            color=color,
                        )
        return self.output
    
    
    def draw_plus(self, circle_coord, color, radius=3):
        x, y = circle_coord
        self.output.ax.plot(
                            x,
                            y,
                            self.labels[0],
                            ms=25,
                            color=color,
                        )
        return self.output
