"""Live demo of target pose matching."""
import utils
import inference
import cv2
import numpy as np
import seaborn as sns
import yaml
from time import sleep
from time import perf_counter as time
from threading import Thread
from collections import deque
from typing import Optional, Tuple
import argparse


class TargetPoses:
    def __init__(
        self,
        target_poses_path: str,
        max_height: int,
        initial_target: Optional[int] = 0,
        border_width: int = 4,
        border_color: Tuple[int, int, int] = (255, 255, 255),
        selected_width: int = 12,
        selected_color: Tuple[int, int, int] = (0, 200, 0),
        completed_alpha: float = 0.7,
    ):
        self.target_poses_path = target_poses_path
        self.target_ind = initial_target
        self.max_height = max_height
        self.border_width = border_width
        self.border_color = border_color
        self.selected_width = selected_width
        self.selected_color = selected_color
        self.completed_alpha = completed_alpha

        # Load data
        with open(self.target_poses_path, "r") as f:
            self.targets = yaml.load(f, yaml.Loader)

        for i in range(len(self.targets)):
            # Convert poses to numpy array.
            self.targets[i]["pose"] = np.array(self.targets[i]["pose"])
            self.targets[i]["norm_pose"] = np.array(self.targets[i]["norm_pose"])

            # Initialize completeness flag
            self.targets[i]["complete"] = False

        # Preload and preprocess viz images.
        self.img_height = self.max_height // len(self.targets)
        for i, target in enumerate(self.targets):
            img = cv2.imread(target["viz_path"])
            scale = self.img_height / img.shape[0]
            img = cv2.resize(img, None, fx=scale, fy=scale)
            img = utils.add_border(img, self.border_width, self.border_color)
            self.targets[i]["img"] = img

        assert (
            sum([target["img"].shape[0] for target in self.targets]) == self.max_height
        )

    def __len__(self) -> int:
        return len(self.targets)

    def next_target(self):
        self.target_ind = (self.target_ind + 1) % len(self)

    @property
    def current_target(self):
        return self.targets[self.target_ind]

    @property
    def width(self) -> int:
        return self.targets[0]["img"].shape[1]

    def reset_completeness(self):
        for i in range(len(self)):
            self.targets[i]["complete"] = False

    def complete_current(self):
        self.targets[self.target_ind]["complete"] = True

    def render(self) -> np.ndarray:
        rendered = []
        for i, target in enumerate(self.targets):
            img = target["img"].copy()

            if self.target_ind is not None and i == self.target_ind:
                img = utils.add_border(img, self.selected_width, self.selected_color)
            elif target["complete"]:
                img = cv2.addWeighted(
                    img,
                    self.completed_alpha,
                    np.full(img.shape, 255, dtype=img.dtype),
                    1 - self.completed_alpha,
                    0,
                )

            rendered.append(img)
        rendered = np.concatenate(rendered, axis=0)
        return rendered


class LiveDemo:
    def __init__(
        self,
        camera: int = 0,
        camera_width: int = 1920,
        camera_height: int = 1080,
        camera_brightness: int = 128,
        fps: float = 30,
        horizontal_mirror: bool = True,
        render_scale: float = 1.0,
        node_cmap: str = "tab20",
        node_radius: int = 8,
        node_thickness: int = 6,
        arrow_thickness: int = 5,
        max_arrow_length: int = 150,
        target_thickness: int = 5,
        rel_target_radius: float = 0.4,
        blend_alpha: float = 0.6,
        model_name: str = "thunder",
        center_pad: bool = False,
        min_score: float = 0.4,
        buffer_size: int = 5,
        target_poses_path: str = "data/target_poses.yaml",
    ):
        self.camera = camera
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_brightness = camera_brightness
        self.fps = fps
        self.horizontal_mirror = horizontal_mirror
        self.render_scale = render_scale
        self.node_cmap = node_cmap
        self.node_radius = node_radius
        self.node_thickness = node_thickness
        self.arrow_thickness = arrow_thickness
        self.max_arrow_length = max_arrow_length
        self.target_thickness = target_thickness
        self.rel_target_radius = rel_target_radius
        self.blend_alpha = blend_alpha
        self.model_name = model_name
        self.center_pad = center_pad
        self.min_score = min_score
        self.buffer_size = buffer_size
        self.target_poses_path = target_poses_path

        # Setup camera.
        self.capture = cv2.VideoCapture(self.camera, cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.capture.set(cv2.CAP_PROP_BRIGHTNESS, self.camera_brightness)

        # Reset properties in case camera doesn't support the specified dimensions.
        self.camera_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.camera_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize rendering.
        cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Load model.
        self.model = inference.load_model(model_name)
        self.image_size = self.model.input.shape[1]
        self.n_nodes = self.model.outputs[0].shape[0]

        # Setup colors.
        self.node_colors = (
            np.array(sns.color_palette(self.node_cmap, self.n_nodes)) * 255
        )

        # Load target poses.
        self.target_poses = TargetPoses(
            target_poses_path=self.target_poses_path,
            max_height=int(self.camera_height * self.render_scale),
        )

        # Specify nodes that will be used as targets.
        # TODO: Parametrize this?
        self.target_nodes = (
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        )

        # Initialize buffers.
        self.frame = None
        self.points = None
        self.scores = None
        self.node_buffers = [
            deque(maxlen=self.buffer_size) for _ in range(self.n_nodes)
        ]
        self.smooth_points = np.full((self.n_nodes, 2), np.nan)

        # Run the update thread.
        self.done = False
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.done:
            dt = 0
            if self.capture.isOpened():
                t0 = time()

                # Fetch webcam image
                cap_status, img = self.capture.read()

                if self.horizontal_mirror:
                    # Horizontal mirror
                    img = img[:, ::-1]

                # Crop to horizontal center to account for targets
                crop_x0 = int((self.target_poses.width / self.render_scale) // 2)
                crop_x1 = self.camera_width - crop_x0
                img = img[:, crop_x0:crop_x1]

                # Predict
                self.points, self.scores = inference.predict(
                    self.model, img[:, :, ::-1], pad=self.center_pad
                )

                # Update keypoint buffer
                for j, (pt, score) in enumerate(zip(self.points, self.scores)):
                    if score > self.min_score:
                        self.node_buffers[j].append(pt)
                    else:
                        self.node_buffers[j].append((np.nan, np.nan))

                    # Compute smoothed version
                    node_buffer = np.array(self.node_buffers[j])  # (n, 2)
                    if np.isnan(node_buffer).all():
                        self.smooth_points[j, :] = np.nan
                    else:
                        self.smooth_points[j] = np.nanmean(node_buffer, axis=0)

                # Upscale for viz
                img = cv2.resize(img, None, fx=self.render_scale, fy=self.render_scale)

                # Copy raw for overlaying with alpha blending
                img0 = img.copy()

                # Draw nodes
                for j in range(self.n_nodes):
                    if not (np.isnan(self.smooth_points[j]).all()):
                        pt = self.smooth_points[j] * self.render_scale
                        img = cv2.circle(
                            img,
                            (int(pt[0]), int(pt[1])),
                            radius=int(self.node_radius * self.render_scale),
                            color=self.node_colors[j][::-1],
                            thickness=int(self.node_thickness * self.render_scale),
                            lineType=cv2.LINE_AA,
                        )

                # Target pose
                target_pose = self.target_poses.current_target
                ref_pts = self.smooth_points[[target_pose["ref1"], target_pose["ref2"]]]
                if not (np.isnan(ref_pts).any()):
                    norm_factor = np.linalg.norm(ref_pts[0] - ref_pts[1])
                    origin = ref_pts.mean(axis=0)

                    n_in_target = 0
                    for node_name in self.target_nodes:
                        j = utils.KEYPOINT_NODES[node_name]
                        target_rel_pos = target_pose["norm_pose"][j]

                        img, in_target = self.render_target(
                            img,
                            keypoint=self.smooth_points[j],
                            target=origin + (norm_factor * target_rel_pos),
                            node_col=self.node_colors[j],
                            target_radius=self.rel_target_radius * norm_factor,
                        )
                        n_in_target += int(in_target)

                    if n_in_target == len(self.target_nodes):
                        # Completed pose! Show visual indicator and move to next pose.
                        # TODO: Hold for a min number of frames?
                        #   Could store this in the pose_targets and use it to render a
                        #   progress bar.
                        self.target_poses.complete_current()
                        self.target_poses.next_target()

                # Alpha blend
                img = cv2.addWeighted(
                    img, self.blend_alpha, img0, 1 - self.blend_alpha, 0
                )

                # Concatenate the rendered targets
                img = np.concatenate([img, self.target_poses.render()], axis=1)
                img = img[:, : int(self.camera_width * self.render_scale)]

                # Save final rendered image
                self.frame = img

                dt = time() - t0

            # Sleep for remainder of duty cycle, if any
            sleep(max((1 / self.fps) - dt, 0))

    def render_target(
        self,
        img,
        keypoint,
        target,
        node_col,
        target_radius,
    ):
        dist_to_target = np.linalg.norm(target - keypoint)
        in_target = dist_to_target < target_radius
        if in_target:
            target_col = (0, 255, 0)
        else:
            target_col = (0, 0, 255)
            unit_vec = (target - keypoint) / dist_to_target
            pt2 = (
                unit_vec
                * min(self.max_arrow_length * self.render_scale, dist_to_target)
            ) + keypoint
            img = cv2.arrowedLine(
                img,
                pt1=(keypoint * self.render_scale).astype(int),
                pt2=(pt2 * self.render_scale).astype(int),
                color=node_col[::-1],
                thickness=int(self.arrow_thickness * self.render_scale),
                line_type=cv2.LINE_AA,
                shift=0,
                tipLength=0.1,
            )
        img = cv2.circle(
            img,
            (int(target[0] * self.render_scale), int(target[1] * self.render_scale)),
            radius=int(min(target_radius, dist_to_target) * self.render_scale),
            color=target_col,
            thickness=int(self.target_thickness * self.render_scale),
            lineType=cv2.LINE_AA,
        )
        return img, in_target

    def show_frame(self):
        """Display rendered frame."""
        if self.frame is None:
            return

        # Update rendered frame.
        cv2.imshow("frame", self.frame)

        # Wait for duty cycle and listen for key press.
        key = cv2.waitKey(int(1000 / self.fps))
        if key == 27 or key == 113:  # Esc or q
            # Quit.
            self.done = True
            self.thread.join()

        elif key >= (1 + 48) and key <= (9 + 48):
            # Set target pose
            key_num = key - 48  # 1-based
            self.target_poses.target_ind = (key_num - 1) % len(self.target_poses)

        elif key == 114:  # r
            # Reset target pose completeness
            self.target_poses.reset_completeness()

        elif key == 9:  # Tab
            # Cycle through target poses
            self.target_poses.next_target()

        elif key > 0:
            print("Pressed:", key)

    def run(self):
        """Run demo until stopped."""
        while not self.done:
            self.show_frame()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--camera", help="Camera index to use.", type=int, default=0
    )
    parser.add_argument("--cpu", help="Disable GPU inference.", action="store_true")
    parser.add_argument(
        "-t",
        "--target-poses",
        help="Path to target poses YAML file.",
        default="data/target_poses.yaml",
    )
    parser.add_argument(
        "--tolerance",
        help="Tolerance to the target pose locations. Higher is easier.",
        type=float,
        default=0.5,
    )
    args = parser.parse_args()

    if args.cpu:
        inference.use_cpu_only()
    else:
        inference.disable_gpu_preallocation()

    # Run until stopped.
    live_demo = LiveDemo(
        camera=args.camera,
        target_poses_path=args.target_poses,
        rel_target_radius=args.tolerance,
    )
    live_demo.run()
